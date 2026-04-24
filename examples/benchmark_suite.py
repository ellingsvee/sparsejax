from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, cast

from sparsejax.backends import BackendName

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rich.console import Console
from rich.table import Table

try:
    from sksparse.cholmod import cho_factor as _sks_cho_factor

    _HAS_SKSPARSE = True
except ImportError:  # pragma: no cover
    _HAS_SKSPARSE = False
    _sks_cho_factor = None  # type: ignore[assignment]

from sparsejax import (
    SparseMatrix,
    backends,
    cholesky_solve,
    logdet,
    spmv,
    spadd,
    spsolve,
    spspmm,
    spdmm,
    cholesky_factor,
)


def laplacian_2d(N: int) -> sp.coo_matrix:
    """2D 5-point Laplacian on an N x N grid (SPD, size n = N*N)."""
    n = N * N
    main = 4.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    off[np.arange(1, n) % N == 0] = 0.0
    far = -1.0 * np.ones(n - N)
    return sp.diags([far, off, main, off, far], [-N, -1, 0, 1, N], format="coo")


def to_sparse_matrix(A_coo: sp.coo_matrix, device) -> SparseMatrix:
    data = jax.device_put(jnp.asarray(A_coo.data, dtype=jnp.float64), device)
    row = np.asarray(A_coo.row, dtype=np.int32)
    col = np.asarray(A_coo.col, dtype=np.int32)
    indices = np.stack([row, col], axis=0)
    return SparseMatrix(data=data, indices=indices, shape=A_coo.shape)


@dataclass
class Result:
    op: str
    backend: str
    device: str
    N: int
    n: int
    nnz: int
    mean_ms: float = float("nan")
    min_ms: float = float("nan")
    std_ms: float = float("nan")
    median_ms: float = float("nan")
    status: str = "ok"
    note: str = ""
    extras: dict = field(default_factory=dict)


def _sync(x):
    jax.block_until_ready(x)


def _short_err(e: BaseException, limit: int = 80) -> str:
    msg = str(e).strip().splitlines()[0]
    if len(msg) > limit:
        msg = msg[: limit - 1] + "…"
    return f"{type(e).__name__}: {msg}"


def time_fn(
    fn: Callable[[], object],
    *,
    warmup: int = 2,
    repeats: int = 5,
    sync: Callable[[object], None] = _sync,
) -> tuple[list[float], str, str]:
    """Time ``fn`` with warmup. Returns (times_ms, status, note)."""
    try:
        for _ in range(warmup):
            sync(fn())
    except Exception as e:  # noqa: BLE001
        return [], "skip", _short_err(e)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        sync(out)
        times.append((time.perf_counter() - t0) * 1e3)
    return times, "ok", ""


def _nosync(_x: object) -> None:
    pass


def _ref_row(
    op: str,
    backend_label: str,
    N: int,
    n: int,
    nnz: int,
    fn: Callable[[], object],
    repeats: int,
    warmup: int,
    extras: dict | None = None,
) -> Result:
    times, status, note = time_fn(fn, warmup=warmup, repeats=repeats, sync=_nosync)
    return Result(
        op=op,
        backend=backend_label,
        device="cpu",
        N=N,
        n=n,
        nnz=nnz,
        status=status,
        note=note,
        extras=extras or {},
        **summarize(times),
    )


def _sksparse_unavailable_row(op: str, N: int, n: int, nnz: int) -> Result:
    return Result(
        op=op,
        backend="cholmod(np)",
        device="cpu",
        N=N,
        n=n,
        nnz=nnz,
        status="skip",
        note="sksparse not installed",
    )


def bench_reference(op: str, A_coo: sp.coo_matrix, N: int, repeats: int, warmup: int) -> list[Result]:
    """Run scipy/sksparse on numpy directly (no JAX) so JAX overhead is visible."""
    n = A_coo.shape[0]
    nnz = A_coo.nnz
    A_csr = A_coo.tocsr()
    A_csc = A_coo.tocsc()
    rng = np.random.default_rng(0)
    rows: list[Result] = []

    if op == "spmv":
        x = rng.standard_normal(n)
        rows.append(_ref_row(op, "scipy(np)", N, n, nnz, lambda: A_csr @ x, repeats, warmup))
    elif op == "spadd":
        B_csr = A_coo.T.tocsr()
        rows.append(_ref_row(op, "scipy(np)", N, n, nnz, lambda: A_csr + B_csr, repeats, warmup))
    elif op == "spdmm":
        X = A_coo.toarray()
        rows.append(_ref_row(op, "scipy(np)", N, n, nnz, lambda: A_csr @ X, repeats, warmup))
    elif op == "spspmm":
        rows.append(_ref_row(op, "scipy(np)", N, n, nnz, lambda: A_csr @ A_csr, repeats, warmup))
    elif op == "spsolve":
        b = rng.standard_normal(n)
        rows.append(_ref_row(op, "scipy(np)", N, n, nnz, lambda: spla.spsolve(A_csc, b), repeats, warmup))
        if _HAS_SKSPARSE:
            rows.append(
                _ref_row(op, "cholmod(np)", N, n, nnz, lambda: _sks_cho_factor(A_csc).solve(b), repeats, warmup)
            )
        else:
            rows.append(_sksparse_unavailable_row(op, N, n, nnz))
    elif op == "cholesky_solve":
        if not _HAS_SKSPARSE:
            rows.append(_sksparse_unavailable_row(op, N, n, nnz))
        else:
            b = rng.standard_normal(n)
            rows.append(
                _ref_row(op, "cholmod(np)", N, n, nnz, lambda: _sks_cho_factor(A_csc).solve(b), repeats, warmup)
            )
    elif op == "cholesky_factor":
        if not _HAS_SKSPARSE:
            rows.append(_sksparse_unavailable_row(op, N, n, nnz))
        else:
            b = rng.standard_normal(n)
            factor_times, f_status, f_note = time_fn(
                lambda: _sks_cho_factor(A_csc),
                warmup=1,
                repeats=max(repeats // 2, 3),
                sync=_nosync,
            )
            if f_status != "ok":
                rows.append(
                    Result(
                        op=op, backend="cholmod(np)", device="cpu",
                        N=N, n=n, nnz=nnz, status=f_status, note=f_note,
                    )
                )
            else:
                factor = _sks_cho_factor(A_csc)
                solve_times, s_status, s_note = time_fn(
                    lambda: factor.solve(b), warmup=warmup, repeats=repeats, sync=_nosync
                )
                summary = summarize(solve_times)
                rows.append(
                    Result(
                        op=op, backend="cholmod(np)", device="cpu",
                        N=N, n=n, nnz=nnz, status=s_status, note=s_note,
                        extras={"factor_ms_median": float(statistics.median(factor_times))},
                        **summary,
                    )
                )
    elif op == "logdet":
        if not _HAS_SKSPARSE:
            rows.append(_sksparse_unavailable_row(op, N, n, nnz))
        else:
            rows.append(
                _ref_row(op, "cholmod(np)", N, n, nnz, lambda: _sks_cho_factor(A_csc).logdet(), repeats, warmup)
            )
    return rows


def summarize(times: list[float]) -> dict:
    if not times:
        return {}
    return {
        "mean_ms": float(statistics.fmean(times)),
        "min_ms": float(min(times)),
        "std_ms": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
        "median_ms": float(statistics.median(times)),
    }


CPU_BACKENDS_SOLVE = ["scipy", "cholmod"]
GPU_BACKENDS_SOLVE = ["cudss_ffi", "cudss"]


def bench_spmv(N: int, repeats: int, warmup: int, cpu, gpu) -> list[Result]:
    A_coo = laplacian_2d(N)
    n = N * N
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(n)
    results: list[Result] = bench_reference("spmv", A_coo, N, repeats, warmup)

    for label, device in (("cpu", cpu), ("gpu", gpu)):
        if device is None:
            continue
        A = to_sparse_matrix(A_coo, device)
        x = jax.device_put(jnp.asarray(x_np), device)
        # Don't wrap in jax.jit — SparseMatrix carries numpy `indices` in its
        # pytree aux, which breaks JIT cache-key hashing. Call eagerly like
        # the other ops do.
        times, status, note = time_fn(
            lambda: spmv(A, x), warmup=warmup, repeats=repeats
        )
        r = Result(
            op="spmv",
            backend="jax",
            device=label,
            N=N,
            n=n,
            nnz=A.nnz,
            status=status,
            note=note,
            **summarize(times),
        )
        results.append(r)
    return results


def bench_spadd(N: int, repeats: int, warmup: int, cpu, gpu) -> list[Result]:
    A_coo = laplacian_2d(N)
    B_coo = laplacian_2d(N).T
    n = N * N
    results: list[Result] = bench_reference("spadd", A_coo, N, repeats, warmup)

    for label, device in (("cpu", cpu), ("gpu", gpu)):
        if device is None:
            continue
        A = to_sparse_matrix(A_coo, device)
        B = to_sparse_matrix(B_coo, device)
        times, status, note = time_fn(
            lambda: spadd(A, B), warmup=warmup, repeats=repeats
        )
        r = Result(
            op="spadd",
            backend="jax",
            device=label,
            N=N,
            n=n,
            nnz=A.nnz,
            status=status,
            note=note,
            **summarize(times),
        )
        results.append(r)
    return results


def bench_spdmm(N: int, repeats: int, warmup: int, cpu, gpu) -> list[Result]:
    A_coo = laplacian_2d(N)
    n = N * N
    results: list[Result] = bench_reference("spdmm", A_coo, N, repeats, warmup)

    for label, device in (("cpu", cpu), ("gpu", gpu)):
        if device is None:
            continue
        A = to_sparse_matrix(A_coo, device)
        X = jax.device_put(jnp.asarray(A_coo.todense()), device)
        times, status, note = time_fn(
            lambda: spdmm(A, X), warmup=warmup, repeats=repeats
        )
        out_nnz = int(spspmm(A, A).nnz)
        r = Result(
            op="spdmm",
            backend="jax",
            device=label,
            N=N,
            n=n,
            nnz=A.nnz,
            status=status,
            note=note,
            extras={"out_nnz": out_nnz},
            **summarize(times),
        )
        results.append(r)
    return results


def bench_spspmm(N: int, repeats: int, warmup: int, cpu, gpu) -> list[Result]:
    A_coo = laplacian_2d(N)
    n = N * N
    results: list[Result] = bench_reference("spspmm", A_coo, N, repeats, warmup)

    for label, device in (("cpu", cpu), ("gpu", gpu)):
        if device is None:
            continue
        A = to_sparse_matrix(A_coo, device)
        times, status, note = time_fn(
            lambda: spspmm(A, A), warmup=warmup, repeats=repeats
        )
        out_nnz = int(spspmm(A, A).nnz)
        r = Result(
            op="spspmm",
            backend="jax",
            device=label,
            N=N,
            n=n,
            nnz=A.nnz,
            status=status,
            note=note,
            extras={"out_nnz": out_nnz},
            **summarize(times),
        )
        results.append(r)
    return results


def bench_spsolve(N: int, repeats: int, warmup: int, cpu, gpu) -> list[Result]:
    A_coo = laplacian_2d(N)
    n = N * N
    rng = np.random.default_rng(0)
    b_np = rng.standard_normal(n)
    results: list[Result] = bench_reference("spsolve", A_coo, N, repeats, warmup)

    for label, device, backend_list in (
        ("cpu", cpu, CPU_BACKENDS_SOLVE),
        ("gpu", gpu, GPU_BACKENDS_SOLVE),
    ):
        if device is None:
            continue
        A = to_sparse_matrix(A_coo, device)
        b = jax.device_put(jnp.asarray(b_np), device)

        for bk in backend_list:
            if not backends.is_available(cast(BackendName, bk)):
                results.append(
                    Result(
                        op="spsolve",
                        backend=bk,
                        device=label,
                        N=N,
                        n=n,
                        nnz=A.nnz,
                        status="skip",
                        note="backend unavailable",
                    )
                )
                continue
            times, status, note = time_fn(
                lambda bk=bk: spsolve(A, b, backend=bk),
                warmup=warmup,
                repeats=repeats,
            )
            results.append(
                Result(
                    op="spsolve",
                    backend=bk,
                    device=label,
                    N=N,
                    n=n,
                    nnz=A.nnz,
                    status=status,
                    note=note,
                    **summarize(times),
                )
            )
    return results


def bench_cholesky_solve(N: int, repeats: int, warmup: int, cpu, gpu) -> list[Result]:
    A_coo = laplacian_2d(N)
    n = N * N
    rng = np.random.default_rng(0)
    b_np = rng.standard_normal(n)
    results: list[Result] = bench_reference("cholesky_solve", A_coo, N, repeats, warmup)

    for label, device, backend_list in (
        ("cpu", cpu, CPU_BACKENDS_SOLVE),
        ("gpu", gpu, GPU_BACKENDS_SOLVE),
    ):
        if device is None:
            continue
        A = to_sparse_matrix(A_coo, device)
        b = jax.device_put(jnp.asarray(b_np), device)

        for bk in backend_list:
            if not backends.is_available(cast(BackendName, bk)):
                results.append(
                    Result(
                        op="cholesky_solve",
                        backend=bk,
                        device=label,
                        N=N,
                        n=n,
                        nnz=A.nnz,
                        status="skip",
                        note="backend unavailable",
                    )
                )
                continue
            times, status, note = time_fn(
                lambda bk=bk: cholesky_solve(A, b, backend=bk),
                warmup=warmup,
                repeats=repeats,
            )
            results.append(
                Result(
                    op="cholesky_solve",
                    backend=bk,
                    device=label,
                    N=N,
                    n=n,
                    nnz=A.nnz,
                    status=status,
                    note=note,
                    **summarize(times),
                )
            )
    return results


def bench_cholesky_factor(N: int, repeats: int, warmup: int, cpu, gpu) -> list[Result]:
    """Time the 'factor once, solve many' pattern — factor cost + per-RHS solve."""
    A_coo = laplacian_2d(N)
    n = N * N
    rng = np.random.default_rng(0)
    b_np = rng.standard_normal(n)
    results: list[Result] = bench_reference("cholesky_factor", A_coo, N, repeats, warmup)

    for label, device, backend_list in (
        ("cpu", cpu, CPU_BACKENDS_SOLVE),
        ("gpu", gpu, GPU_BACKENDS_SOLVE),
    ):
        if device is None:
            continue
        A = to_sparse_matrix(A_coo, device)
        b = jax.device_put(jnp.asarray(b_np), device)

        for bk in backend_list:
            if not backends.is_available(cast(BackendName, bk)):
                results.append(
                    Result(
                        op="cholesky_factor",
                        backend=bk,
                        device=label,
                        N=N,
                        n=n,
                        nnz=A.nnz,
                        status="skip",
                        note="backend unavailable",
                    )
                )
                continue

            # factor cost (single invocation, median of `repeats`)
            factor_times, f_status, f_note = time_fn(
                lambda bk=bk: cholesky_factor(A, backend=bk),
                warmup=1,
                repeats=max(repeats // 2, 3),
            )
            if f_status != "ok":
                results.append(
                    Result(
                        op="cholesky_factor",
                        backend=bk,
                        device=label,
                        N=N,
                        n=n,
                        nnz=A.nnz,
                        status=f_status,
                        note=f_note,
                    )
                )
                continue

            solve_fn = cholesky_factor(A, backend=bk)
            solve_times, s_status, s_note = time_fn(
                lambda: solve_fn(b), warmup=warmup, repeats=repeats
            )
            summary = summarize(solve_times)
            summary["extras"] = {
                "factor_ms_median": float(statistics.median(factor_times))
            }
            results.append(
                Result(
                    op="cholesky_factor",
                    backend=bk,
                    device=label,
                    N=N,
                    n=n,
                    nnz=A.nnz,
                    status=s_status,
                    note=s_note,
                    extras=summary.pop("extras"),
                    **summary,
                )
            )
    return results


def bench_logdet(N: int, repeats: int, warmup: int, cpu, gpu) -> list[Result]:
    A_coo = laplacian_2d(N)
    n = N * N
    results: list[Result] = bench_reference("logdet", A_coo, N, repeats, warmup)

    for label, device, backend_list in (
        ("cpu", cpu, CPU_BACKENDS_SOLVE),
        ("gpu", gpu, GPU_BACKENDS_SOLVE),
    ):
        if device is None:
            continue
        A = to_sparse_matrix(A_coo, device)
        for bk in backend_list:
            if not backends.is_available(cast(BackendName, bk)):
                results.append(
                    Result(
                        op="logdet",
                        backend=bk,
                        device=label,
                        N=N,
                        n=n,
                        nnz=A.nnz,
                        status="skip",
                        note="backend unavailable",
                    )
                )
                continue
            times, status, note = time_fn(
                lambda bk=bk: logdet(A, backend=bk),
                warmup=warmup,
                repeats=repeats,
            )
            results.append(
                Result(
                    op="logdet",
                    backend=bk,
                    device=label,
                    N=N,
                    n=n,
                    nnz=A.nnz,
                    status=status,
                    note=note,
                    **summarize(times),
                )
            )
    return results


OPS = {
    "spadd": bench_spadd,
    "spmv": bench_spmv,
    "spdmm": bench_spdmm,
    "spspmm": bench_spspmm,
    "spsolve": bench_spsolve,
    "cholesky_solve": bench_cholesky_solve,
    "cholesky_factor": bench_cholesky_factor,
    "logdet": bench_logdet,
}


def render_table(console: Console, op: str, results: list[Result]) -> None:
    table = Table(
        title=f"[bold]{op}[/bold]",
        header_style="bold cyan",
        title_style="bold white",
        show_lines=False,
        expand=False,
        pad_edge=False,
    )
    table.add_column("N", justify="right", no_wrap=True)
    table.add_column("n", justify="right", no_wrap=True)
    table.add_column("nnz", justify="right", no_wrap=True)
    table.add_column("dev", no_wrap=True)
    table.add_column("backend", no_wrap=True)
    table.add_column("mean ms", justify="right", no_wrap=True)
    table.add_column("min ms", justify="right", no_wrap=True)
    table.add_column("std ms", justify="right", no_wrap=True)
    table.add_column("rel", justify="right", no_wrap=True)
    table.add_column("notes", overflow="fold", max_width=40)

    # Compute relative speedup vs best (fastest mean) per N.
    by_N: dict[int, list[Result]] = {}
    for r in results:
        by_N.setdefault(r.N, []).append(r)

    best_mean: dict[int, float] = {}
    for N, rs in by_N.items():
        oks = [r for r in rs if r.status == "ok"]
        if oks:
            best_mean[N] = min(r.mean_ms for r in oks)

    def fmt(x: float) -> str:
        if x != x:  # NaN
            return "-"
        return f"{x:8.3f}"

    prev_N = None
    for r in sorted(results, key=lambda r: (r.N, r.device, r.backend)):
        if prev_N is not None and r.N != prev_N:
            table.add_section()
        prev_N = r.N

        if r.status == "ok":
            ratio = r.mean_ms / best_mean[r.N]
            rel = f"{ratio:5.2f}x"
            rel_style = (
                "green" if ratio <= 1.05 else ("yellow" if ratio <= 3 else "red")
            )
            rel_cell = f"[{rel_style}]{rel}[/{rel_style}]"
            note = r.note
            if op == "cholesky_factor" and "factor_ms_median" in r.extras:
                note = (note + " " if note else "") + (
                    f"factor={r.extras['factor_ms_median']:.1f} ms"
                )
            if op == "spspmm" and "out_nnz" in r.extras:
                note = (note + " " if note else "") + f"out_nnz={r.extras['out_nnz']}"
            table.add_row(
                str(r.N),
                f"{r.n:,}",
                f"{r.nnz:,}",
                r.device,
                r.backend,
                fmt(r.mean_ms),
                fmt(r.min_ms),
                fmt(r.std_ms),
                rel_cell,
                note,
            )
        else:
            table.add_row(
                str(r.N),
                f"{r.n:,}",
                f"{r.nnz:,}",
                r.device,
                r.backend,
                "-",
                "-",
                "-",
                "-",
                f"[dim]{r.status}: {r.note}[/dim]",
            )
    console.print(table)
    console.print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 96])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=list(OPS.keys()),
        default=list(OPS.keys()),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Use a generous width when stdout isn't a tty (e.g. piping to a file)
    # so rich doesn't fold columns into ellipses.
    console = Console(width=140) if not Console().is_terminal else Console()

    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        gpu = None

    meta = Table(title="[bold]Environment[/bold]", show_header=False, box=None)
    meta.add_row("JAX CPU device", str(cpu))
    meta.add_row("JAX GPU device", str(gpu) if gpu is not None else "[dim]none[/dim]")
    meta.add_row("Available backends", ", ".join(backends.available_backends()))
    meta.add_row("Sizes (N)", ", ".join(str(s) for s in args.sizes))
    meta.add_row("Warmup / repeats", f"{args.warmup} / {args.repeats}")
    console.print(meta)
    console.print()

    for op in args.ops:
        console.print(f"[bold white on blue] running {op} [/]")
        rows: list[Result] = []
        for N in args.sizes:
            rows.extend(OPS[op](N, args.repeats, args.warmup, cpu, gpu))
        render_table(console, op, rows)


if __name__ == "__main__":
    main()
