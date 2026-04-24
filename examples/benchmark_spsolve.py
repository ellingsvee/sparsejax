"""Benchmark sparsejax.spsolve vs. dense jnp.linalg.solve on CPU and GPU.

Test problem: 2D 5-point Laplacian on an N x N grid (so matrix size n = N*N).
The matrix is SPD, which lets us exercise the Cholesky / cuDSS paths.

Run:
    uv run python examples/benchmark_spsolve.py
    uv run python examples/benchmark_spsolve.py --sizes 32 64 96 --repeats 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from sparsejax import SparseMatrix, spsolve
from sparsejax import backends


def laplacian_2d(N: int):
    """Build the 2D 5-point Laplacian as a scipy CSR matrix."""
    n = N * N
    main = 4.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    # break the row-to-row neighbour every N entries
    off[np.arange(1, n) % N == 0] = 0.0
    far = -1.0 * np.ones(n - N)
    A = sp.diags([far, off, main, off, far], [-N, -1, 0, 1, N], format="coo")
    return A.tocoo()


def to_sparse_matrix(A_coo: sp.coo_matrix, device) -> SparseMatrix:
    data = jax.device_put(jnp.asarray(A_coo.data, dtype=jnp.float64), device)
    row = np.asarray(A_coo.row, dtype=np.int32)
    col = np.asarray(A_coo.col, dtype=np.int32)
    indices = np.stack([row, col], axis=0)
    return SparseMatrix(data=data, indices=indices, shape=A_coo.shape)


@dataclass
class Timing:
    label: str
    size: int
    mean_ms: float
    min_ms: float
    ok: bool


def bench(label: str, fn, repeats: int, size: int) -> Timing:
    # warmup (compile / first-call cost)
    try:
        out = fn()
        jax.block_until_ready(out)
    except Exception as e:
        print(f"  [skip] {label}: {e}")
        return Timing(label, size, float("nan"), float("nan"), ok=False)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1e3)
    return Timing(label, size, float(np.mean(times)), float(np.min(times)), True)


def run_size(N: int, repeats: int, cpu, gpu) -> list[Timing]:
    n = N * N
    A_coo = laplacian_2d(N)
    rng = np.random.default_rng(0)
    b_np = rng.standard_normal(n)

    results: list[Timing] = []

    # ---- CPU ----
    A_cpu = to_sparse_matrix(A_coo, cpu)
    b_cpu = jax.device_put(jnp.asarray(b_np), cpu)

    results.append(
        bench(
            "sparsejax spsolve (CPU / scipy)",
            lambda: spsolve(A_cpu, b_cpu, backend="scipy", method="lu"),
            repeats,
            n,
        )
    )

    if backends.is_available("cholmod"):
        results.append(
            bench(
                "sparsejax spsolve (CPU / cholmod, spd)",
                lambda: spsolve(A_cpu, b_cpu, backend="cholmod", spd=True),
                repeats,
                n,
            )
        )

    A_dense_cpu = A_cpu.to_dense()
    jitted_dense_cpu = jax.jit(jnp.linalg.solve, device=cpu)
    results.append(
        bench(
            "jnp.linalg.solve dense (CPU)",
            lambda: jitted_dense_cpu(A_dense_cpu, b_cpu),
            repeats,
            n,
        )
    )

    # ---- GPU ----
    if gpu is not None:
        A_gpu = to_sparse_matrix(A_coo, gpu)
        b_gpu = jax.device_put(jnp.asarray(b_np), gpu)

        if backends.is_available("cudss_ffi"):
            results.append(
                bench(
                    "sparsejax spsolve (GPU / cudss_ffi)",
                    lambda: spsolve(A_gpu, b_gpu, backend="cudss_ffi"),
                    repeats,
                    n,
                )
            )
        if backends.is_available("cudss"):
            results.append(
                bench(
                    "sparsejax spsolve (GPU / cudss)",
                    lambda: spsolve(A_gpu, b_gpu, backend="cudss"),
                    repeats,
                    n,
                )
            )

        A_dense_gpu = A_gpu.to_dense()
        jitted_dense_gpu = jax.jit(jnp.linalg.solve, device=gpu)
        results.append(
            bench(
                "jnp.linalg.solve dense (GPU)",
                lambda: jitted_dense_gpu(A_dense_gpu, b_gpu),
                repeats,
                n,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 96],
        help="Grid sizes N (matrix dimension is N*N).",
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        gpu = None

    print(f"JAX devices: cpu={cpu}, gpu={gpu}")
    print(f"Available backends: {backends.available_backends()}")
    print()

    header = f"{'size (n)':>10}  {'backend':<42}  {'mean (ms)':>10}  {'min (ms)':>10}"
    print(header)
    print("-" * len(header))

    for N in args.sizes:
        rows = run_size(N, args.repeats, cpu, gpu)
        for r in rows:
            if r.ok:
                print(
                    f"{r.size:>10}  {r.label:<42}  "
                    f"{r.mean_ms:>10.2f}  {r.min_ms:>10.2f}"
                )
            else:
                print(f"{r.size:>10}  {r.label:<42}  {'n/a':>10}  {'n/a':>10}")
        print()


if __name__ == "__main__":
    main()
