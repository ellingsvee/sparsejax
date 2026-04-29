# sparsejax

Sparse operations for JAX, with optional CPU/GPU backends. All operations support automatic differentiation (although with some restrictions to ensure sparse VJPs) and JIT compilation (although with some different performance gains).

For CPU

- `cholmod`: Compute Cholesky factorization using CHOLMOD (through `scikit-sparse`). This can be used to solve liner systems and compute log-determinants.
- `cholmod_takahashi`: Uses CHOLMOD for factorization and a Rust implementation of the Takahashi selected-inverse recurrence for log-det gradients. This is preferred automatically for SPD CPU matrices when the Rust extension is available.
- `scipy.sparse`: Installed by default.

For GPU

- `cuda`: Utilize the cuSPARSE and cuDSS libraries. Note that the `cudss_ffi` is highly recommended as it used the XLA FFI interface. However, this requires building the C++ backend from source.

## Development install

```bash
uv sync --all-extras
# Or f.ex. alternatively --extra cuda --extra cholmod
uv build
```

## Optional Rust backend

There is an optional Rust backend which computes the log-det gradient from CHOLMOD's Cholesky factor using the Takahashi selected-inverse recurrence instead of solving against all sparse columns. It is used through `backend="cholmod_takahashi"` and is selected by `backend="auto"` for SPD CPU matrices when available.

This backend needs both:

- `cholmod`: installs `scikit-sparse`, which provides the CHOLMOD factorization.
- `rust`: installs `maturin`, which is used to build the Rust extension.

For a local sparsejax checkout:

```bash
uv sync --extra cholmod --extra rust
uv run maturin develop --release
```

You can verify that the extension is importable with:

```bash
uv run python -c "import sparsejax.rust_backend; print('ok')"
```
