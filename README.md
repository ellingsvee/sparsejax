# sparsejax

Sparse operations for JAX, with optional CPU/GPU backends. All operations support automatic differentiation (although with some restrictions to ensure sparse VJPs) and JIT compilation (although with some different performance gains).

For CPU

- `cholmod`: Compute Cholesky factorization using CHOLMOD (through `scikit-sparse`). This can be used to solve liner systems and compute log-determinants.
- `scipy.sparse`: Installed by default.

For GPU

- `cuda`: Utilize the cuSPARSE and cuDSS libraries. Note that the `cudss_ffi` is highly recommended as it used the XLA FFI interface. However, this requires building the C++ backend from source.

Experimental:

- `cholmod_takahashi`: Compute the log-det gradient using the Takahashi method on the Cholesky factor. I have implemented this in Rust, but the performance is not competitive with f.ex. the PARDISO implementation. Use with caution!

## Development install

```bash
uv sync --all-extras
# Or f.ex. alternatively --extra cuda --extra cholmod
uv build
```

## Optional Rust backend

There is an experimental Rust which computes the log-det gradient from CHOLMOD's Cholesky factor using the Rust Takahashi implementation instead of solving against all sparse columns. It is used through the experimental `backend="cholmod_takahashi"` path.

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
