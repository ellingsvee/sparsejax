# sparsejax

Sparse operations for JAX, with optional CPU/GPU backends.

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
