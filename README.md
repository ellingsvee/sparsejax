# sparsejax

Sparse operations for JAX, with optional CPU/GPU backends.

## Development install

```bash
uv sync --all-extras
uv build
```

## Optional Rust Takahashi backend

The qinv path is optional. It is used through the experimental `backend="cholmod_takahashi"` path, which computes the log-det gradient from CHOLMOD's Cholesky factor using the Rust Takahashi implementation instead of solving against all sparse columns.

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

### Using it from another project

If sparsejax is installed as a normal dependency, use the PEP 508 extras form:

```bash
uv add "sparsejax[cholmod,rust]"
```

That installs the Python dependencies and the `maturin` build tool, but with the current packaging setup it does **not** automatically build the Rust extension. The project still uses `scikit-build-core` as its wheel build backend; the `[tool.maturin]` section is used when `maturin` is run explicitly.

For an editable local dependency from another project:

```bash
uv add --editable "/path/to/sparsejax[cholmod,rust]"
cd /path/to/sparsejax
uv run --project /path/to/your-project maturin develop --release
```

After that, the consuming project's environment should be able to import `sparsejax.rust_backend`, and `backend="cholmod_takahashi"` can be used explicitly. The default `backend="auto"` path is unchanged and does not require the Rust extension.
