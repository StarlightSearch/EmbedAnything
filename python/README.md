# embed_anything Python extension

This crate is the Rust/PyO3 extension for the `embed_anything` Python package. It is **not** built with plain `cargo`.

## Building

From the **repository root** (not from this directory), use maturin so the extension links against your Python installation:

```bash
# From repository root
maturin develop    # build and install in the current environment (editable)
# or
maturin build      # build a wheel
```

Or install the package (maturin is the build backend in `pyproject.toml`):

```bash
pip install -e .
```

Do **not** run `cargo build` in this directory or `cargo build -p embed_anything_python` from the root—the linker will fail with undefined Python symbols because cargo does not link libpython.
