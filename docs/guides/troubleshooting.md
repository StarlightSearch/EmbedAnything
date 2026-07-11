# Troubleshooting

## macOS + Metal: `use of undeclared identifier 'bfloat'`

### Symptom

When running a **Candle** model (e.g. `from_pretrained_hf`, such as EmbeddingGemma /
`gemma3`) with Metal acceleration on macOS, you get an error like:

```text
ValueError: Metal error Error while loading library: ...
program_source:523:35: error: use of undeclared identifier 'bfloat'; did you mean 'float'?
instantiate_gemv_blocks(bfloat16, bfloat)
...
error: duplicate explicit instantiation of 'gemv<float, ...>'
```

Two things are usually true when this happens:

- It fails **only through Python** (the `pip`-installed wheel / a `pyo3` build), while the
  same code built as a native Rust binary (`cargo run --features metal`) works fine.
- It fails **only for some models** — typically ones that trigger a matrix×vector
  (GEMV) kernel, e.g. embedding a **single query** with a model that has a dense
  projection head. ONNX (`from_pretrained_onnx`) and `model2vec` models never hit it.

### Root cause

Candle compiles its Metal kernels at runtime (JIT). Its `gemv.metal` references the
`bfloat` type, which only exists when the Metal Shading Language version is **≥ 3.1**
(macOS 14+). Metal derives the *default* MSL version from the **host process's** linked
macOS SDK — i.e. the **Python interpreter**, not EmbedAnything.

Many Python distributions are built against an old SDK. For example, Miniconda/Anaconda
Pythons often report macOS 11–12:

```bash
otool -l "$(python -c 'import sys; print(sys.executable)')" | grep -A3 LC_BUILD_VERSION
# ...
#     minos 11.1
#       sdk 11.1
```

With an old `minos`/`sdk` (< 14), Metal defaults to an MSL version where `bfloat` is
undeclared, so the kernel fails to compile — regardless of the dtype you actually use.
A native `cargo` binary works because it is linked against your current (modern) SDK.

### Fix (choose one)

Any Python whose Mach-O `minos` is **≥ 14** makes the JIT use MSL ≥ 3.1 and resolves the
error. **No rebuild of EmbedAnything is needed — the published wheel works as-is.**

**Option A — Use a modern-SDK Python (recommended, simplest)**

Homebrew Pythons are built on your machine against the current SDK:

```bash
brew install python@3.13
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install embed-anything
```

Verify the interpreter is modern:

```bash
otool -l "$(python -c 'import sys; print(sys.executable)')" | grep -A3 LC_BUILD_VERSION
#     minos 15.0   (or higher)
```

**Option B — Relink an existing (e.g. conda) interpreter**

If you must keep a conda environment, rewrite its interpreter's build version to 14.0 and
re-sign it. This edits the interpreter binary only (not your project) and is reversible:

```bash
CPY="$(python -c 'import sys; print(sys.executable)')"
cp -p "$CPY" "$CPY.bak"                                        # backup
vtool -set-build-version macos 14.0 14.0 -replace -output "$CPY.new" "$CPY"
codesign -f -s - "$CPY.new"                                    # ad-hoc re-sign
mv "$CPY.new" "$CPY"
```

Note: if conda later reinstalls/updates Python, the interpreter reverts to its old SDK
and the error returns — re-run the commands above on the new binary. To revert manually,
restore the `.bak` copy.

### What does *not* work

- Setting `MACOSX_DEPLOYMENT_TARGET` when building the wheel. Metal reads the **host
  interpreter's** SDK, not the extension module's, so bumping the extension's deployment
  target changes the wheel tag but not the runtime behavior.
- Selecting a "macOS 14" target in your build. Same reason.

### CPU fallback

If you don't need GPU acceleration, install the CPU wheel (`pip install embed-anything`)
and it will run on CPU without touching Metal.
