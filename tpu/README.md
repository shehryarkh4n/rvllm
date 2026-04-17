# rvLLM → TPU (StableHLO) Port Suite

Conversion scaffold for taking every rvLLM CUDA kernel under `../kernels/` and
emitting a StableHLO `.mlir` module that XLA/TPU can ingest. Front-end is JAX
(`jax.jit(fn).lower(...).compiler_ir(dialect="stablehlo")`), which is the
sanctioned path to StableHLO.

This suite is **batch-ready, not yet executed**. The intent is:

1. `manifest.toml` enumerates every CUDA kernel → port file → status.
2. `ports/*.py` contain JAX reference implementations matching CUDA semantics.
3. `harness/emit_all.py` walks the manifest and writes `out/<name>.mlir`.
4. `harness/verify_all.py` (future) compares JAX output against CUDA kernel
   output (driven off `../kernels/` + rvllm runtime harness).

## Layout

```
tpu/
  manifest.toml         canonical kernel list, status, family, shape template
  pyproject.toml        jax/jaxlib[tpu] deps
  Makefile              port / emit / status / verify targets
  SPEC.md               CUDA-op → StableHLO-op mapping, lowering notes
  ports/                one module per kernel family
  harness/              batch emitter + (future) numerical verifier
  out/                  emitted .mlir (gitignored)
```

## One-shot batch emit (when ready)

```
make install        # jax + jaxlib, numpy
make port           # run emit_all.py, fills out/*.mlir
make status         # prints per-kernel port + emit state from manifest
```

Do not run against a TPU yet. Emission is host-only.

## Porting principles

- **No fallbacks.** If a CUDA kernel has semantics JAX cannot express
  losslessly (e.g. FP8 E4M3 scaled GEMM, async TMA, wgmma), the port raises
  `NotImplementedError` with a `# TODO:` marker. No silent FP32 stand-ins.
- **Dtype parity.** F16 variants collapse into the same port function —
  dtype is a parameter, not a separate file.
- **Shape parity.** Port signatures mirror CUDA launch + buffer layout so
  the rvLLM executor binds them identically.
- **One op per jit.** Every port is a single `@jax.jit`-compatible pure
  function; no Python-side control flow after trace.
