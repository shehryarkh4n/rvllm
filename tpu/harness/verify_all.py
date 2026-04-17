"""Numerical parity verifier (skeleton — not run yet).

For each `impl` entry in manifest.toml, the verifier will:
  1. build random inputs from the trace spec
  2. run the JAX port on CPU
  3. run the matching CUDA kernel via the rvllm runtime (ctypes/FFI into
     ../kernels/*.so) against the same inputs
  4. compare with per-dtype tolerances (bf16: atol=5e-2 rtol=5e-2,
     f32: atol=1e-5 rtol=1e-5)

Step 3 requires the rvllm runtime shim. The suite ships this scaffold so
that once the shim lands we can flip on verification in the same batch.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("verify_all: not wired yet — see harness/verify_all.py docstring.")
    print("Next step: implement a ctypes loader pointing at ../kernels/*.so and")
    print("feed identical inputs to the JAX port and the CUDA kernel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
