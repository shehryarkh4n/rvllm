#!/usr/bin/env python3
"""Generate a minimal policy.json for a single decode bucket.

Hardcodes variant=0 (cutlass_fp8_gemm) for non-residual shapes and
variant=100 (cutlass_fp8_gemm_residual) for residual-epilogue shapes.
For a v3 bring-up run; a real autotune produces per-shape best variants.
"""
import json
import sys

SHAPES = {
    # Qwen2.5-7B: hidden=3584, heads=28, kv_heads=4, head_dim=128, intermediate=18944
    # For each bucket M, emit the 6 plans a layer needs + lm_head.
    "buckets": [1, 4, 8, 16, 32, 64, 128],
    "hidden": 3584,
    "q_dim": 3584,  # 28 * 128
    "kv_dim": 512,  # 4 * 128
    "intermediate": 18944,
    "vocab": 152064,
}


def main():
    if len(sys.argv) < 3:
        sys.exit(
            f"usage: {sys.argv[0]} <out_path> <revision> [nonres_variant=0] [res_variant=100]"
        )
    out_path = sys.argv[1]
    revision = sys.argv[2]
    nonres_variant = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    res_variant = int(sys.argv[4]) if len(sys.argv) >= 5 else 100

    hidden = SHAPES["hidden"]
    q_dim = SHAPES["q_dim"]
    kv_dim = SHAPES["kv_dim"]
    inter = SHAPES["intermediate"]

    entries = {}
    ws = 16 * 1024 * 1024  # conservative workspace budget

    qkv_rows = q_dim + 2 * kv_dim  # fused Q||K||V projection
    for m in SHAPES["buckets"]:
        # Non-residual: fused QKV, gate_up, lm_head
        for n, k in [
            (qkv_rows, hidden),     # fused QKV (4608 for Qwen2.5-7B)
            (2 * inter, hidden),    # gate||up
            (SHAPES["vocab"], hidden),  # lm_head
        ]:
            key = f"{m}_{n}_{k}_Fp8E4M3"
            entries[key] = {"variant": nonres_variant, "workspace_bytes": ws}
        # Residual: O, down — keyed with _res suffix so they don't
        # collide with a same-shape non-residual entry (e.g. Q vs O
        # both = hidden x hidden for Qwen).
        for n, k in [
            (hidden, q_dim),        # O
            (hidden, inter),        # down
        ]:
            key = f"{m}_{n}_{k}_Fp8E4M3_res"
            entries[key] = {"variant": res_variant, "workspace_bytes": ws}

    # Two variants in the catalog: non-residual id=0, residual id=100.
    # Schedule pairing Coop/Coop (matched).
    def v(id):
        return {
            "id": id,
            "tile": {"m": 128, "n": 128, "k": 128},
            "cluster": {"m": 1, "n": 1, "k": 1},
            "mainloop": "Coop",
            "epilogue": "Coop",
        }

    variants = [v(nonres_variant), v(res_variant)]

    policy = {
        "revision": revision,
        "arch": "sm_90",
        "variants": variants,
        "entries": entries,
    }
    with open(out_path, "w") as f:
        json.dump(policy, f, indent=2)
    print(f"wrote {out_path} with {len(entries)} entries")


if __name__ == "__main__":
    main()
