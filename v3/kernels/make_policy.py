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
    # For each bucket M, emit the 6 plans a layer needs.
    "buckets": [1, 4, 8, 16, 32, 64, 128],
    "hidden": 3584,
    "q_dim": 3584,  # 28 * 128
    "kv_dim": 512,  # 4 * 128
    "intermediate": 18944,
}


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <out_path> <revision>")
    out_path = sys.argv[1]
    revision = sys.argv[2]

    hidden = SHAPES["hidden"]
    q_dim = SHAPES["q_dim"]
    kv_dim = SHAPES["kv_dim"]
    inter = SHAPES["intermediate"]

    entries = {}
    ws = 16 * 1024 * 1024  # conservative workspace budget

    for m in SHAPES["buckets"]:
        # Non-residual: Q, K, V, gate_up
        for n, k in [
            (q_dim, hidden),        # Q
            (kv_dim, hidden),       # K
            (kv_dim, hidden),       # V (same shape; duplicate key ok)
            (2 * inter, hidden),    # gate||up
        ]:
            key = f"{m}_{n}_{k}_Fp8E4M3"
            entries[key] = {"variant": 0, "workspace_bytes": ws}
        # Residual: O, down — keyed with _res suffix so they don't
        # collide with a same-shape non-residual entry (e.g. Q vs O
        # both = hidden x hidden for Qwen).
        for n, k in [
            (hidden, q_dim),        # O
            (hidden, inter),        # down
        ]:
            key = f"{m}_{n}_{k}_Fp8E4M3_res"
            entries[key] = {"variant": 100, "workspace_bytes": ws}

    # Two variants in the catalog: non-residual id=0, residual id=100.
    # Schedule pairing Coop/Coop (matched).
    variants = [
        {
            "id": 0,
            "tile": {"m": 128, "n": 128, "k": 128},
            "cluster": {"m": 1, "n": 1, "k": 1},
            "mainloop": "Coop",
            "epilogue": "Coop",
        },
        {
            "id": 100,
            "tile": {"m": 128, "n": 128, "k": 128},
            "cluster": {"m": 1, "n": 1, "k": 1},
            "mainloop": "Coop",
            "epilogue": "Coop",
        },
    ]

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
