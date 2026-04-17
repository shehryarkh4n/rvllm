"""End-to-end tok/s + TTFT bench on TPU via Flax/JAX.

Pure JAX — no torch, no vLLM. Loads HF model via transformers Flax backend
(from_pt=True for PyTorch→Flax weight conversion). Runs generate() on TPU.
"""

from __future__ import annotations
import argparse, json, os, sys, time

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seq-out", type=int, default=64)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--out", default="/tmp/tpu_e2e_bench.json")
    args = ap.parse_args()

    print(f"jax {jax.__version__} backend={jax.default_backend()} "
          f"devices={jax.devices()}", flush=True)

    from transformers import AutoTokenizer

    # Try Flax model first, fall back to from_pt conversion
    model_cls = None
    try:
        from transformers import FlaxAutoModelForCausalLM
        model_cls = FlaxAutoModelForCausalLM
    except ImportError:
        print("FlaxAutoModelForCausalLM not available", flush=True)
        sys.exit(1)

    print(f"loading {args.model} (from_pt=True)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = model_cls.from_pretrained(args.model, from_pt=True,
                                          dtype=jnp.bfloat16)
    except Exception as exc:
        print(f"model load failed: {exc}", flush=True)
        # Try a known Flax-compatible model
        fallback = "meta-llama/Llama-3.2-1B"
        print(f"falling back to {fallback}", flush=True)
        args.model = fallback
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = model_cls.from_pretrained(fallback, from_pt=True,
                                          dtype=jnp.bfloat16)

    print("model loaded", flush=True)

    prompt = "The meaning of life is"
    results = {"model": args.model, "device": str(jax.devices()[0]), "runs": []}

    for n in [int(x) for x in args.batches.split(",")]:
        print(f"bench N={n}...", flush=True)
        inputs = tokenizer([prompt] * n, return_tensors="np", padding=True)
        input_ids = jnp.array(inputs["input_ids"])
        attn_mask = jnp.array(inputs["attention_mask"])

        # warmup
        try:
            _ = model.generate(input_ids, attention_mask=attn_mask,
                               max_new_tokens=2, do_sample=False)
        except Exception as exc:
            print(f"N={n}: warmup failed: {exc}", flush=True)
            results["runs"].append({"n": n, "error": str(exc)})
            continue

        # TTFT: time to generate 1 token
        t0 = time.perf_counter()
        _ = model.generate(input_ids, attention_mask=attn_mask,
                           max_new_tokens=1, do_sample=False)
        ttft = (time.perf_counter() - t0) * 1000

        # throughput: generate seq_out tokens
        t0 = time.perf_counter()
        out = model.generate(input_ids, attention_mask=attn_mask,
                             max_new_tokens=args.seq_out, do_sample=False)
        wall = time.perf_counter() - t0
        gen_tokens = out.sequences.shape[1] - input_ids.shape[1]
        total_out = gen_tokens * n
        toks = total_out / wall

        print(f"N={n:4d}  toks={toks:10.2f}/s  ttft={ttft:8.2f}ms  "
              f"wall={wall:6.2f}s  out={total_out}", flush=True)
        results["runs"].append({
            "n": n, "toks": round(toks, 2), "ttft_ms": round(ttft, 2),
            "wall_s": round(wall, 2), "out_tokens": total_out,
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
