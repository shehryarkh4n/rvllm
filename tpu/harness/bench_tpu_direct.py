"""Direct tok/s + TTFT bench on TPU via transformers + torch_xla.

No vLLM, no Pallas, no ZMQ. Just model.generate() on XLA device.
Reports the same metrics as bench.html: output tok/s + TTFT at each N.

Caveat vs H100 bench: no paged KV cache, no continuous batching — naive
batch decode. Numbers represent raw TPU throughput, not a production server.
"""

from __future__ import annotations
import argparse, json, os, sys, time

os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer


def bench_batch(model, tokenizer, device, n: int, seq_in: int, seq_out: int):
    prompt = "The meaning of life is"
    inputs = tokenizer([prompt] * n, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    # warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    xm.mark_step()

    # timed run
    xm.mark_step()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=seq_out, do_sample=False)
    xm.mark_step()
    wall = time.perf_counter() - t0

    gen_tokens = out.shape[1] - input_len
    total_out = gen_tokens * n
    toks = total_out / wall

    # TTFT approximation: time to first token = time for 1-token generate
    xm.mark_step()
    t1 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    xm.mark_step()
    ttft = (time.perf_counter() - t1) * 1000

    return toks, ttft, wall, total_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seq-in", type=int, default=16)
    ap.add_argument("--seq-out", type=int, default=512)
    ap.add_argument("--batches", default="1,8,16,64,128")
    ap.add_argument("--out", default="/tmp/tpu_e2e_bench.json")
    args = ap.parse_args()

    device = xm.xla_device()
    print(f"device: {device}", flush=True)

    print(f"loading {args.model} bf16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print("model loaded", flush=True)

    results = {"model": args.model, "device": str(device), "runs": []}
    for n in [int(x) for x in args.batches.split(",")]:
        try:
            toks, ttft, wall, total_out = bench_batch(
                model, tokenizer, device, n, args.seq_in, args.seq_out
            )
            print(f"N={n:4d}  toks={toks:10.2f}/s  ttft={ttft:8.2f}ms  "
                  f"wall={wall:6.2f}s  out={total_out}", flush=True)
            results["runs"].append({
                "n": n, "toks": round(toks, 2), "ttft_ms": round(ttft, 2),
                "wall_s": round(wall, 2), "out_tokens": total_out,
            })
        except Exception as exc:
            print(f"N={n}: FAILED {exc}", flush=True)
            results["runs"].append({"n": n, "error": str(exc)})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
