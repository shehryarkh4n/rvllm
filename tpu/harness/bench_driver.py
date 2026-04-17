"""Minimal tok/s + TTFT bench driver.

Hits the vLLM OpenAI-compatible /v1/completions endpoint with N concurrent
requests, each requesting `seq_in` prompt tokens + `seq_out` output tokens
with ignore_eos. Reports:
  - toks = total output tokens / wall-time window (output throughput)
  - ttft = mean time from request send -> first streamed token

Mirrors vLLM's own `vllm bench serve` methodology — same metric names so
the numbers drop straight into bench.html's existing N columns.
"""

from __future__ import annotations
import argparse, asyncio, json, random, string, sys, time
import aiohttp


async def one_request(session, url, model, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    out_tokens = 0
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    async with session.post(url, json=body) as resp:
        async for raw in resp.content:
            line = raw.decode(errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            text = chunk.get("choices", [{}])[0].get("text", "")
            if text:
                # rough token count: vllm sends one token per chunk in stream
                out_tokens += 1
    wall = time.perf_counter() - t0
    return ttft, out_tokens, wall


def mkprompt(n_tokens: int) -> str:
    # ~1.3 chars/token for random garbage -> pad generously
    chars = string.ascii_lowercase + " "
    return "".join(random.choices(chars, k=n_tokens * 4))


async def run_batch(url, model, n, seq_in, seq_out):
    prompts = [mkprompt(seq_in) for _ in range(n)]
    conn = aiohttp.TCPConnector(limit=n)
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        t0 = time.perf_counter()
        tasks = [one_request(s, url, model, p, seq_out) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - t0
    ttfts = []
    total_out = 0
    for r in results:
        if isinstance(r, Exception):
            print("   err:", r, file=sys.stderr)
            continue
        ttft, ntok, _ = r
        if ttft is not None:
            ttfts.append(ttft)
        total_out += ntok
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else float("nan")
    toks = total_out / wall if wall > 0 else 0.0
    return toks, mean_ttft, wall, total_out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-in", type=int, default=16)
    ap.add_argument("--seq-out", type=int, default=512)
    ap.add_argument("--batches", default="1,8,16,64,128,256,512")
    ap.add_argument("--out", default="/tmp/tpu_e2e_bench.json")
    args = ap.parse_args()

    out = {"model": args.model, "runs": []}
    for n in [int(x) for x in args.batches.split(",")]:
        for phase in ("cold", "hot"):
            toks, ttft, wall, total_out = await run_batch(
                args.url, args.model, n, args.seq_in, args.seq_out
            )
            print(f"N={n:4d} {phase:4s}  toks={toks:10.2f}/s  ttft={ttft:8.2f} ms  "
                  f"wall={wall:6.2f}s  out_tok={total_out}")
            out["runs"].append({"n": n, "phase": phase,
                                "toks": toks, "ttft_ms": ttft,
                                "wall_s": wall, "out_tokens": total_out})
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
