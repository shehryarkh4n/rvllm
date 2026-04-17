"""Perplexity measurement against a vLLM OpenAI endpoint.

Mirrors the rvllm_ppl methodology (v3/crates/rvllm-bench/src/bin/rvllm_ppl.rs):
  - tokenize the corpus
  - split into fixed-length chunks (default 128 tokens)
  - for each chunk, sum NLL from logprobs
  - perplexity = exp(total_nll / total_tokens)

Uses vLLM's OpenAI completions endpoint with echo=True + logprobs=1 so the
server returns a logprob per prompt token. Output JSON matches rvllm_ppl's
shape so the two results drop into the same row on bench.html.
"""

from __future__ import annotations
import argparse, json, math, sys, time
from urllib import request


WIKITEXT_SAMPLE_PATH = "/tmp/wikitext2_test.txt"


def load_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def tokenize_via_vllm(url: str, model: str, text: str):
    req = request.Request(
        url + "/v1/tokenize",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    return body.get("tokens", [])


def detok_prompt(url: str, model: str, token_ids):
    # Some vLLM builds don't have a detokenize endpoint; we just pass the
    # token string back as a JSON array — vLLM accepts prompt=[ids].
    return token_ids


def nll_for_chunk(url: str, model: str, token_ids) -> tuple[float, int]:
    # prompt=token_ids with echo=True + logprobs=1 returns logprobs for each
    # of the N prompt positions. Position 0 has no prior context, so skip it.
    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
        "temperature": 0.0,
    }
    req = request.Request(
        url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    lp = body["choices"][0]["logprobs"]["token_logprobs"]
    # first entry is always null (no context). drop it.
    vals = [x for x in lp if x is not None]
    nll = -sum(vals)
    return nll, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default=WIKITEXT_SAMPLE_PATH)
    ap.add_argument("--chunk-len", type=int, default=128)
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="0 = all chunks")
    ap.add_argument("--out", default="/tmp/tpu_ppl.json")
    args = ap.parse_args()

    text = load_text(args.text)
    tokens = tokenize_via_vllm(args.url, args.model, text)
    print(f"total tokens: {len(tokens)}", file=sys.stderr)

    chunks = [tokens[i:i + args.chunk_len]
              for i in range(0, len(tokens), args.chunk_len)]
    chunks = [c for c in chunks if len(c) == args.chunk_len]
    if args.max_chunks > 0:
        chunks = chunks[:args.max_chunks]
    print(f"evaluating {len(chunks)} chunks of {args.chunk_len} tokens",
          file=sys.stderr)

    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()
    for ci, chunk in enumerate(chunks):
        try:
            nll, n = nll_for_chunk(args.url, args.model, chunk)
        except Exception as exc:
            print(f"chunk {ci}: error {exc}", file=sys.stderr)
            continue
        total_nll += nll
        total_tokens += n
        running_ppl = math.exp(total_nll / total_tokens) if total_tokens else 0
        chunk_ppl = math.exp(nll / n) if n else 0
        dt = time.perf_counter() - t0
        print(f"chunk {ci+1}/{len(chunks)}: chunk_ppl={chunk_ppl:.4f} "
              f"running_ppl={running_ppl:.4f} ({total_tokens/dt:.1f} tok/s)",
              file=sys.stderr)

    ppl = math.exp(total_nll / total_tokens) if total_tokens else float("nan")
    elapsed = time.perf_counter() - t0
    result = {
        "perplexity": round(ppl, 4),
        "tokens": total_tokens,
        "chunk_len": args.chunk_len,
        "elapsed_s": round(elapsed, 1),
    }
    print(json.dumps(result))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
