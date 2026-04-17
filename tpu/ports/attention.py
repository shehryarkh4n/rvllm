"""Attention family.

CUDA: flash_attention.cu, flash_attention_3.cu (+ prefill, v3, sm90 wrapper),
      paged_attention.cu, split_kv_attention.cu
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._base import sds


def _gqa_expand(k, num_heads: int):
    # k: [*, num_kv_heads, head_dim] -> [*, num_heads, head_dim] via group repeat
    num_kv_heads = k.shape[-2]
    reps = num_heads // num_kv_heads
    return jnp.repeat(k, reps, axis=-2)


def _flash_reference(q, k, v, scale: float, causal: bool):
    # Device-agnostic reference: same math as splash_attention, used on
    # CPU/GPU (where Pallas TPU kernels are not available) and during
    # StableHLO emission.
    num_heads = q.shape[-2]
    k_exp = _gqa_expand(k, num_heads)
    v_exp = _gqa_expand(v, num_heads)

    scores = jnp.einsum("bshd,bthd->bhst", q, k_exp) * scale  # [B,H,Sq,Sk]
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        mask = jnp.tril(jnp.ones((sq, sk), dtype=jnp.bool_), k=sk - sq)
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp((scores - m).astype(jnp.float32))
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    p = p.astype(q.dtype)
    return jnp.einsum("bhst,bthd->bshd", p, v_exp)


def flash_attention(q, k, v, scale: float, causal: bool = True):
    # q: [B, Sq, Hq, D]
    # k,v: [B, Sk, Hkv, D]
    # On TPU, dispatches to jax.experimental.pallas splash_attention (the
    # TPU-native fused kernel — tiled Q·Kᵀ with online softmax running
    # entirely in VMEM, BF16 matmul on MXU). Falls back to an XLA-only
    # reference on CPU/GPU where the Pallas TPU kernel isn't available.
    if jax.default_backend() == "tpu":
        from jax.experimental.pallas.ops.tpu.splash_attention import (
            splash_attention_kernel as sak,
            splash_attention_mask as sam,
        )
        num_heads = q.shape[-2]
        # splash expects [B, H, S, D] (head-major); we hold [B, S, H, D]
        q_bhsd = jnp.transpose(q, (0, 2, 1, 3))
        # expand GQA so K/V have the same head count as Q — splash is MHA
        k_exp = jnp.transpose(_gqa_expand(k, num_heads), (0, 2, 1, 3))
        v_exp = jnp.transpose(_gqa_expand(v, num_heads), (0, 2, 1, 3))
        mask = sam.CausalMask(shape=(q.shape[1], k.shape[1])) if causal \
               else sam.FullMask(_shape=(q.shape[1], k.shape[1]))
        multi_head_mask = sam.MultiHeadMask(masks=[mask] * num_heads)
        kernel = sak.make_splash_mha(
            mask=multi_head_mask, head_shards=1, q_seq_shards=1
        )
        # splash_mha wants q scaled into its dot; pass scale by pre-multiplying.
        out_bhsd = jax.vmap(kernel)(q_bhsd * scale, k_exp, v_exp)
        return jnp.transpose(out_bhsd, (0, 2, 1, 3))
    return _flash_reference(q, k, v, scale, causal)


def flash_attention_trace_spec(shapes, dtype="bf16"):
    b = shapes["batch"]
    sq, sk = shapes["seq_q"], shapes["seq_k"]
    h, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    return (
        sds((b, sq, h, d), dtype),
        sds((b, sk, kh, d), dtype),
        sds((b, sk, kh, d), dtype),
    ), {"scale": 1.0 / (d ** 0.5), "causal": True}


def paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale: float):
    # q: [num_seqs, num_heads, head_dim]
    # k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    # block_tables: [num_seqs, max_ctx_blocks] int32
    # context_lens: [num_seqs] int32 (actual token count per seq)
    num_seqs, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    max_ctx_blocks = block_tables.shape[1]
    max_ctx = max_ctx_blocks * block_size

    # gather per-sequence K/V: [num_seqs, max_ctx, num_kv_heads, head_dim]
    k_gathered = k_cache[block_tables].reshape(num_seqs, max_ctx, num_kv_heads, head_dim)
    v_gathered = v_cache[block_tables].reshape(num_seqs, max_ctx, num_kv_heads, head_dim)

    k_exp = _gqa_expand(k_gathered, num_heads)
    v_exp = _gqa_expand(v_gathered, num_heads)

    # scores: [num_seqs, num_heads, max_ctx]
    scores = jnp.einsum("sHd,stHd->sHt", q, k_exp) * scale

    # mask out positions past context_len
    ar = jnp.arange(max_ctx)[None, :]  # [1, max_ctx]
    valid = ar < context_lens[:, None]   # [num_seqs, max_ctx]
    scores = jnp.where(valid[:, None, :], scores, jnp.finfo(scores.dtype).min)

    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp((scores - m).astype(jnp.float32))
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    p = p.astype(q.dtype)
    out = jnp.einsum("sHt,stHd->sHd", p, v_exp)
    return out


def paged_attention_trace_spec(shapes, dtype="bf16"):
    s = shapes["num_seqs"]
    h, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    nb, bs = shapes["num_blocks"], shapes["block_size"]
    mb = shapes["max_ctx_blocks"]
    return (
        sds((s, h, d), dtype),
        sds((nb, bs, kh, d), dtype),
        sds((nb, bs, kh, d), dtype),
        sds((s, mb), "i32"),
        sds((s,), "i32"),
    ), {"scale": 1.0 / (d ** 0.5)}


def split_kv_attention(
    q, k_cache, v_cache, block_tables, context_lens, split_table, scale: float
):
    # Split-KV is a GPU launch-tiling optimisation; the math is identical to
    # paged_attention. XLA:TPU retiles for the mesh, so the StableHLO port
    # collapses back to plain paged_attention. `split_table` is unused but
    # kept in the signature for binder parity with the CUDA call site.
    del split_table
    return paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale)


def split_kv_attention_trace_spec(shapes, dtype="bf16"):
    s = shapes["num_seqs"]
    h, kh, d = shapes["num_heads"], shapes["num_kv_heads"], shapes["head_dim"]
    nb, bs = shapes["num_blocks"], shapes["block_size"]
    ns = shapes["num_splits"]
    # reuse paged_attention layout for the first 5 args; add split_table last
    (q, k, v, bt, cl), _ = paged_attention_trace_spec(
        {**shapes, "max_ctx_blocks": shapes.get("max_ctx_blocks", 256)}, dtype
    )
    return (q, k, v, bt, cl, sds((s, ns), "i32")), {"scale": 1.0 / (d ** 0.5)}
