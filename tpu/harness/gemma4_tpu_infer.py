#!/usr/bin/env python3
"""Gemma 4 31B inference on TPU v6e-4 via JAX SPMD (TP=4).

Flat scan over all 60 layers. Weights padded to max shape so one scan
body handles both sliding and global layers via jax.lax.cond.

Usage:
    python3 gemma4_tpu_infer.py --model-dir /path/to/gemma-4-31B-it \
        --max-tokens 32 --prompt "2,3257"
"""
import argparse, json, os, struct, sys, time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import ml_dtypes

H      = 5376
NH     = 32
INTER  = 21504
VOCAB  = 262144
NL     = 60
WINDOW = 1024
SOFTCAP_VAL = 30.0
EPS    = 1e-6
B      = 1

# Max shapes across layer types (pad smaller to these)
MAX_Q  = 16384   # max(8192, 16384)
MAX_KV = 4096    # max(4096, 2048)
MAX_O  = 16384   # max(8192, 16384)
MAX_NORM_HD = 512  # max(256, 512)

# Sliding params
S_Q, S_KV, S_HD, S_KVH = 8192, 4096, 256, 16
S_GQA = NH // S_KVH  # 2
# Global params
G_Q, G_KV, G_HD, G_KVH = 16384, 2048, 512, 4
G_GQA = NH // G_KVH  # 8

LAYER_IS_GLOBAL = np.array([1 if (i+1) % 6 == 0 else 0 for i in range(NL)], dtype=np.int32)

def make_mesh():
    devs = jax.devices()
    return Mesh(np.array(devs[:4]), ('tp',))

def rms_norm(x, g):
    x32 = x.astype(jnp.float32)
    return (x * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + EPS).astype(x.dtype)) * g

def head_norm(h, g):
    h32 = h.astype(jnp.float32)
    return (h * jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + EPS).astype(h.dtype)) * g

def head_norm_noscale(h):
    h32 = h.astype(jnp.float32)
    return (h * jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + EPS).astype(h.dtype))

def rope(x, cos, sin, rot_dim):
    half = rot_dim // 2
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    x0, x1 = xr[..., :half], xr[..., half:]
    rotated = jnp.concatenate([
        x0 * cos.astype(x.dtype) - x1 * sin.astype(x.dtype),
        x0 * sin.astype(x.dtype) + x1 * cos.astype(x.dtype),
    ], axis=-1)
    return jnp.concatenate([rotated, xp], axis=-1)

def precompute_rope(theta, rot_dim, max_pos):
    half = rot_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    angles = np.outer(np.arange(max_pos, dtype=np.float32), freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

# ── attention branches ──

def _sliding_attn(q_flat, k_flat, v_flat, qn, kn, cos_s, sin_s, kc, vc, pos, ctx, max_ctx):
    q = head_norm(q_flat[:, :S_Q].reshape(B, NH, S_HD), qn[:S_HD])
    k = head_norm(k_flat[:, :S_KV].reshape(B, S_KVH, S_HD), kn[:S_HD])
    v = head_norm_noscale(v_flat[:, :S_KV].reshape(B, S_KVH, S_HD))
    c = cos_s[pos][None, None, :]
    s = sin_s[pos][None, None, :]
    q = rope(q, c, s, S_HD)
    k = rope(k, c, s, S_HD)
    kc = kc.at[pos].set(k.reshape(B, S_KV)[0].astype(kc.dtype))
    vc = vc.at[pos].set(v.reshape(B, S_KV)[0].astype(vc.dtype))
    k_ctx = kc[:max_ctx].reshape(max_ctx, S_KVH, S_HD)
    v_ctx = vc[:max_ctx].reshape(max_ctx, S_KVH, S_HD)
    q_g = q.reshape(B, S_KVH, S_GQA, S_HD)
    sc = jnp.einsum('bghd,tgd->bght', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))
    # Gemma 4 uses scaling=1.0 (QK-norm handles magnitude)
    # sc = sc / jnp.sqrt(jnp.float32(S_HD))
    t = jnp.arange(max_ctx)
    valid = (t < ctx) & (t >= jnp.maximum(0, pos - WINDOW + 1))
    sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('bght,tgd->bghd', p, v_ctx).reshape(B, S_Q)
    return out, kc, vc

def _global_attn(q_flat, k_flat, v_flat, qn, kn, cos_g, sin_g, kc, vc, pos, ctx, max_ctx):
    k_raw = k_flat[:, :G_KV].reshape(B, G_KVH, G_HD)
    q = head_norm(q_flat[:, :G_Q].reshape(B, NH, G_HD), qn)
    k = head_norm(k_raw, kn)
    v = head_norm_noscale(k_raw)  # k_eq_v: V = v_norm(raw_K), no scale, no RoPE
    c = cos_g[pos][None, None, :]
    s = sin_g[pos][None, None, :]
    q = rope(q, c, s, 128)
    k = rope(k, c, s, 128)
    k_val = jnp.pad(k.reshape(B, G_KV)[0], (0, MAX_KV - G_KV)).astype(kc.dtype)
    v_val = jnp.pad(v.reshape(B, G_KV)[0], (0, MAX_KV - G_KV)).astype(vc.dtype)
    kc = kc.at[pos].set(k_val)
    vc = vc.at[pos].set(v_val)
    k_ctx = kc[:max_ctx, :G_KV].reshape(max_ctx, G_KVH, G_HD)
    v_ctx = vc[:max_ctx, :G_KV].reshape(max_ctx, G_KVH, G_HD)
    q_g = q.reshape(B, G_KVH, G_GQA, G_HD)
    sc = jnp.einsum('bghd,tgd->bght', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))
    # Gemma 4 uses scaling=1.0 (QK-norm handles magnitude)
    # sc = sc / jnp.sqrt(jnp.float32(G_HD))
    valid = jnp.arange(max_ctx) < ctx
    sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('bght,tgd->bghd', p, v_ctx).reshape(B, G_Q)
    return out, kc, vc

# ── flat scan body ──

def one_layer(carry, xs):
    x, pos, ctx, cos_s, sin_s, cos_g, sin_g = carry
    max_ctx = xs['kc'].shape[0]

    residual = x
    h = rms_norm(x, xs['ln1'])

    q_flat = int8_matmul(h, xs['qw'], xs['qw_s'])
    k_flat = int8_matmul(h, xs['kw'], xs['kw_s'])
    v_flat = int8_matmul(h, xs['vw'], xs['vw_s'])

    ig = xs['ig']

    def do_sliding(args):
        q, k, v, qn, kn, kc, vc = args
        out, kc2, vc2 = _sliding_attn(q, k, v, qn, kn, cos_s, sin_s, kc, vc, pos, ctx, max_ctx)
        # Pad attn output to MAX_Q
        return jnp.pad(out, ((0,0),(0, MAX_Q - S_Q))), kc2, vc2

    def do_global(args):
        q, k, v, qn, kn, kc, vc = args
        out, kc2, vc2 = _global_attn(q, k, v, qn, kn, cos_g, sin_g, kc, vc, pos, ctx, max_ctx)
        return out, kc2, vc2  # already MAX_Q size

    attn_out, kc, vc = jax.lax.cond(
        ig, do_global, do_sliding,
        (q_flat, k_flat, v_flat, xs['qn'], xs['kn'], xs['kc'], xs['vc']))

    o_out = int8_matmul(attn_out, xs['ow'], xs['ow_s'])

    h = o_out
    h = rms_norm(h, xs['ln2'])
    x = residual + h * xs['ls']

    residual = x
    h = rms_norm(x, xs['ln3'])
    gate = int8_matmul(h, xs['gw'], xs['gw_s'])
    up = int8_matmul(h, xs['uw'], xs['uw_s'])
    h = jax.nn.gelu(gate, approximate=True) * up
    h = int8_matmul(h, xs['dw'], xs['dw_s'])
    h = rms_norm(h, xs['ln4'])
    x = residual + h * xs['ls']

    return (x, pos, ctx, cos_s, sin_s, cos_g, sin_g), {'kc': kc, 'vc': vc}

# ── forward ──

def forward(token_id, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g):
    x = embed[token_id].reshape(B, H) * jnp.sqrt(jnp.float32(H))
    init = (x, pos, ctx, cos_s, sin_s, cos_g, sin_g)

    layer_xs = {**weights, 'kc': caches['kc'], 'vc': caches['vc']}
    final, scan_out = jax.lax.scan(one_layer, init, layer_xs)
    x = final[0]
    x = rms_norm(x, final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    new_caches = {'kc': scan_out['kc'], 'vc': scan_out['vc']}
    return jnp.argmax(logits, axis=-1).astype(jnp.int32), log_probs, new_caches

# ── safetensors reader ──

def read_safetensors(path):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        data = np.memmap(path, dtype=np.uint8, mode='r', offset=data_start)
    tensors = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        shape = tuple(info['shape'])
        dtype_str = info['dtype']
        start, end = info['data_offsets']
        raw = np.array(data[start:end])
        if dtype_str in ('BF16', 'bf16', 'bfloat16'):
            tensors[name] = raw.view(np.uint16).reshape(shape)
        elif dtype_str in ('F16', 'f16', 'float16'):
            tensors[name] = raw.view(np.float16).reshape(shape)
        elif dtype_str in ('F32', 'f32', 'float32'):
            tensors[name] = raw.view(np.float32).reshape(shape)
    return tensors

def to_np_bf16(arr):
    if arr.dtype == np.uint16:
        return arr.view(ml_dtypes.bfloat16)
    if arr.dtype == np.float16:
        return arr.astype(np.float32).astype(ml_dtypes.bfloat16)
    if arr.dtype == np.float32:
        return arr.astype(ml_dtypes.bfloat16)
    if arr.dtype == ml_dtypes.bfloat16:
        return arr
    raise ValueError(f"unsupported dtype {arr.dtype}")

# ── weight loading ──

def pad_to(arr, target_shape):
    pads = [(0, t - s) for s, t in zip(arr.shape, target_shape)]
    return np.pad(arr, pads)

def quantize_int8_perchannel(arr_bf16):
    """Quantize bf16 weight to int8 with per-output-channel scale."""
    w = arr_bf16.astype(np.float32)
    amax = np.abs(w).max(axis=-1, keepdims=True).clip(min=1e-10)
    scale = (amax / 127.0).astype(np.float32)
    w_int8 = np.round(w / scale).clip(-127, 127).astype(np.int8)
    return w_int8, scale.squeeze(-1).astype(ml_dtypes.bfloat16)

def int8_matmul(x, w_int8, scale):
    """x @ (w_int8 * scale).T with fused int8 read."""
    return (x @ w_int8.astype(jnp.bfloat16).T) * scale

def load_model(model_dir, mesh, max_ctx):
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
        weight_map = index['weight_map']
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ['model.safetensors']
        weight_map = None

    prefix = 'model'
    if weight_map:
        for k in weight_map:
            if k.startswith('model.language_model.'):
                prefix = 'model.language_model'
                break

    print(f"loading from {model_dir}, prefix={prefix}", file=sys.stderr)
    all_t = {}
    for sn in shard_names:
        print(f"  reading {sn}...", file=sys.stderr)
        all_t.update(read_safetensors(os.path.join(model_dir, sn)))
    print(f"  {len(all_t)} tensors", file=sys.stderr)

    def get(name):
        return all_t[name]
    def has(name):
        return name in all_t

    def put(arr, spec):
        return jax.device_put(to_np_bf16(arr), NamedSharding(mesh, spec))

    embed = put(get(f'{prefix}.embed_tokens.weight'), P(None, None))
    final_norm = put(get(f'{prefix}.norm.weight'), P(None))

    # Stack all 60 layers with padding to max shapes, quantize large weights to int8
    print("  stacking 60 layers (padded, int8 quantized)...", file=sys.stderr)
    matmul_keys = ['qw','kw','vw','ow','gw','uw','dw']
    bf16_keys = ['qn','kn','ln1','ln2','ln3','ln4','ls']
    stacked_i8 = {k: [] for k in matmul_keys}
    stacked_sc = {k+'_s': [] for k in matmul_keys}
    stacked_bf = {k: [] for k in bf16_keys}
    stacked_ig = []

    for i in range(NL):
        lp = f'{prefix}.layers.{i}'
        is_global = LAYER_IS_GLOBAL[i]

        qw = to_np_bf16(get(f'{lp}.self_attn.q_proj.weight'))
        kw = to_np_bf16(get(f'{lp}.self_attn.k_proj.weight'))
        ow = to_np_bf16(get(f'{lp}.self_attn.o_proj.weight'))

        if has(f'{lp}.self_attn.v_proj.weight'):
            vw = to_np_bf16(get(f'{lp}.self_attn.v_proj.weight'))
        else:
            vw = np.zeros((MAX_KV, H), dtype=ml_dtypes.bfloat16)

        gw = to_np_bf16(get(f'{lp}.mlp.gate_proj.weight'))
        uw = to_np_bf16(get(f'{lp}.mlp.up_proj.weight'))
        dw = to_np_bf16(get(f'{lp}.mlp.down_proj.weight'))

        raw = {'qw': pad_to(qw, (MAX_Q, H)), 'kw': pad_to(kw, (MAX_KV, H)),
               'vw': pad_to(vw, (MAX_KV, H)), 'ow': pad_to(ow, (H, MAX_O)),
               'gw': gw, 'uw': uw, 'dw': dw}
        for k in matmul_keys:
            w_i8, sc = quantize_int8_perchannel(raw[k])
            stacked_i8[k].append(w_i8)
            stacked_sc[k+'_s'].append(sc)

        stacked_bf['qn'].append(pad_to(to_np_bf16(get(f'{lp}.self_attn.q_norm.weight')), (MAX_NORM_HD,)))
        stacked_bf['kn'].append(pad_to(to_np_bf16(get(f'{lp}.self_attn.k_norm.weight')), (MAX_NORM_HD,)))
        stacked_bf['ln1'].append(to_np_bf16(get(f'{lp}.input_layernorm.weight')))
        stacked_bf['ln2'].append(to_np_bf16(get(f'{lp}.post_attention_layernorm.weight')))
        stacked_bf['ln3'].append(to_np_bf16(get(f'{lp}.pre_feedforward_layernorm.weight')))
        stacked_bf['ln4'].append(to_np_bf16(get(f'{lp}.post_feedforward_layernorm.weight')))
        stacked_bf['ls'].append(to_np_bf16(get(f'{lp}.layer_scalar')) if has(f'{lp}.layer_scalar') else np.array([1.0], dtype=ml_dtypes.bfloat16))
        stacked_ig.append(np.array(is_global, dtype=np.int32))

        if i % 15 == 0:
            print(f"    layer {i}", file=sys.stderr)

    # Sharding for int8 weights (same partition axes as bf16)
    i8_sharding = {
        'qw': P(None, 'tp', None), 'kw': P(None, 'tp', None), 'vw': P(None, 'tp', None),
        'ow': P(None, None, 'tp'), 'gw': P(None, 'tp', None), 'uw': P(None, 'tp', None),
        'dw': P(None, None, 'tp'),
    }
    sc_sharding = {
        'qw_s': P(None, 'tp'), 'kw_s': P(None, 'tp'), 'vw_s': P(None, 'tp'),
        'ow_s': P(None, None), 'gw_s': P(None, 'tp'), 'uw_s': P(None, 'tp'),
        'dw_s': P(None, None),
    }

    def put_i8(arr, spec):
        return jax.device_put(jnp.array(arr, dtype=jnp.int8), NamedSharding(mesh, spec))

    weights = {}
    for k in matmul_keys:
        arr = np.stack(stacked_i8[k])
        weights[k] = put_i8(arr, i8_sharding[k])
        print(f"    {k}: {arr.shape} int8", file=sys.stderr)
        sc_arr = np.stack(stacked_sc[k+'_s'])
        weights[k+'_s'] = put(sc_arr, sc_sharding[k+'_s'])
    for k in bf16_keys:
        arr = np.stack(stacked_bf[k])
        weights[k] = put(arr, P(None, None))
        print(f"    {k}: {arr.shape} bf16", file=sys.stderr)
    weights['ig'] = jax.device_put(jnp.array(np.array(stacked_ig)), NamedSharding(mesh, P(None)))

    kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
    caches = {
        'kc': jax.device_put(jnp.zeros((NL, max_ctx, MAX_KV), dtype=jnp.bfloat16), kv_sh),
        'vc': jax.device_put(jnp.zeros((NL, max_ctx, MAX_KV), dtype=jnp.bfloat16), kv_sh),
    }

    del all_t, stacked_i8, stacked_sc, stacked_bf
    print("  done loading", file=sys.stderr)
    return embed, final_norm, weights, caches

# ── main ──

def load_tokenizer(model_dir):
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tok_path)
    return None

def run_perplexity(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g):
    tokenizer = load_tokenizer(args.model_dir)
    if tokenizer is None:
        print("ERROR: no tokenizer.json found", file=sys.stderr)
        return

    if args.ppl_file:
        with open(args.ppl_file) as f:
            text = f.read()
    else:
        text = "The quick brown fox jumps over the lazy dog. In the beginning was the Word, and the Word was with God, and the Word was God. He was in the beginning with God. All things were made through him, and without him was not any thing made that was made. In him was life, and the life was the light of men. The light shines in the darkness, and the darkness has not overcome it."

    token_ids = tokenizer.encode(text).ids
    max_tokens = min(len(token_ids), args.max_ctx - 1)
    token_ids = [2] + token_ids[:max_tokens]  # BOS + text
    print(f"perplexity eval: {len(token_ids)} tokens", file=sys.stderr)

    fwd_jit = jax.jit(forward, donate_argnums=(6,))
    total_nll = 0.0
    n_tokens = 0
    t_start = time.time()

    for step in range(len(token_ids) - 1):
        tok_arr = jnp.array([token_ids[step]], dtype=jnp.int32)
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)

        next_tok, log_probs, caches = fwd_jit(
            tok_arr, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)

        target = token_ids[step + 1]
        nll = -float(log_probs[0, target])
        total_nll += nll
        n_tokens += 1

        if step % 50 == 0:
            ppl_so_far = float(np.exp(total_nll / max(n_tokens, 1)))
            elapsed = time.time() - t_start
            tps = n_tokens / max(elapsed, 0.01)
            print(f"  step {step}/{len(token_ids)-1} nll={nll:.3f} ppl={ppl_so_far:.2f} ({tps:.1f} tok/s)", file=sys.stderr)

    avg_nll = total_nll / n_tokens
    ppl = float(np.exp(avg_nll))
    elapsed = time.time() - t_start
    print(file=sys.stderr)
    print(f"=== Perplexity ===", file=sys.stderr)
    print(f"tokens:     {n_tokens}", file=sys.stderr)
    print(f"avg NLL:    {avg_nll:.4f}", file=sys.stderr)
    print(f"perplexity: {ppl:.2f}", file=sys.stderr)
    print(f"time:       {elapsed:.1f}s ({n_tokens/elapsed:.1f} tok/s)", file=sys.stderr)

def run_generate(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g):
    prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]
    print(f"prompt: {len(prompt_ids)} tokens {prompt_ids[:10]}", file=sys.stderr)

    fwd_jit = jax.jit(forward, donate_argnums=(6,))
    generated = []
    last_sampled = None
    total_steps = len(prompt_ids) + args.max_tokens
    t_start = time.time()
    ttft = None

    for step in range(total_steps):
        if step < len(prompt_ids):
            token_id = prompt_ids[step]
        else:
            token_id = last_sampled
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        tok_arr = jnp.array([token_id], dtype=jnp.int32)

        t0 = time.time()
        next_tok, _log_probs, caches = fwd_jit(
            tok_arr, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
        next_tok.block_until_ready()
        dt = time.time() - t0

        sampled = int(next_tok[0])
        last_sampled = sampled

        if step < len(prompt_ids):
            print('.', end='', file=sys.stderr, flush=True)
            if step == len(prompt_ids) - 1:
                ttft = time.time() - t_start
                print(f"\nTTFT: {ttft*1000:.1f}ms ({len(prompt_ids)} prompt tokens)", file=sys.stderr)
        else:
            generated.append(sampled)
            if sampled in (1, 2, 107):
                print(f"\n[EOS tok={sampled} at step {step}]", file=sys.stderr)
                break
            print(f"[{sampled}]", end='', file=sys.stderr, flush=True)
            if step < len(prompt_ids) + 3:
                print(f" ({dt*1000:.1f}ms)", end='', file=sys.stderr, flush=True)

    total = time.time() - t_start
    print(file=sys.stderr)
    print("=== Results ===", file=sys.stderr)
    print(f"prompt tokens:    {len(prompt_ids)}", file=sys.stderr)
    print(f"generated tokens: {len(generated)}", file=sys.stderr)
    if ttft:
        print(f"TTFT:             {ttft*1000:.1f}ms", file=sys.stderr)
    if len(generated) > 1 and ttft:
        decode_time = total - ttft
        tps = len(generated) / decode_time
        print(f"decode tok/s:     {tps:.1f}", file=sys.stderr)
        print(f"ms/token:         {decode_time/len(generated)*1000:.1f}", file=sys.stderr)
    print(f"total time:       {total:.1f}s", file=sys.stderr)
    print(f"generated:        {generated[:20]}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--max-ctx', type=int, default=2048)
    parser.add_argument('--prompt', default='2')
    parser.add_argument('--perplexity', action='store_true')
    parser.add_argument('--ppl-file', default=None)
    args = parser.parse_args()

    mesh = make_mesh()
    print(f"mesh: {mesh}", file=sys.stderr)

    max_ctx = args.max_ctx
    embed, final_norm, weights, caches = load_model(args.model_dir, mesh, max_ctx)

    cos_s, sin_s = precompute_rope(10000.0, S_HD, max_ctx)
    cos_g, sin_g = precompute_rope(1000000.0, 128, max_ctx)
    cos_s = jax.device_put(jnp.array(cos_s), NamedSharding(mesh, P(None, None)))
    sin_s = jax.device_put(jnp.array(sin_s), NamedSharding(mesh, P(None, None)))
    cos_g = jax.device_put(jnp.array(cos_g), NamedSharding(mesh, P(None, None)))
    sin_g = jax.device_put(jnp.array(sin_g), NamedSharding(mesh, P(None, None)))

    if args.perplexity:
        run_perplexity(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)
    else:
        run_generate(args, mesh, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g)

if __name__ == '__main__':
    main()
