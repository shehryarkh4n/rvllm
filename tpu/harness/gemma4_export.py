#!/usr/bin/env python3
"""Export Gemma 4 31B forward pass as StableHLO artifact for Rust consumption.

Traces the SPMD forward function with concrete shapes, serializes the
StableHLO module, and writes a JSON manifest describing every input/output
tensor (name, shape, dtype, sharding) so the Rust runtime can bind buffers.

Usage:
    python3 gemma4_export.py --model-dir /path/to/gemma-4-31B-it \
        --max-ctx 2048 --output gemma4_step
"""
import argparse, json, os, sys, time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax import ShapeDtypeStruct as sds

from gemma4_tpu_infer import (
    forward, make_mesh, load_model, precompute_rope,
    H, NH, INTER, VOCAB, NL, WINDOW,
    MAX_Q, MAX_KV, MAX_O, MAX_NORM_HD,
    S_HD, B,
)


def build_abstract_args(max_ctx):
    """Build ShapeDtypeStruct specs for every forward() argument."""

    # Scalars / small tensors
    token_id = sds((1,), jnp.int32)
    pos = sds((), jnp.int32)
    ctx = sds((), jnp.int32)

    # Embedding table and final norm
    embed = sds((VOCAB, H), jnp.bfloat16)
    final_norm = sds((H,), jnp.bfloat16)

    # Stacked weights dict -- mirrors load_model output
    weights = {
        # int8 matmul weights [NL, out, in]
        'qw': sds((NL, MAX_Q, H), jnp.int8),
        'kw': sds((NL, MAX_KV, H), jnp.int8),
        'vw': sds((NL, MAX_KV, H), jnp.int8),
        'ow': sds((NL, H, MAX_O), jnp.int8),
        'gw': sds((NL, INTER, H), jnp.int8),
        'uw': sds((NL, INTER, H), jnp.int8),
        'dw': sds((NL, H, INTER), jnp.int8),
        # per-channel scales [NL, out]
        'qw_s': sds((NL, MAX_Q), jnp.bfloat16),
        'kw_s': sds((NL, MAX_KV), jnp.bfloat16),
        'vw_s': sds((NL, MAX_KV), jnp.bfloat16),
        'ow_s': sds((NL, H), jnp.bfloat16),
        'gw_s': sds((NL, INTER), jnp.bfloat16),
        'uw_s': sds((NL, INTER), jnp.bfloat16),
        'dw_s': sds((NL, H), jnp.bfloat16),
        # bf16 per-layer params
        'qn': sds((NL, MAX_NORM_HD), jnp.bfloat16),
        'kn': sds((NL, MAX_NORM_HD), jnp.bfloat16),
        'ln1': sds((NL, H), jnp.bfloat16),
        'ln2': sds((NL, H), jnp.bfloat16),
        'ln3': sds((NL, H), jnp.bfloat16),
        'ln4': sds((NL, H), jnp.bfloat16),
        'ls': sds((NL, 1), jnp.bfloat16),
        # layer type flag
        'ig': sds((NL,), jnp.int32),
    }

    # KV caches
    caches = {
        'kc': sds((NL, max_ctx, MAX_KV), jnp.bfloat16),
        'vc': sds((NL, max_ctx, MAX_KV), jnp.bfloat16),
    }

    # RoPE tables
    cos_s = sds((max_ctx, S_HD // 2), jnp.float32)
    sin_s = sds((max_ctx, S_HD // 2), jnp.float32)
    cos_g = sds((max_ctx, 64), jnp.float32)
    sin_g = sds((max_ctx, 64), jnp.float32)

    return token_id, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g


def build_shardings(mesh, max_ctx):
    """Build NamedSharding specs matching load_model placement."""

    token_id_s = NamedSharding(mesh, P(None))
    pos_s = NamedSharding(mesh, P())
    ctx_s = NamedSharding(mesh, P())
    embed_s = NamedSharding(mesh, P(None, None))
    final_norm_s = NamedSharding(mesh, P(None))

    weights_s = {
        'qw': NamedSharding(mesh, P(None, 'tp', None)),
        'kw': NamedSharding(mesh, P(None, 'tp', None)),
        'vw': NamedSharding(mesh, P(None, 'tp', None)),
        'ow': NamedSharding(mesh, P(None, None, 'tp')),
        'gw': NamedSharding(mesh, P(None, 'tp', None)),
        'uw': NamedSharding(mesh, P(None, 'tp', None)),
        'dw': NamedSharding(mesh, P(None, None, 'tp')),
        'qw_s': NamedSharding(mesh, P(None, 'tp')),
        'kw_s': NamedSharding(mesh, P(None, 'tp')),
        'vw_s': NamedSharding(mesh, P(None, 'tp')),
        'ow_s': NamedSharding(mesh, P(None, None)),
        'gw_s': NamedSharding(mesh, P(None, 'tp')),
        'uw_s': NamedSharding(mesh, P(None, 'tp')),
        'dw_s': NamedSharding(mesh, P(None, None)),
        'qn': NamedSharding(mesh, P(None, None)),
        'kn': NamedSharding(mesh, P(None, None)),
        'ln1': NamedSharding(mesh, P(None, None)),
        'ln2': NamedSharding(mesh, P(None, None)),
        'ln3': NamedSharding(mesh, P(None, None)),
        'ln4': NamedSharding(mesh, P(None, None)),
        'ls': NamedSharding(mesh, P(None, None)),
        'ig': NamedSharding(mesh, P(None)),
    }

    caches_s = {
        'kc': NamedSharding(mesh, P(None, None, 'tp')),
        'vc': NamedSharding(mesh, P(None, None, 'tp')),
    }

    cos_s_s = NamedSharding(mesh, P(None, None))
    sin_s_s = NamedSharding(mesh, P(None, None))
    cos_g_s = NamedSharding(mesh, P(None, None))
    sin_g_s = NamedSharding(mesh, P(None, None))

    return (token_id_s, pos_s, ctx_s, embed_s, final_norm_s,
            weights_s, caches_s, cos_s_s, sin_s_s, cos_g_s, sin_g_s)


def sds_to_dict(name, spec):
    """Convert ShapeDtypeStruct to a JSON-serializable dict."""
    return {
        'name': name,
        'shape': list(spec.shape),
        'dtype': str(spec.dtype),
    }


def sharding_to_str(sharding):
    """Extract PartitionSpec from NamedSharding as a string."""
    if hasattr(sharding, 'spec'):
        return str(sharding.spec)
    return str(sharding)


def build_manifest(max_ctx, abstract_args, shardings, output_file):
    """Build JSON manifest describing all inputs/outputs for Rust."""
    token_id, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g = abstract_args
    (token_id_s, pos_s, ctx_s, embed_s, final_norm_s,
     weights_s, caches_s, cos_s_s, sin_s_s, cos_g_s, sin_g_s) = shardings

    manifest = {
        'model': 'gemma-4-31b',
        'max_ctx': max_ctx,
        'num_layers': NL,
        'hidden_dim': H,
        'num_heads': NH,
        'intermediate': INTER,
        'vocab_size': VOCAB,
        'batch_size': B,
    }

    inputs = []

    # Flat args in forward() signature order
    flat_specs = [
        ('token_id', token_id, token_id_s),
        ('pos', pos, pos_s),
        ('ctx', ctx, ctx_s),
        ('embed', embed, embed_s),
        ('final_norm', final_norm, final_norm_s),
    ]
    for name, spec, sh in flat_specs:
        d = sds_to_dict(name, spec)
        d['sharding'] = sharding_to_str(sh)
        inputs.append(d)

    # Weights dict (sorted for determinism)
    for k in sorted(weights.keys()):
        d = sds_to_dict(f'weights.{k}', weights[k])
        d['sharding'] = sharding_to_str(weights_s[k])
        inputs.append(d)

    # Caches
    for k in sorted(caches.keys()):
        d = sds_to_dict(f'caches.{k}', caches[k])
        d['sharding'] = sharding_to_str(caches_s[k])
        inputs.append(d)

    # RoPE
    for name, spec, sh in [
        ('cos_s', cos_s, cos_s_s), ('sin_s', sin_s, sin_s_s),
        ('cos_g', cos_g, cos_g_s), ('sin_g', sin_g, sin_g_s),
    ]:
        d = sds_to_dict(name, spec)
        d['sharding'] = sharding_to_str(sh)
        inputs.append(d)

    manifest['inputs'] = inputs

    # Outputs: forward returns (next_token, log_probs, new_caches)
    manifest['outputs'] = [
        {'name': 'next_token', 'shape': [B], 'dtype': 'int32'},
        {'name': 'log_probs', 'shape': [B, VOCAB], 'dtype': 'float32'},
        {'name': 'new_caches.kc', 'shape': [NL, max_ctx, MAX_KV], 'dtype': 'bfloat16'},
        {'name': 'new_caches.vc', 'shape': [NL, max_ctx, MAX_KV], 'dtype': 'bfloat16'},
    ]

    manifest['artifact_file'] = f'{output_file}.hlo.pb'

    return manifest


def export_stablehlo(mesh, max_ctx, output_file):
    """Trace forward(), export as serialized StableHLO, write manifest."""
    abstract_args = build_abstract_args(max_ctx)
    shardings = build_shardings(mesh, max_ctx)

    token_id, pos, ctx, embed, final_norm, weights, caches, cos_s, sin_s, cos_g, sin_g = abstract_args
    (token_id_s, pos_s, ctx_s, embed_s, final_norm_s,
     weights_s, caches_s, cos_s_s, sin_s_s, cos_g_s, sin_g_s) = shardings

    # Build in_shardings tree matching forward() signature
    in_shardings = (
        token_id_s, pos_s, ctx_s, embed_s, final_norm_s,
        weights_s, caches_s, cos_s_s, sin_s_s, cos_g_s, sin_g_s,
    )

    # Output shardings: (next_token, log_probs, new_caches)
    out_shardings = (
        NamedSharding(mesh, P(None)),                          # next_token [B]
        NamedSharding(mesh, P(None, None)),                    # log_probs [B, VOCAB]
        {
            'kc': NamedSharding(mesh, P(None, None, 'tp')),    # new KV caches
            'vc': NamedSharding(mesh, P(None, None, 'tp')),
        },
    )

    # JIT with SPMD shardings
    fwd_jit = jax.jit(forward, in_shardings=in_shardings, out_shardings=out_shardings)

    print(f"lowering forward() with max_ctx={max_ctx}...", file=sys.stderr)
    t0 = time.time()
    lowered = fwd_jit.lower(*abstract_args)
    dt_lower = time.time() - t0
    print(f"  lowered in {dt_lower:.1f}s", file=sys.stderr)

    # -- Method 1: jax.export (portable serialized artifact) --
    print("exporting via jax.export...", file=sys.stderr)
    t0 = time.time()
    exported = jax.export.export(
        fwd_jit,
        platforms=['tpu'],
    )(*abstract_args)
    dt_export = time.time() - t0
    print(f"  exported in {dt_export:.1f}s", file=sys.stderr)

    # Serialize the exported artifact (StableHLO bytecode)
    pb_path = f'{output_file}.hlo.pb'
    print(f"serializing to {pb_path}...", file=sys.stderr)
    serialized = exported.serialize()
    with open(pb_path, 'wb') as f:
        f.write(serialized)
    pb_size = len(serialized)
    print(f"  wrote {pb_path} ({pb_size / 1024 / 1024:.1f} MB)", file=sys.stderr)

    # -- Method 2: MLIR text for debugging --
    mlir_path = f'{output_file}.mlir'
    print(f"writing MLIR text to {mlir_path}...", file=sys.stderr)
    ir_text = str(lowered.compiler_ir(dialect="stablehlo"))
    with open(mlir_path, 'w') as f:
        f.write(ir_text)
    print(f"  wrote {mlir_path} ({ir_text.count(chr(10))} lines)", file=sys.stderr)

    # -- JSON manifest --
    manifest = build_manifest(max_ctx, abstract_args, shardings, output_file)
    manifest['lower_time_s'] = round(dt_lower, 2)
    manifest['export_time_s'] = round(dt_export, 2)
    manifest['artifact_size_bytes'] = pb_size
    manifest['mlir_file'] = mlir_path
    manifest['mlir_lines'] = ir_text.count('\n')

    # Record flattened input order from the exported module
    manifest['num_flat_inputs'] = exported.nr_devices
    manifest['calling_convention_version'] = str(exported.calling_convention_version)
    manifest['platforms'] = list(exported.platforms)

    json_path = f'{output_file}.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {json_path}", file=sys.stderr)

    return pb_path, mlir_path, json_path


def export_from_loaded(args, mesh):
    """Load real weights, trace with concrete data, export."""
    from gemma4_tpu_infer import load_model, precompute_rope

    max_ctx = args.max_ctx
    embed, final_norm, weights, caches = load_model(args.model_dir, mesh, max_ctx)

    cos_s_np, sin_s_np = precompute_rope(10000.0, S_HD, max_ctx)
    cos_g_np, sin_g_np = precompute_rope(1000000.0, 128, max_ctx)
    cos_s = jax.device_put(jnp.array(cos_s_np), NamedSharding(mesh, P(None, None)))
    sin_s = jax.device_put(jnp.array(sin_s_np), NamedSharding(mesh, P(None, None)))
    cos_g = jax.device_put(jnp.array(cos_g_np), NamedSharding(mesh, P(None, None)))
    sin_g = jax.device_put(jnp.array(sin_g_np), NamedSharding(mesh, P(None, None)))

    # Trace with real shardings from loaded weights
    fwd_jit = jax.jit(forward, donate_argnums=(6,))

    token_id = jnp.array([2], dtype=jnp.int32)
    pos = jnp.int32(0)
    ctx = jnp.int32(1)

    print(f"tracing forward() with loaded weights, max_ctx={max_ctx}...", file=sys.stderr)
    t0 = time.time()
    lowered = fwd_jit.lower(
        token_id, pos, ctx, embed, final_norm, weights, caches,
        cos_s, sin_s, cos_g, sin_g,
    )
    dt = time.time() - t0
    print(f"  lowered in {dt:.1f}s", file=sys.stderr)

    # MLIR text
    mlir_path = f'{args.output}.mlir'
    ir_text = str(lowered.compiler_ir(dialect="stablehlo"))
    with open(mlir_path, 'w') as f:
        f.write(ir_text)
    print(f"  wrote {mlir_path} ({ir_text.count(chr(10))} lines)", file=sys.stderr)

    # Serialized HLO
    pb_path = f'{args.output}.hlo.pb'
    compiled = lowered.compile()
    serialized = compiled.as_serialized_computation()
    with open(pb_path, 'wb') as f:
        f.write(serialized)
    print(f"  wrote {pb_path} ({len(serialized) / 1024 / 1024:.1f} MB)", file=sys.stderr)

    # Manifest
    abstract_args = build_abstract_args(max_ctx)
    shardings = build_shardings(mesh, max_ctx)
    manifest = build_manifest(max_ctx, abstract_args, shardings, args.output)
    manifest['lower_time_s'] = round(dt, 2)
    manifest['artifact_size_bytes'] = len(serialized)
    manifest['mlir_file'] = mlir_path
    manifest['mlir_lines'] = ir_text.count('\n')
    manifest['mode'] = 'loaded'

    json_path = f'{args.output}.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {json_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Export Gemma 4 31B forward pass as StableHLO artifact')
    parser.add_argument('--model-dir', default=None,
                        help='Path to model weights (optional -- traces with abstract shapes if omitted)')
    parser.add_argument('--max-ctx', type=int, default=2048,
                        help='Maximum context length')
    parser.add_argument('--output', default='gemma4_step',
                        help='Output file prefix (produces .hlo.pb, .mlir, .json)')
    args = parser.parse_args()

    mesh = make_mesh()
    print(f"mesh: {mesh} ({len(jax.devices())} devices)", file=sys.stderr)

    if args.model_dir:
        # Load real weights and trace with concrete data
        export_from_loaded(args, mesh)
    else:
        # Abstract-only export (no weights needed, uses ShapeDtypeStruct)
        export_stablehlo(mesh, args.max_ctx, args.output)

    print("done.", file=sys.stderr)


if __name__ == '__main__':
    main()
