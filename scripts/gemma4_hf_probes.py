#!/usr/bin/env python3
"""Single-token HF Gemma probe for rvLLM parity work."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM

def fmt4(t):
    v = t.flatten()[:4].float().tolist()
    return f"[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}, {v[3]:.4f}]"

def fmt5(t):
    v = t.flatten()[:5].float().tolist()
    return f"[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}, {v[3]:.4f}, {v[4]:.4f}]"

captures = {}

def patched_forward(self, original_forward, layer_idx, hidden_states, **kwargs):
    """Replace layer forward to capture intermediates at each sub-step."""
    c = {}
    c["input"] = hidden_states.detach().clone()

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(hidden_states=hidden_states, **kwargs)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    c["after_attn_residual"] = hidden_states.detach().clone()

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    c["after_ff_residual"] = hidden_states.detach().clone()

    # layer_scalar: check parameter, then state_dict
    scalar = None
    if hasattr(self, "layer_scalar"):
        scalar = self.layer_scalar.data.float()
    if scalar is not None:
        c["after_layer_scalar"] = (hidden_states * scalar.to(hidden_states.device)).detach().clone()
        c["scalar_val"] = scalar.item()
        # Apply scalar to output too (HF may not do this natively)
        hidden_states = hidden_states * scalar.to(hidden_states.device)
    else:
        c["after_layer_scalar"] = hidden_states.detach().clone()

    captures[layer_idx] = c
    return hidden_states

def stat_line(label, tensor):
    flat = tensor.detach().float().reshape(-1)
    print(
        f"{label}: shape={tuple(tensor.shape)} "
        f"min={flat.min().item():.6f} max={flat.max().item():.6f} "
        f"mean={flat.mean().item():.6f} amax={flat.abs().max().item():.6f}"
    )

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/workspace/models/gemma4-31b-fp8"),
    )
    p.add_argument("--token-id", type=int, default=2)
    p.add_argument("--probe-layers", default="0,1")
    p.add_argument("--norm-stats-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    probe_layers = {
        int(x) for x in args.probe_layers.split(",") if x.strip()
    }

    print(f"Loading {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tc = model.config.text_config if hasattr(model.config, "text_config") else model.config
    softcap = getattr(tc, "final_logit_softcapping", None)
    hidden_size = tc.hidden_size
    print(f"hidden_size={hidden_size}, softcap={softcap}")

    text_model = model.model if hasattr(model, "model") else model
    layers = text_model.layers

    print("\n[norm stats]")
    stat_line("final_norm.weight(raw)", text_model.norm.weight)
    for idx in sorted(probe_layers):
        layer = layers[idx]
        stat_line(f"L{idx}.input_ln.weight(raw)", layer.input_layernorm.weight)
        stat_line(f"L{idx}.post_attn_ln.weight(raw)", layer.post_attention_layernorm.weight)
        stat_line(f"L{idx}.pre_ff_ln.weight(raw)", layer.pre_feedforward_layernorm.weight)
        stat_line(f"L{idx}.post_ff_ln.weight(raw)", layer.post_feedforward_layernorm.weight)
        stat_line(f"L{idx}.q_norm.weight(raw)", layer.self_attn.q_norm.weight)
        stat_line(f"L{idx}.k_norm.weight(raw)", layer.self_attn.k_norm.weight)

    if args.norm_stats_only:
        print("\nDone.")
        return

    # Check for layer_scalar in state dict
    sd = model.state_dict()
    for idx in probe_layers:
        key = f"model.layers.{idx}.layer_scalar"
        if key in sd:
            print(f"  L{idx} layer_scalar = {sd[key].float().item():.6f}")
            # Attach as parameter if not already there
            if not hasattr(layers[idx], "layer_scalar"):
                layers[idx].layer_scalar = torch.nn.Parameter(sd[key], requires_grad=False)

    # Monkey-patch forward for probe layers
    import types
    originals = {}
    for idx in probe_layers:
        layer = layers[idx]
        originals[idx] = layer.forward
        layer.forward = types.MethodType(
            lambda self, hidden_states, _of=layer.forward, _idx=idx, **kw:
                patched_forward(self, _of, _idx, hidden_states, **kw),
            layer,
        )

    # Build input
    device = next(model.parameters()).device
    input_ids = torch.tensor([[args.token_id]], dtype=torch.long, device=device)

    # Embedding probe
    embed = text_model.embed_tokens(input_ids)
    print(f"\n[embed] first4={fmt4(embed)}")

    # Forward pass
    print("Running forward pass ...")
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    # Print captures
    for idx in sorted(probe_layers):
        c = captures.get(idx)
        if c is None:
            print(f"[L{idx}] NOT CAPTURED"); continue
        sv = f" (scalar={c['scalar_val']:.6f})" if "scalar_val" in c else ""
        print(f"  [L{idx} input] first4={fmt4(c['input'])}")
        print(f"  [L{idx} after_attn+post_attn_norm+residual] first4={fmt4(c['after_attn_residual'])}")
        print(f"  [L{idx} after_ff+post_ff_norm+residual] first4={fmt4(c['after_ff_residual'])}")
        print(f"  [L{idx} after_layer_scalar{sv}] first4={fmt4(c['after_layer_scalar'])}")

    # Logits (softcap already applied by model.forward)
    logits = out.logits[0, -1]
    label = f"(after softcap={softcap})" if softcap else "(no softcap)"
    print(f"\n[logits] {label} first5={fmt5(logits)}")

    # Raw logits before softcap
    if softcap is not None:
        # Invert softcap: logits = softcap * tanh(raw / softcap)
        # raw = softcap * atanh(logits / softcap)
        raw = softcap * torch.atanh(logits[:5].float() / softcap)
        print(f"[logits] (before softcap) first5={fmt5(raw)}")

    # Restore originals
    for idx in probe_layers:
        layers[idx].forward = originals[idx]

    print("\nDone.")


if __name__ == "__main__":
    main()
