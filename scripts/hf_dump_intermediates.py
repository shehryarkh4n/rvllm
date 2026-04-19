#!/usr/bin/env python3
"""Dump all intermediate tensors from HF Gemma4 for parity checking."""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/gemma4-31b-fp8"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hf_dump"
os.makedirs(out_dir, exist_ok=True)

print(f"Loading {model_path}...", file=sys.stderr)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()

text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
layers = text_model.layers

# Fixed input: BOS + "The quick brown"
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
ids = [2] + tok.encode("The quick brown")[:3]
print(f"Token IDs: {ids}", file=sys.stderr)
device = next(model.parameters()).device
input_ids = torch.tensor([ids], dtype=torch.long, device=device)

# Dump embedding
embed_raw = text_model.embed_tokens.weight[ids[0]].detach().float().cpu()
tc = model.config.text_config if hasattr(model.config, "text_config") else model.config
H = tc.hidden_size
embed_scaled = embed_raw * (H ** 0.5)
torch.save(embed_scaled, f"{out_dir}/embed_token0.pt")
print(f"embed_token0: first4={embed_scaled[:4].tolist()}", file=sys.stderr)

# Hook every layer to capture input/output residual
captures = {}

def make_hook(layer_idx):
    def hook(module, args, output):
        # input is args[0] = hidden_states (the residual going IN)
        inp = args[0] if isinstance(args, tuple) else args
        captures[f"L{layer_idx}_input"] = inp.detach().float().cpu()[0, 0, :]  # first token
        captures[f"L{layer_idx}_output"] = output.detach().float().cpu()[0, 0, :]
    return hook

for i in range(min(len(layers), 6)):
    layers[i].register_forward_hook(make_hook(i))

# Hook final norm
def norm_hook(module, args, output):
    captures["final_norm_out"] = output.detach().float().cpu()[0, 0, :]
text_model.norm.register_forward_hook(norm_hook)

print("Running forward...", file=sys.stderr)
with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=False)

# Logits for first token
logits = out.logits[0, 0].float().cpu()
captures["logits"] = logits
print(f"logits first5: {logits[:5].tolist()}", file=sys.stderr)
print(f"logits amax: {logits.abs().max().item():.4f}", file=sys.stderr)

# Save all
for name, tensor in captures.items():
    torch.save(tensor, f"{out_dir}/{name}.pt")
    v = tensor.flatten()[:4].tolist()
    print(f"{name}: first4={[f'{x:.6f}' for x in v]}", file=sys.stderr)

print(f"\nSaved {len(captures)} tensors to {out_dir}", file=sys.stderr)
