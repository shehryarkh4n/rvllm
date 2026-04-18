import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/models/gemma4-31b-fp8", dtype=torch.bfloat16, device_map="auto"
)
ids = torch.tensor([[818]], device="cuda")
cap = {}

layer0 = model.model.language_model.layers[0]
attn = layer0.self_attn

attn.q_proj.register_forward_hook(lambda m,i,o: cap.update({"q_proj": o.float().detach().cpu()}))
attn.k_proj.register_forward_hook(lambda m,i,o: cap.update({"k_proj": o.float().detach().cpu()}))
attn.v_proj.register_forward_hook(lambda m,i,o: cap.update({"v_proj": o.float().detach().cpu()}))
attn.o_proj.register_forward_pre_hook(lambda m,i: cap.update({"attn_out_pre_o_proj": i[0].float().detach().cpu()}))
attn.o_proj.register_forward_hook(lambda m,i,o: cap.update({"o_proj": o.float().detach().cpu()}))

if hasattr(attn, "q_norm"):
    attn.q_norm.register_forward_hook(lambda m,i,o: cap.update({"q_norm": o.float().detach().cpu()}))
if hasattr(attn, "k_norm"):
    attn.k_norm.register_forward_hook(lambda m,i,o: cap.update({"k_norm": o.float().detach().cpu()}))

layer0.input_layernorm.register_forward_hook(lambda m,i,o: cap.update({"input_norm": o.float().detach().cpu()}))
layer0.post_attention_layernorm.register_forward_hook(lambda m,i,o: cap.update({"post_attn_norm": o.float().detach().cpu()}))

with torch.no_grad():
    out = model(ids)

print("=== HF Layer 0 ATTENTION internals ===")
for name in [
    "input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "attn_out_pre_o_proj",
    "o_proj",
    "post_attn_norm",
]:
    if name in cap:
        t = cap[name].squeeze()
        flat = t.flatten()
        print(f"{name}: shape={list(t.shape)} amax={t.abs().max():.4f} first8={flat[:8].tolist()}")
    else:
        print(f"{name}: NOT CAPTURED")
