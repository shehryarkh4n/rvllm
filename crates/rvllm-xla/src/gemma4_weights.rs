//! Gemma 4 31B weight loader for TPU (PJRT).
//!
//! Loads safetensors shards, quantizes to int8 with per-channel absmax scales,
//! pads heterogeneous sliding/global attention weights to uniform max shapes,
//! stacks all 60 layers into [60, ...] tensors for the scan body, and uploads
//! to PJRT buffers with tensor-parallel sharding.
//!
//! Architecture (from config.json / Gemma4Arch):
//!   hidden=5376, heads=32, layers=60, intermediate=21504, vocab=262144
//!   Sliding (every non-6th): head_dim=256, kv_heads=16, theta=10000
//!   Global  (every 6th):     head_dim=512, kv_heads=4,  theta=1M, partial_rotary=0.25
//!
//! Weight shapes per layer type:
//!   Sliding: q=[8192,5376]  k=[4096,5376]  v=[4096,5376]  o=[5376,8192]
//!   Global:  q=[16384,5376] k=[2048,5376]  v=absent(=k)   o=[5376,16384]
//!
//! Padded max shapes for stacking:
//!   q=[16384,5376]  k=[4096,5376]  v=[4096,5376]  o=[5376,16384]
//!
//! Norms: input, post_attn, pre_ff, post_ff = [5376] each
//! q_norm: [256] sliding / [512] global -> padded to [512]
//! k_norm: [256] both -> [256]
//! layer_scalar: [1]

#![cfg(feature = "tpu")]

use std::collections::BTreeMap;
use std::path::Path;

use crate::client::{PjrtBufferHandle, PjrtClientHandle};
use crate::ffi::PjrtElementType;

// Gemma 4 31B constants
const NUM_LAYERS: usize = 60;
const HIDDEN: usize = 5376;
const NUM_HEADS: usize = 32;
const HEAD_DIM_SLIDING: usize = 256;
const HEAD_DIM_GLOBAL: usize = 512;
const NUM_KV_HEADS_SLIDING: usize = 16;
const NUM_KV_HEADS_GLOBAL: usize = 4;
const INTERMEDIATE: usize = 21504;
const VOCAB: usize = 262144;

// Derived padded dims (max across layer types)
const MAX_Q_DIM: usize = NUM_HEADS * HEAD_DIM_GLOBAL; // 32*512 = 16384
const MAX_KV_DIM: usize = NUM_KV_HEADS_SLIDING * HEAD_DIM_SLIDING; // 16*256 = 4096
const MAX_O_OUT: usize = MAX_Q_DIM; // 16384
const MAX_Q_NORM: usize = HEAD_DIM_GLOBAL; // 512
const K_NORM_DIM: usize = HEAD_DIM_SLIDING; // 256

/// All weight buffers for Gemma 4, organized for the fused scan execute call.
pub struct Gemma4Weights {
    // Non-layer weights
    pub embedding: PjrtBufferHandle,     // [vocab, hidden] bf16
    pub final_norm: PjrtBufferHandle,    // [hidden] bf16
    pub lm_head: PjrtBufferHandle,       // [vocab, hidden] bf16

    // RoPE tables
    pub rope_cos_sliding: PjrtBufferHandle, // [max_pos, head_dim_sliding/2] f32
    pub rope_sin_sliding: PjrtBufferHandle,
    pub rope_cos_global: PjrtBufferHandle,  // [max_pos, rotary_dim_global/2] f32
    pub rope_sin_global: PjrtBufferHandle,

    // Stacked [60, ...] int8 weight tensors + bf16 scale tensors
    pub all_q_i8: PjrtBufferHandle,      // [60, MAX_Q_DIM, hidden] S8
    pub all_q_scales: PjrtBufferHandle,  // [60, MAX_Q_DIM] BF16
    pub all_k_i8: PjrtBufferHandle,      // [60, MAX_KV_DIM, hidden] S8
    pub all_k_scales: PjrtBufferHandle,  // [60, MAX_KV_DIM] BF16
    pub all_v_i8: PjrtBufferHandle,      // [60, MAX_KV_DIM, hidden] S8
    pub all_v_scales: PjrtBufferHandle,  // [60, MAX_KV_DIM] BF16
    pub all_o_i8: PjrtBufferHandle,      // [60, hidden, MAX_O_OUT] S8
    pub all_o_scales: PjrtBufferHandle,  // [60, hidden] BF16
    pub all_gate_i8: PjrtBufferHandle,   // [60, INTERMEDIATE, hidden] S8
    pub all_gate_scales: PjrtBufferHandle, // [60, INTERMEDIATE] BF16
    pub all_up_i8: PjrtBufferHandle,     // [60, INTERMEDIATE, hidden] S8
    pub all_up_scales: PjrtBufferHandle, // [60, INTERMEDIATE] BF16
    pub all_down_i8: PjrtBufferHandle,   // [60, hidden, INTERMEDIATE] S8
    pub all_down_scales: PjrtBufferHandle, // [60, hidden] BF16

    // Stacked [60, ...] norm/scalar tensors (bf16, replicated)
    pub all_input_ln: PjrtBufferHandle,       // [60, hidden] BF16
    pub all_post_attn_ln: PjrtBufferHandle,   // [60, hidden] BF16
    pub all_pre_ff_ln: PjrtBufferHandle,      // [60, hidden] BF16
    pub all_post_ff_ln: PjrtBufferHandle,     // [60, hidden] BF16
    pub all_q_norm: PjrtBufferHandle,         // [60, MAX_Q_NORM] BF16
    pub all_k_norm: PjrtBufferHandle,         // [60, K_NORM_DIM] BF16
    pub all_layer_scalar: PjrtBufferHandle,   // [60, 1] BF16

    // Layer type mask: 0.0 for sliding, 1.0 for global (bf16)
    pub layer_type_mask: PjrtBufferHandle,    // [60] BF16
}

/// Safetensors tensor metadata
struct TensorMeta {
    #[allow(dead_code)]
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
}

type TensorMap = BTreeMap<String, TensorMeta>;

pub fn load_gemma4_weights(
    client: &PjrtClientHandle,
    model_dir: &Path,
    num_devices: usize,
) -> Gemma4Weights {
    let tensor_data = index_safetensors(model_dir);
    let prefix = detect_prefix(&tensor_data);
    eprintln!("[gemma4] {} tensors indexed, prefix=\"{prefix}\"", tensor_data.len());

    // --- Non-layer weights ---

    let embed_name = format!("{prefix}.embed_tokens.weight");
    // Gemma scales embeddings by sqrt(hidden_size)
    let embedding = {
        let mut bf16 = get_bf16_bytes(&tensor_data, &embed_name);
        let scale = (HIDDEN as f32).sqrt();
        scale_bf16_inplace(&mut bf16, scale);
        upload_bf16(client, &bf16, &[VOCAB as i64, HIDDEN as i64], 0)
    };

    let norm_name = format!("{prefix}.norm.weight");
    let final_norm = upload_bf16(
        client,
        &get_bf16_bytes(&tensor_data, &norm_name),
        &[HIDDEN as i64],
        0,
    );

    let lm_head = if tensor_data.contains_key("lm_head.weight") {
        upload_bf16(
            client,
            &get_bf16_bytes(&tensor_data, "lm_head.weight"),
            &[VOCAB as i64, HIDDEN as i64],
            0,
        )
    } else {
        // Tied embeddings: reuse embed_tokens (pre-scaled)
        let mut bf16 = get_bf16_bytes(&tensor_data, &embed_name);
        let scale = (HIDDEN as f32).sqrt();
        scale_bf16_inplace(&mut bf16, scale);
        upload_bf16(client, &bf16, &[VOCAB as i64, HIDDEN as i64], 0)
    };

    // --- RoPE ---
    let max_pos: usize = 8192; // decode context window
    let rope_cos_sliding = precompute_rope(client, 10000.0, HEAD_DIM_SLIDING, HEAD_DIM_SLIDING, max_pos, true);
    let rope_sin_sliding = precompute_rope(client, 10000.0, HEAD_DIM_SLIDING, HEAD_DIM_SLIDING, max_pos, false);
    // Global: partial rotary = 0.25 * 512 = 128 dims rotated
    let global_rotary_dim = 128;
    let rope_cos_global = precompute_rope(client, 1_000_000.0, HEAD_DIM_GLOBAL, global_rotary_dim, max_pos, true);
    let rope_sin_global = precompute_rope(client, 1_000_000.0, HEAD_DIM_GLOBAL, global_rotary_dim, max_pos, false);

    // --- Per-layer weight stacking ---

    // Weight byte accumulators
    let mut all_q_i8_bytes = Vec::with_capacity(NUM_LAYERS * MAX_Q_DIM * HIDDEN);
    let mut all_q_scales_bytes = Vec::with_capacity(NUM_LAYERS * MAX_Q_DIM * 2);
    let mut all_k_i8_bytes = Vec::with_capacity(NUM_LAYERS * MAX_KV_DIM * HIDDEN);
    let mut all_k_scales_bytes = Vec::with_capacity(NUM_LAYERS * MAX_KV_DIM * 2);
    let mut all_v_i8_bytes = Vec::with_capacity(NUM_LAYERS * MAX_KV_DIM * HIDDEN);
    let mut all_v_scales_bytes = Vec::with_capacity(NUM_LAYERS * MAX_KV_DIM * 2);
    let mut all_o_i8_bytes = Vec::with_capacity(NUM_LAYERS * HIDDEN * MAX_O_OUT);
    let mut all_o_scales_bytes = Vec::with_capacity(NUM_LAYERS * HIDDEN * 2);
    let mut all_gate_i8_bytes = Vec::with_capacity(NUM_LAYERS * INTERMEDIATE * HIDDEN);
    let mut all_gate_scales_bytes = Vec::with_capacity(NUM_LAYERS * INTERMEDIATE * 2);
    let mut all_up_i8_bytes = Vec::with_capacity(NUM_LAYERS * INTERMEDIATE * HIDDEN);
    let mut all_up_scales_bytes = Vec::with_capacity(NUM_LAYERS * INTERMEDIATE * 2);
    let mut all_down_i8_bytes = Vec::with_capacity(NUM_LAYERS * HIDDEN * INTERMEDIATE);
    let mut all_down_scales_bytes = Vec::with_capacity(NUM_LAYERS * HIDDEN * 2);

    let mut all_input_ln_bytes = Vec::with_capacity(NUM_LAYERS * HIDDEN * 2);
    let mut all_post_attn_ln_bytes = Vec::with_capacity(NUM_LAYERS * HIDDEN * 2);
    let mut all_pre_ff_ln_bytes = Vec::with_capacity(NUM_LAYERS * HIDDEN * 2);
    let mut all_post_ff_ln_bytes = Vec::with_capacity(NUM_LAYERS * HIDDEN * 2);
    let mut all_q_norm_bytes = Vec::with_capacity(NUM_LAYERS * MAX_Q_NORM * 2);
    let mut all_k_norm_bytes = Vec::with_capacity(NUM_LAYERS * K_NORM_DIM * 2);
    let mut all_layer_scalar_bytes = Vec::with_capacity(NUM_LAYERS * 2);
    let mut layer_mask_f32 = Vec::with_capacity(NUM_LAYERS);

    for l in 0..NUM_LAYERS {
        let is_global = (l + 1) % 6 == 0;
        let ln = |s: &str| format!("{prefix}.layers.{l}.{s}");

        // Determine this layer's actual dims
        let (hd, nkvh) = if is_global {
            (HEAD_DIM_GLOBAL, NUM_KV_HEADS_GLOBAL)
        } else {
            (HEAD_DIM_SLIDING, NUM_KV_HEADS_SLIDING)
        };
        let q_dim = NUM_HEADS * hd;
        let kv_dim = nkvh * hd;

        // --- Q projection: [q_dim, hidden] -> pad to [MAX_Q_DIM, hidden] ---
        let q_bf16 = get_bf16_bytes(&tensor_data, &ln("self_attn.q_proj.weight"));
        let (qi8, qsc) = quantize_i8_perchannel(&q_bf16, q_dim, HIDDEN);
        let (qi8_padded, qsc_padded) = pad_weight_rows(&qi8, &qsc, q_dim, HIDDEN, MAX_Q_DIM);
        all_q_i8_bytes.extend_from_slice(&qi8_padded);
        all_q_scales_bytes.extend_from_slice(&qsc_padded);

        // --- K projection: [kv_dim, hidden] -> pad to [MAX_KV_DIM, hidden] ---
        let k_bf16 = get_bf16_bytes(&tensor_data, &ln("self_attn.k_proj.weight"));
        let (ki8, ksc) = quantize_i8_perchannel(&k_bf16, kv_dim, HIDDEN);
        let (ki8_padded, ksc_padded) = pad_weight_rows(&ki8, &ksc, kv_dim, HIDDEN, MAX_KV_DIM);
        all_k_i8_bytes.extend_from_slice(&ki8_padded);
        all_k_scales_bytes.extend_from_slice(&ksc_padded);

        // --- V projection: [kv_dim, hidden] or absent (global reuses K) ---
        let v_name = ln("self_attn.v_proj.weight");
        let v_bf16 = if tensor_data.contains_key(&v_name) {
            get_bf16_bytes(&tensor_data, &v_name)
        } else {
            k_bf16.clone()
        };
        let (vi8, vsc) = quantize_i8_perchannel(&v_bf16, kv_dim, HIDDEN);
        let (vi8_padded, vsc_padded) = pad_weight_rows(&vi8, &vsc, kv_dim, HIDDEN, MAX_KV_DIM);
        all_v_i8_bytes.extend_from_slice(&vi8_padded);
        all_v_scales_bytes.extend_from_slice(&vsc_padded);

        // --- O projection: [hidden, q_dim] -> pad to [hidden, MAX_O_OUT] ---
        let o_bf16 = get_bf16_bytes(&tensor_data, &ln("self_attn.o_proj.weight"));
        let (oi8, osc) = quantize_i8_perchannel(&o_bf16, HIDDEN, q_dim);
        let (oi8_padded, osc_padded) = pad_weight_cols(&oi8, &osc, HIDDEN, q_dim, MAX_O_OUT);
        all_o_i8_bytes.extend_from_slice(&oi8_padded);
        all_o_scales_bytes.extend_from_slice(&osc_padded);

        // --- Gate projection: [intermediate, hidden] ---
        let gate_bf16 = get_bf16_bytes(&tensor_data, &ln("mlp.gate_proj.weight"));
        let (gi8, gsc) = quantize_i8_perchannel(&gate_bf16, INTERMEDIATE, HIDDEN);
        all_gate_i8_bytes.extend_from_slice(&gi8);
        all_gate_scales_bytes.extend_from_slice(&gsc);

        // --- Up projection: [intermediate, hidden] ---
        let up_bf16 = get_bf16_bytes(&tensor_data, &ln("mlp.up_proj.weight"));
        let (ui8, usc) = quantize_i8_perchannel(&up_bf16, INTERMEDIATE, HIDDEN);
        all_up_i8_bytes.extend_from_slice(&ui8);
        all_up_scales_bytes.extend_from_slice(&usc);

        // --- Down projection: [hidden, intermediate] ---
        let down_bf16 = get_bf16_bytes(&tensor_data, &ln("mlp.down_proj.weight"));
        let (di8, dsc) = quantize_i8_perchannel(&down_bf16, HIDDEN, INTERMEDIATE);
        all_down_i8_bytes.extend_from_slice(&di8);
        all_down_scales_bytes.extend_from_slice(&dsc);

        // --- Norms: [hidden] bf16, Gemma uses (1+gamma) so pre-add 1.0 ---
        let iln = gemma_norm_bf16(&get_bf16_bytes(&tensor_data, &ln("input_layernorm.weight")));
        all_input_ln_bytes.extend_from_slice(&iln);

        let paln = gemma_norm_bf16(&get_bf16_bytes(&tensor_data, &ln("post_attention_layernorm.weight")));
        all_post_attn_ln_bytes.extend_from_slice(&paln);

        let pffln = gemma_norm_bf16(&get_bf16_bytes(&tensor_data, &ln("pre_feedforward_layernorm.weight")));
        all_pre_ff_ln_bytes.extend_from_slice(&pffln);

        let postffln = gemma_norm_bf16(&get_bf16_bytes(&tensor_data, &ln("post_feedforward_layernorm.weight")));
        all_post_ff_ln_bytes.extend_from_slice(&postffln);

        // --- q_norm: [head_dim] -> pad to [MAX_Q_NORM=512] ---
        let qn = get_bf16_bytes(&tensor_data, &ln("self_attn.q_norm.weight"));
        let qn_padded = pad_1d_bf16(&qn, hd, MAX_Q_NORM);
        all_q_norm_bytes.extend_from_slice(&qn_padded);

        // --- k_norm: [head_dim_sliding=256] always, no padding needed ---
        // Global layers have k_norm [256] too (shared head_dim for norm after projection)
        let kn = get_bf16_bytes(&tensor_data, &ln("self_attn.k_norm.weight"));
        let kn_padded = pad_1d_bf16(&kn, kn.len() / 2, K_NORM_DIM);
        all_k_norm_bytes.extend_from_slice(&kn_padded);

        // --- layer_scalar: [1] bf16 ---
        let ls = get_bf16_bytes(&tensor_data, &ln("layer_scalar"));
        all_layer_scalar_bytes.extend_from_slice(&ls);

        // Layer type mask
        layer_mask_f32.push(if is_global { 1.0f32 } else { 0.0f32 });

        if l < 2 || l == NUM_LAYERS - 1 {
            eprintln!("[gemma4] layer {l} ({}) stacked", if is_global { "global" } else { "sliding" });
        } else if l == 2 {
            eprintln!("[gemma4] ...");
        }
    }

    // --- Upload stacked tensors ---
    let nl = NUM_LAYERS as i64;
    let h = HIDDEN as i64;
    let mqd = MAX_Q_DIM as i64;
    let mkd = MAX_KV_DIM as i64;
    let mo = MAX_O_OUT as i64;
    let inter = INTERMEDIATE as i64;

    let dev = shard_device(0, num_devices);

    // Int8 weights + bf16 scales
    let all_q_i8 = client.buffer_from_host(&all_q_i8_bytes, &[nl, mqd, h], PjrtElementType::S8, dev).unwrap();
    let all_q_scales = client.buffer_from_host(&all_q_scales_bytes, &[nl, mqd], PjrtElementType::BF16, dev).unwrap();
    let all_k_i8 = client.buffer_from_host(&all_k_i8_bytes, &[nl, mkd, h], PjrtElementType::S8, dev).unwrap();
    let all_k_scales = client.buffer_from_host(&all_k_scales_bytes, &[nl, mkd], PjrtElementType::BF16, dev).unwrap();
    let all_v_i8 = client.buffer_from_host(&all_v_i8_bytes, &[nl, mkd, h], PjrtElementType::S8, dev).unwrap();
    let all_v_scales = client.buffer_from_host(&all_v_scales_bytes, &[nl, mkd], PjrtElementType::BF16, dev).unwrap();
    let all_o_i8 = client.buffer_from_host(&all_o_i8_bytes, &[nl, h, mo], PjrtElementType::S8, dev).unwrap();
    let all_o_scales = client.buffer_from_host(&all_o_scales_bytes, &[nl, h], PjrtElementType::BF16, dev).unwrap();
    let all_gate_i8 = client.buffer_from_host(&all_gate_i8_bytes, &[nl, inter, h], PjrtElementType::S8, dev).unwrap();
    let all_gate_scales = client.buffer_from_host(&all_gate_scales_bytes, &[nl, inter], PjrtElementType::BF16, dev).unwrap();
    let all_up_i8 = client.buffer_from_host(&all_up_i8_bytes, &[nl, inter, h], PjrtElementType::S8, dev).unwrap();
    let all_up_scales = client.buffer_from_host(&all_up_scales_bytes, &[nl, inter], PjrtElementType::BF16, dev).unwrap();
    let all_down_i8 = client.buffer_from_host(&all_down_i8_bytes, &[nl, h, inter], PjrtElementType::S8, dev).unwrap();
    let all_down_scales = client.buffer_from_host(&all_down_scales_bytes, &[nl, h], PjrtElementType::BF16, dev).unwrap();

    // Norm/scalar tensors (replicated)
    let all_input_ln = client.buffer_from_host(&all_input_ln_bytes, &[nl, h], PjrtElementType::BF16, dev).unwrap();
    let all_post_attn_ln = client.buffer_from_host(&all_post_attn_ln_bytes, &[nl, h], PjrtElementType::BF16, dev).unwrap();
    let all_pre_ff_ln = client.buffer_from_host(&all_pre_ff_ln_bytes, &[nl, h], PjrtElementType::BF16, dev).unwrap();
    let all_post_ff_ln = client.buffer_from_host(&all_post_ff_ln_bytes, &[nl, h], PjrtElementType::BF16, dev).unwrap();
    let all_q_norm = client.buffer_from_host(&all_q_norm_bytes, &[nl, MAX_Q_NORM as i64], PjrtElementType::BF16, dev).unwrap();
    let all_k_norm = client.buffer_from_host(&all_k_norm_bytes, &[nl, K_NORM_DIM as i64], PjrtElementType::BF16, dev).unwrap();
    let all_layer_scalar = client.buffer_from_host(&all_layer_scalar_bytes, &[nl, 1], PjrtElementType::BF16, dev).unwrap();

    // Layer type mask
    let mask_bf16: Vec<u8> = layer_mask_f32.iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();
    let layer_type_mask = client.buffer_from_host(&mask_bf16, &[nl], PjrtElementType::BF16, dev).unwrap();

    eprintln!("[gemma4] all weights uploaded ({NUM_LAYERS} layers, int8 quantized)");

    Gemma4Weights {
        embedding,
        final_norm,
        lm_head,
        rope_cos_sliding,
        rope_sin_sliding,
        rope_cos_global,
        rope_sin_global,
        all_q_i8,
        all_q_scales,
        all_k_i8,
        all_k_scales,
        all_v_i8,
        all_v_scales,
        all_o_i8,
        all_o_scales,
        all_gate_i8,
        all_gate_scales,
        all_up_i8,
        all_up_scales,
        all_down_i8,
        all_down_scales,
        all_input_ln,
        all_post_attn_ln,
        all_pre_ff_ln,
        all_post_ff_ln,
        all_q_norm,
        all_k_norm,
        all_layer_scalar,
        layer_type_mask,
    }
}

// ---------------------------------------------------------------------------
// Safetensors parsing
// ---------------------------------------------------------------------------

fn index_safetensors(model_dir: &Path) -> TensorMap {
    let idx_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    let shard_paths: Vec<std::path::PathBuf> = if idx_path.exists() {
        let idx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&idx_path).unwrap()).unwrap();
        let wm = idx["weight_map"].as_object().unwrap();
        let mut shards: Vec<String> = wm
            .values()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        shards.iter().map(|s| model_dir.join(s)).collect()
    } else if single_path.exists() {
        vec![single_path]
    } else {
        panic!("no safetensors found in {model_dir:?}");
    };

    let mut tensor_data: TensorMap = BTreeMap::new();
    for sp in &shard_paths {
        let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(sp).unwrap()).unwrap() };
        let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + header_len]).unwrap();
        let data_start = 8 + header_len;
        for (name, info) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let dtype = info["dtype"].as_str().unwrap().to_string();
            let offsets = info["data_offsets"].as_array().unwrap();
            let start = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;
            let bytes = mmap[data_start + start..data_start + end].to_vec();
            tensor_data.insert(
                name.clone(),
                TensorMeta {
                    shape,
                    dtype,
                    data: bytes,
                },
            );
        }
    }
    tensor_data
}

fn detect_prefix(tensors: &TensorMap) -> String {
    for key in tensors.keys() {
        if key.starts_with("model.language_model.") {
            return "model.language_model.model".to_string();
        }
        if key.starts_with("language_model.model.") {
            return "language_model.model".to_string();
        }
        if key.starts_with("language_model.") {
            return "language_model".to_string();
        }
    }
    "model".to_string()
}

// ---------------------------------------------------------------------------
// Dtype conversion helpers
// ---------------------------------------------------------------------------

fn get_bf16_bytes(tensors: &TensorMap, name: &str) -> Vec<u8> {
    let t = tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor: {name}"));
    match t.dtype.as_str() {
        "BF16" | "bf16" => t.data.clone(),
        "F16" | "f16" => f16_to_bf16(&t.data),
        "F32" | "f32" => f32_to_bf16(&t.data),
        other => panic!("unsupported dtype {other} for {name}"),
    }
}

fn f16_to_bf16(bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        let v = half::f16::from_bits(bits).to_f32();
        out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    out
}

fn f32_to_bf16(bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    out
}

fn bf16_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        out.push(half::bf16::from_bits(bits).to_f32());
    }
    out
}

fn scale_bf16_inplace(bytes: &mut [u8], scale: f32) {
    let n = bytes.len() / 2;
    for i in 0..n {
        let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        let v = half::bf16::from_bits(bits).to_f32() * scale;
        let out = half::bf16::from_f32(v).to_le_bytes();
        bytes[2 * i] = out[0];
        bytes[2 * i + 1] = out[1];
    }
}

/// Gemma RMSNorm uses (1 + gamma) * x. Pre-add 1.0 to the gamma weights.
fn gemma_norm_bf16(bf16_bytes: &[u8]) -> Vec<u8> {
    let n = bf16_bytes.len() / 2;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let bits = u16::from_le_bytes([bf16_bytes[2 * i], bf16_bytes[2 * i + 1]]);
        let v = half::bf16::from_bits(bits).to_f32() + 1.0;
        out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Int8 per-channel quantization
// ---------------------------------------------------------------------------

/// Per-channel (axis 0) absmax quantization.
/// Input: bf16 bytes for a [rows, cols] weight matrix.
/// Output: (i8_bytes [rows*cols], scale_bf16_bytes [rows*2])
///
/// For each row: scale = max(|row|) / 127
///               i8[i] = clamp(round(bf16[i] / scale), -127, 127)
fn quantize_i8_perchannel(bf16_bytes: &[u8], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(
        bf16_bytes.len(),
        rows * cols * 2,
        "bf16 byte count mismatch: expected {} got {}",
        rows * cols * 2,
        bf16_bytes.len()
    );

    let f32_vals = bf16_to_f32_slice(bf16_bytes);
    let mut i8_bytes = Vec::with_capacity(rows * cols);
    let mut scale_bytes = Vec::with_capacity(rows * 2);

    for r in 0..rows {
        let row = &f32_vals[r * cols..(r + 1) * cols];
        let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
        let inv_scale = 1.0 / scale;

        for &v in row {
            let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            i8_bytes.push(q as u8);
        }
        scale_bytes.extend_from_slice(&half::bf16::from_f32(scale).to_le_bytes());
    }

    (i8_bytes, scale_bytes)
}

// ---------------------------------------------------------------------------
// Padding helpers
// ---------------------------------------------------------------------------

/// Pad a [rows, cols] i8 weight to [target_rows, cols] with zero rows.
/// Also pads the scale vector from [rows] to [target_rows] with zeros.
fn pad_weight_rows(
    i8_bytes: &[u8],
    scale_bytes: &[u8],
    rows: usize,
    cols: usize,
    target_rows: usize,
) -> (Vec<u8>, Vec<u8>) {
    if rows == target_rows {
        return (i8_bytes.to_vec(), scale_bytes.to_vec());
    }
    assert!(rows <= target_rows, "cannot shrink rows {rows} -> {target_rows}");

    let mut padded_i8 = Vec::with_capacity(target_rows * cols);
    padded_i8.extend_from_slice(i8_bytes);
    padded_i8.resize(target_rows * cols, 0u8);

    let mut padded_sc = Vec::with_capacity(target_rows * 2);
    padded_sc.extend_from_slice(scale_bytes);
    padded_sc.resize(target_rows * 2, 0u8);

    (padded_i8, padded_sc)
}

/// Pad a [rows, cols] i8 weight to [rows, target_cols] with zero columns.
/// Scales are per-row so they remain unchanged.
fn pad_weight_cols(
    i8_bytes: &[u8],
    scale_bytes: &[u8],
    rows: usize,
    cols: usize,
    target_cols: usize,
) -> (Vec<u8>, Vec<u8>) {
    if cols == target_cols {
        return (i8_bytes.to_vec(), scale_bytes.to_vec());
    }
    assert!(cols <= target_cols, "cannot shrink cols {cols} -> {target_cols}");

    let mut padded = Vec::with_capacity(rows * target_cols);
    for r in 0..rows {
        padded.extend_from_slice(&i8_bytes[r * cols..(r + 1) * cols]);
        padded.resize(padded.len() + (target_cols - cols), 0u8);
    }
    (padded, scale_bytes.to_vec())
}

/// Pad a 1D bf16 vector from [dim] to [target_dim] with bf16 zeros.
fn pad_1d_bf16(bf16_bytes: &[u8], dim: usize, target_dim: usize) -> Vec<u8> {
    if dim == target_dim {
        return bf16_bytes.to_vec();
    }
    assert!(dim <= target_dim, "cannot shrink 1d {dim} -> {target_dim}");
    let mut out = Vec::with_capacity(target_dim * 2);
    out.extend_from_slice(&bf16_bytes[..dim * 2]);
    out.resize(target_dim * 2, 0u8);
    out
}

// ---------------------------------------------------------------------------
// RoPE precompute
// ---------------------------------------------------------------------------

fn precompute_rope(
    client: &PjrtClientHandle,
    theta: f32,
    _head_dim: usize,
    rotary_dim: usize,
    max_pos: usize,
    is_cos: bool,
) -> PjrtBufferHandle {
    let half_rot = rotary_dim / 2;
    let mut data = vec![0f32; max_pos * half_rot];
    for pos in 0..max_pos {
        for i in 0..half_rot {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32);
            let angle = pos as f32 * freq;
            data[pos * half_rot + i] = if is_cos { angle.cos() } else { angle.sin() };
        }
    }
    client
        .buffer_from_host(
            bytemuck::cast_slice(&data),
            &[max_pos as i64, half_rot as i64],
            PjrtElementType::F32,
            0,
        )
        .unwrap()
}

// ---------------------------------------------------------------------------
// Upload helpers
// ---------------------------------------------------------------------------

fn upload_bf16(
    client: &PjrtClientHandle,
    bf16_bytes: &[u8],
    shape: &[i64],
    device: usize,
) -> PjrtBufferHandle {
    client
        .buffer_from_host(bf16_bytes, shape, PjrtElementType::BF16, device)
        .unwrap()
}

/// Map a shard index to a device for tensor-parallel placement.
/// With a single device, everything goes to device 0.
fn shard_device(shard_idx: usize, num_devices: usize) -> usize {
    if num_devices <= 1 {
        0
    } else {
        shard_idx % num_devices
    }
}
