//! Megakernel instruction tape builder for whole-model decode.

/// Matches MkInstr in megakernel_decode.cu (64 bytes)
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct MkInstr {
    pub instr_type: i32,
    pub flags: i32,
    pub wait_counter: i32,
    pub wait_value: i32,
    pub signal_counter: i32,
    pub dim_out: i32,
    pub dim_in: i32,
    pub dim_aux: i32,
    pub weight_idx: i32,
    pub norm_idx: i32,
    pub bias_idx: i32,
    pub gmem_in: i32,
    pub gmem_out: i32,
    pub gmem_aux: i32,
    pub gmem_aux2: i32,
    pub eps: f32,
}

impl Default for MkInstr {
    fn default() -> Self {
        Self {
            instr_type: 0, flags: 0, wait_counter: -1, wait_value: 0,
            signal_counter: -1, dim_out: 0, dim_in: 0, dim_aux: 0,
            weight_idx: -1, norm_idx: -1, bias_idx: -1,
            gmem_in: 0, gmem_out: 0, gmem_aux: 0, gmem_aux2: -1, eps: 0.0,
        }
    }
}

const _: () = assert!(core::mem::size_of::<MkInstr>() == 64);

// Instruction types (match CUDA kernel)
const INSTR_ADD_RMSNORM: i32 = 1;
const INSTR_GEMV_STRIDED: i32 = 2;
const INSTR_GEMV_CHUNKED: i32 = 3;
const INSTR_ROPE_CACHE: i32 = 4;
const INSTR_GQA_ATTENTION: i32 = 5;
const INSTR_ARGMAX: i32 = 6;

// Sync counter indices -- rolling window of 2 layer sets
const COUNTERS_PER_LAYER: i32 = 7;
fn sync_idx(layer: usize, phase: usize) -> i32 {
    ((layer % 2) as i32) * COUNTERS_PER_LAYER + phase as i32
}
const SC_QKV: usize = 0;
const SC_ROPE: usize = 1;
const SC_ATTN: usize = 2;
const SC_OPROJ: usize = 3;
const SC_GATEUP: usize = 4;
const SC_ALL_DOWN: usize = 6;

// Wait value formula: counters accumulate across layer reuses.
// Layer L uses counter set (L%2). After L signals, counter = grid * (L/2 + 1).
fn wait_val(layer: usize, grid: i32) -> i32 {
    grid * ((layer / 2) as i32 + 1)
}

// Load barrier counters for ADD_RMSNORM in-place residual race avoidance.
const LOAD_BARRIER_BASE: i32 = COUNTERS_PER_LAYER * 2; // after rolling slots
fn load_barrier_idx(layer: usize, sub: usize) -> i32 {
    LOAD_BARRIER_BASE + (layer * 2 + sub) as i32
}

// Scratch buffer layout (byte offsets, f16 = 2 bytes each)
struct ScratchLayout {
    qkv: i32,
    attn: i32,
    oproj: i32,
    gateup: i32,
    down: i32,
    residual_a: i32,
    residual_b: i32,
    lm_head_out: i32,
    total_bytes: usize,
}

impl ScratchLayout {
    fn new(
        hidden: usize, qkv_dim: usize, q_dim: usize,
        gate_up_dim: usize, _intermediate: usize, vocab: usize,
    ) -> Self {
        let mut off = 0usize;
        let qkv = off; off += qkv_dim * 2;
        let attn = off; off += q_dim * 2;
        let oproj = off; off += hidden * 2;
        let gateup = off; off += gate_up_dim * 2;
        let down = off; off += hidden * 2;
        let residual_a = off; off += hidden * 2;
        let residual_b = off; off += hidden * 2;
        let lm_head_out = off; off += vocab * 2;
        ScratchLayout {
            qkv: qkv as i32, attn: attn as i32, oproj: oproj as i32,
            gateup: gateup as i32, down: down as i32,
            residual_a: residual_a as i32, residual_b: residual_b as i32,
            lm_head_out: lm_head_out as i32, total_bytes: off,
        }
    }
}

const WEIGHTS_PER_LAYER: usize = 7;
const WD_INPUT_NORM: usize = 0;
const WD_QKV: usize = 1;
const WD_QKV_BIAS: usize = 2;
const WD_O_PROJ: usize = 3;
const WD_POST_NORM: usize = 4;
const WD_GATE_UP: usize = 5;
const WD_DOWN_PROJ: usize = 6;

fn weight_idx(layer: usize, slot: usize) -> i32 {
    (layer * WEIGHTS_PER_LAYER + slot) as i32
}
fn global_weight_idx(slot: usize) -> i32 {
    (28 * WEIGHTS_PER_LAYER + slot) as i32
}

/// Build the instruction tape for `num_layers` transformer layers + LM head.
pub fn build_instruction_tape(
    num_layers: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    vocab_size: usize,
    eps: f32,
    has_qkv_bias: bool,
) -> Vec<MkInstr> {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_dim = q_dim + 2 * kv_dim;
    let gate_up_dim = intermediate * 2;
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
    let s = ScratchLayout::new(hidden, qkv_dim, q_dim, gate_up_dim, intermediate, vocab_size);
    let grid = 256i32;

    let mut tape: Vec<MkInstr> = Vec::with_capacity(num_layers * 11 + 3);

    for layer in 0..num_layers {
        let is_first = layer == 0;

        // Double-buffer: layer N reads buf[N%2], writes buf[(N+1)%2]
        let res_in  = if layer % 2 == 0 { s.residual_a } else { s.residual_b };
        let res_out = if layer % 2 == 0 { s.residual_b } else { s.residual_a };

        // Inter-layer wait: wait for previous layer's down-proj
        let prev_wait = if is_first { -1 } else { sync_idx(layer - 1, SC_ALL_DOWN) };
        let prev_wait_val = if is_first { 0 } else { wait_val(layer - 1, grid) };

        // 0: ADD_RMSNORM (pre-attention)
        tape.push(MkInstr {
            instr_type: INSTR_ADD_RMSNORM,
            flags: if is_first { 1 | 8 } else { 8 },
            wait_counter: prev_wait,
            wait_value: prev_wait_val,
            dim_out: hidden as i32,
            dim_in: hidden as i32,
            norm_idx: weight_idx(layer, WD_INPUT_NORM),
            gmem_in: res_in,
            gmem_out: res_out,
            gmem_aux: s.down,
            gmem_aux2: load_barrier_idx(layer, 0),
            eps,
            ..Default::default()
        });

        // 1: QKV GEMV
        tape.push(MkInstr {
            instr_type: INSTR_GEMV_STRIDED,
            signal_counter: sync_idx(layer, SC_QKV),
            dim_out: qkv_dim as i32,
            dim_in: hidden as i32,
            weight_idx: weight_idx(layer, WD_QKV),
            gmem_out: s.qkv,
            ..Default::default()
        });

        // 2: ROPE + KV cache write (weight_idx = layer index for cache array)
        tape.push(MkInstr {
            instr_type: INSTR_ROPE_CACHE,
            flags: if has_qkv_bias { 2 } else { 0 },
            wait_counter: sync_idx(layer, SC_QKV),
            wait_value: wait_val(layer, grid),
            signal_counter: sync_idx(layer, SC_ROPE),
            dim_out: num_heads as i32,
            dim_in: num_kv_heads as i32,
            dim_aux: head_dim as i32,
            weight_idx: layer as i32, // layer index for cache arrays
            bias_idx: if has_qkv_bias { weight_idx(layer, WD_QKV_BIAS) } else { -1 },
            gmem_in: s.qkv,
            gmem_out: s.qkv,
            ..Default::default()
        });

        // 3: GQA attention (weight_idx = layer index for cache array)
        tape.push(MkInstr {
            instr_type: INSTR_GQA_ATTENTION,
            wait_counter: sync_idx(layer, SC_ROPE),
            wait_value: wait_val(layer, grid),
            signal_counter: sync_idx(layer, SC_ATTN),
            dim_out: num_heads as i32,
            dim_in: num_kv_heads as i32,
            dim_aux: head_dim as i32,
            weight_idx: layer as i32, // layer index for cache arrays
            gmem_in: s.qkv,
            gmem_out: s.attn,
            eps: attn_scale,
            ..Default::default()
        });

        // 4: O-proj GEMV (flag bit4: load gmem_in into smem)
        tape.push(MkInstr {
            instr_type: INSTR_GEMV_STRIDED,
            flags: 16,
            wait_counter: sync_idx(layer, SC_ATTN),
            wait_value: wait_val(layer, grid),
            signal_counter: sync_idx(layer, SC_OPROJ),
            dim_out: hidden as i32,
            dim_in: q_dim as i32,
            weight_idx: weight_idx(layer, WD_O_PROJ),
            gmem_in: s.attn,
            gmem_out: s.oproj,
            ..Default::default()
        });

        // 5: ADD_RMSNORM (post-attention: res_out + oproj -> normed)
        tape.push(MkInstr {
            instr_type: INSTR_ADD_RMSNORM,
            flags: 8,
            wait_counter: sync_idx(layer, SC_OPROJ),
            wait_value: wait_val(layer, grid),
            dim_out: hidden as i32,
            dim_in: hidden as i32,
            norm_idx: weight_idx(layer, WD_POST_NORM),
            gmem_in: res_out,  // reads what pre-attn wrote
            gmem_out: res_out, // overwrites in place (safe: O-proj sync ensures pre-attn done)
            gmem_aux: s.oproj,
            gmem_aux2: load_barrier_idx(layer, 1),
            eps,
            ..Default::default()
        });

        // 6: GateUp GEMV
        tape.push(MkInstr {
            instr_type: INSTR_GEMV_STRIDED,
            signal_counter: sync_idx(layer, SC_GATEUP),
            dim_out: gate_up_dim as i32,
            dim_in: hidden as i32,
            weight_idx: weight_idx(layer, WD_GATE_UP),
            gmem_out: s.gateup,
            ..Default::default()
        });

        // 7-10: Chunked SiLU*mul + down-proj (4 chunks)
        let chunk_size = intermediate / 4;
        for chunk in 0..4u32 {
            tape.push(MkInstr {
                instr_type: INSTR_GEMV_CHUNKED,
                flags: if chunk == 0 { 4 } else { 0 },
                wait_counter: sync_idx(layer, SC_GATEUP),
                wait_value: wait_val(layer, grid),
                signal_counter: if chunk == 3 { sync_idx(layer, SC_ALL_DOWN) } else { -1 },
                dim_out: hidden as i32,
                dim_in: chunk_size as i32,
                dim_aux: chunk as i32,
                weight_idx: weight_idx(layer, WD_DOWN_PROJ),
                gmem_in: s.gateup,
                gmem_out: s.down,
                gmem_aux: s.gateup + (intermediate * 2) as i32,
                ..Default::default()
            });
        }
    }

    // Final: norm + lm_head + argmax
    let last = num_layers - 1;
    // Last layer wrote to buf[(last+1)%2]
    let last_res = if (last + 1) % 2 == 0 { s.residual_a } else { s.residual_b };

    tape.push(MkInstr {
        instr_type: INSTR_ADD_RMSNORM,
        wait_counter: sync_idx(last, SC_ALL_DOWN),
        wait_value: wait_val(last, grid),
        dim_out: hidden as i32,
        dim_in: hidden as i32,
        norm_idx: global_weight_idx(0),
        gmem_in: last_res,
        gmem_aux: s.down,
        gmem_aux2: -1, // no in-place race (no write_residual flag)
        eps,
        ..Default::default()
    });

    tape.push(MkInstr {
        instr_type: INSTR_GEMV_STRIDED,
        dim_out: vocab_size as i32,
        dim_in: hidden as i32,
        weight_idx: global_weight_idx(1),
        gmem_out: s.lm_head_out,
        ..Default::default()
    });

    tape.push(MkInstr {
        instr_type: INSTR_ARGMAX,
        dim_out: vocab_size as i32,
        gmem_in: s.lm_head_out,
        ..Default::default()
    });

    tape
}

/// Required scratch buffer size in bytes.
pub fn scratch_size_bytes(
    hidden: usize, num_heads: usize, num_kv_heads: usize,
    head_dim: usize, intermediate: usize, vocab_size: usize,
) -> usize {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_dim = q_dim + 2 * kv_dim;
    let gate_up_dim = intermediate * 2;
    ScratchLayout::new(hidden, qkv_dim, q_dim, gate_up_dim, intermediate, vocab_size).total_bytes
}

/// Total number of weight pointer slots needed.
pub fn weight_ptr_count(num_layers: usize) -> usize {
    num_layers * WEIGHTS_PER_LAYER + 2
}

/// Total number of sync counters needed.
pub fn sync_counter_count(num_layers: usize) -> usize {
    (COUNTERS_PER_LAYER * 2) as usize + num_layers * 2
}
