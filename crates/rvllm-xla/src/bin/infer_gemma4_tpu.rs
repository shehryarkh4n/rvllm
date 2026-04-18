#[cfg(not(feature = "tpu"))]
fn main() {
    eprintln!("ERROR: infer-gemma4-tpu requires the 'tpu' feature.");
    eprintln!("Build with: cargo run --release -p rvllm-xla --features tpu --bin infer-gemma4-tpu");
    std::process::exit(1);
}

#[cfg(feature = "tpu")]
fn main() {
    tpu_main::run();
}

#[cfg(feature = "tpu")]
mod tpu_main {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::Instant;

    use clap::Parser;
    use rvllm_xla::client::{CompiledExecutable, PjrtBufferHandle, PjrtClientHandle};
    use rvllm_xla::ffi::PjrtElementType;

    #[derive(Parser)]
    #[command(name = "infer-gemma4-tpu", about = "Gemma 4 31B inference on TPU via PJRT (TP=4)")]
    struct Args {
        /// Path to gemma-4-31B-it safetensors directory
        #[arg(long)]
        model_dir: PathBuf,
        /// Path to exported StableHLO artifact directory
        #[arg(long)]
        artifact_dir: PathBuf,
        /// Number of tokens to generate
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        /// Max context length
        #[arg(long, default_value_t = 2048)]
        max_ctx: usize,
        /// Comma-separated token IDs (default "2" for BOS)
        #[arg(long, default_value = "2")]
        prompt: String,
    }

    // Gemma 4 31B architecture constants
    const HIDDEN: usize = 5376;
    const NUM_HEADS: usize = 32;
    const NUM_LAYERS: usize = 60;
    const INTERMEDIATE: usize = 21504;
    const VOCAB: usize = 262144;
    #[allow(dead_code)]
    const LOGIT_SOFTCAP: f32 = 30.0;

    // Sliding layers (majority): head_dim=256, 16 KV heads
    const SLIDING_HEAD_DIM: usize = 256;
    const SLIDING_KV_HEADS: usize = 16;
    #[allow(dead_code)]
    const SLIDING_WINDOW: usize = 1024;

    // Global layers (every 6th: 5, 11, 17, ...): head_dim=512, 4 KV heads
    const GLOBAL_HEAD_DIM: usize = 512;
    const GLOBAL_KV_HEADS: usize = 4;

    // RoPE parameters
    const ROPE_THETA_SLIDING: f32 = 10_000.0;
    const ROPE_THETA_GLOBAL: f32 = 1_000_000.0;
    const ROPE_GLOBAL_FRAC: f32 = 0.25; // only 25% of global dims rotated

    // TP sharding
    const TP: usize = 4;

    fn is_global_layer(l: usize) -> bool {
        // Global layers at indices 5, 11, 17, 23, 29, 35, 41, 47, 53, 59
        l % 6 == 5
    }

    pub fn run() {
        let args = Args::parse();

        // Parse prompt token IDs
        let prompt_ids: Vec<i32> = args
            .prompt
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if prompt_ids.is_empty() {
            eprintln!("FATAL: no valid token IDs in --prompt");
            std::process::exit(1);
        }

        // --- PJRT init ---
        let mut client = PjrtClientHandle::new().unwrap_or_else(|e| {
            eprintln!("FATAL: PJRT init failed: {e}");
            std::process::exit(1);
        });
        let num_devices = client.num_devices();
        eprintln!("PJRT client: {} device(s)", num_devices);
        if num_devices < TP {
            eprintln!("FATAL: need {} devices for TP={}, only {} available", TP, TP, num_devices);
            std::process::exit(1);
        }

        // Load TP=4 compile options
        let opts_path = args.artifact_dir.join("compile_options_tp4.pb");
        if opts_path.exists() {
            let opts = std::fs::read(&opts_path).unwrap();
            eprintln!("loaded compile options from {:?} ({} bytes)", opts_path, opts.len());
            client.set_compile_options(opts);
        } else {
            eprintln!("WARNING: {:?} not found, using default compile options", opts_path);
        }

        // --- Compile artifact ---
        eprintln!("compiling StableHLO artifact...");
        let t0 = Instant::now();
        let artifact = load_and_compile_artifact(&client, &args.artifact_dir);
        eprintln!("compiled in {:.1}s", t0.elapsed().as_secs_f32());

        // --- Load weights ---
        eprintln!("loading Gemma 4 31B weights from {:?}...", args.model_dir);
        let t0 = Instant::now();
        let weights = load_gemma4_weights(&client, &args.model_dir);
        eprintln!("loaded {} layers + embed/norms in {:.1}s", NUM_LAYERS, t0.elapsed().as_secs_f32());

        // --- Precompute RoPE tables ---
        let t0 = Instant::now();
        let rope = precompute_rope_tables(&client, args.max_ctx);
        eprintln!("RoPE tables precomputed in {:.1}s", t0.elapsed().as_secs_f32());

        // --- Allocate KV caches ---
        let t0 = Instant::now();
        let mut kv_caches = allocate_kv_caches(&client, args.max_ctx);
        eprintln!("KV caches allocated in {:.1}s", t0.elapsed().as_secs_f32());

        // --- Decode loop ---
        let mut generated: Vec<i32> = Vec::new();
        let mut context_len: i32 = 0;
        let mut ttft_ns: Option<u128> = None;
        let mut decode_start: Option<Instant> = None;
        let mut step_times_us: Vec<u128> = Vec::new();

        eprintln!("--- inference ---");
        eprintln!("prompt: {} tokens {:?}", prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(10)]);
        let prompt_start = Instant::now();
        let total_steps = prompt_ids.len() + args.max_tokens;

        for step in 0..total_steps {
            let token_id = if step < prompt_ids.len() {
                prompt_ids[step]
            } else {
                *generated.last().unwrap_or(&0)
            };

            let step_start = Instant::now();

            // Prepare scalar inputs
            let tok_buf = client
                .buffer_from_host(
                    bytemuck::cast_slice(&[token_id]),
                    &[1],
                    PjrtElementType::S32,
                    0,
                )
                .unwrap();
            let pos_buf = client
                .buffer_from_host(
                    bytemuck::cast_slice(&[context_len]),
                    &[1],
                    PjrtElementType::S32,
                    0,
                )
                .unwrap();
            let ctx_buf = client
                .buffer_from_host(
                    bytemuck::cast_slice(&[context_len + 1]),
                    &[1],
                    PjrtElementType::S32,
                    0,
                )
                .unwrap();

            // Build input list: token, pos, ctx, weights, caches, rope
            let inputs: Vec<&PjrtBufferHandle> = build_inputs(
                &tok_buf,
                &pos_buf,
                &ctx_buf,
                &weights,
                &kv_caches,
                &rope,
            );

            let t_exec = Instant::now();
            let step_out = client.execute(&artifact, &inputs).unwrap();
            let exec_us = t_exec.elapsed().as_micros();

            // Outputs: (next_token, updated_kv_caches...)
            let mut outs = step_out.into_iter();
            let token_out_buf = outs.next().expect("missing next_token output");

            // Update KV caches from outputs
            update_kv_caches(&mut kv_caches, &mut outs);

            // DtoH: extract next token
            let t_dtoh = Instant::now();
            let mut token_bytes = [0u8; 4];
            client.buffer_to_host(&token_out_buf, &mut token_bytes).unwrap();
            let dtoh_us = t_dtoh.elapsed().as_micros();

            let sampled: i32 = i32::from_le_bytes(token_bytes);
            let step_us = step_start.elapsed().as_micros();
            step_times_us.push(step_us);

            // Profile first few decode steps
            if step >= prompt_ids.len() && step < prompt_ids.len() + 3 {
                let ds = step - prompt_ids.len();
                eprintln!(
                    "[PROFILE] decode step {} total={}us exec={}us dtoh={}us",
                    ds, step_us, exec_us, dtoh_us
                );
            }

            context_len += 1;

            if step < prompt_ids.len() {
                eprint!(".");
                if step == prompt_ids.len() - 1 {
                    use std::io::Write;
                    std::io::stderr().flush().ok();
                    ttft_ns = Some(prompt_start.elapsed().as_nanos());
                    decode_start = Some(Instant::now());
                    eprintln!(
                        "\nTTFT: {:.2}ms ({} prompt tokens)",
                        ttft_ns.unwrap() as f64 / 1_000_000.0,
                        prompt_ids.len()
                    );
                    std::io::stderr().flush().ok();
                }
            } else {
                generated.push(sampled);
                // EOS tokens: 1 (</s>), 2 (<s>/BOS reused), 107 (<end_of_turn>)
                if sampled == 1 || sampled == 107 {
                    eprintln!("[EOS at step {}]", step);
                    break;
                }
                eprint!("[{}]", sampled);
            }
        }

        // --- Timing report ---
        let total_elapsed = prompt_start.elapsed();
        let decode_elapsed = decode_start.map(|s| s.elapsed());

        eprintln!();
        eprintln!("=== Results ===");
        eprintln!("prompt tokens:    {}", prompt_ids.len());
        eprintln!("generated tokens: {}", generated.len());
        if let Some(ttft) = ttft_ns {
            eprintln!("TTFT:             {:.2}ms", ttft as f64 / 1_000_000.0);
        }
        if let Some(dt) = decode_elapsed {
            let toks = generated.len();
            if toks > 0 {
                let tok_s = toks as f64 / dt.as_secs_f64();
                let ms_per_tok = dt.as_secs_f64() * 1000.0 / toks as f64;
                eprintln!("decode tok/s:     {:.1}", tok_s);
                eprintln!("ms/token:         {:.1}", ms_per_tok);
            }
        }
        // Per-step latency stats (decode only)
        if step_times_us.len() > prompt_ids.len() {
            let decode_steps: Vec<f64> = step_times_us[prompt_ids.len()..]
                .iter()
                .map(|&us| us as f64 / 1000.0)
                .collect();
            let min = decode_steps.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = decode_steps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = decode_steps.iter().sum::<f64>() / decode_steps.len() as f64;
            eprintln!("step latency:     min={:.1}ms avg={:.1}ms max={:.1}ms", min, mean, max);
        }
        eprintln!("total time:       {:.2}s", total_elapsed.as_secs_f64());
        eprintln!(
            "generated:        {:?}",
            &generated[..generated.len().min(20)]
        );
    }

    // -----------------------------------------------------------------------
    // Artifact loading
    // -----------------------------------------------------------------------

    fn load_and_compile_artifact(
        client: &PjrtClientHandle,
        artifact_dir: &PathBuf,
    ) -> CompiledExecutable {
        // Try bytecode first, then text MLIR
        let bytecode_path = artifact_dir.join("gemma4_decode_step.mlir.bc");
        let text_path = artifact_dir.join("gemma4_decode_step.mlir");

        if bytecode_path.exists() {
            let bc = std::fs::read(&bytecode_path).expect("read bytecode artifact");
            eprintln!("  artifact: bytecode ({} bytes)", bc.len());
            client.compile_bytecode(&bc).expect("compile bytecode artifact")
        } else if text_path.exists() {
            let mlir = std::fs::read_to_string(&text_path).expect("read text artifact");
            eprintln!("  artifact: text MLIR ({} bytes)", mlir.len());
            client.compile(&mlir).expect("compile text artifact")
        } else {
            eprintln!("FATAL: no artifact found at {:?}", artifact_dir);
            eprintln!("  expected: gemma4_decode_step.mlir.bc or gemma4_decode_step.mlir");
            std::process::exit(1);
        }
    }

    // -----------------------------------------------------------------------
    // Weight loading
    // -----------------------------------------------------------------------

    struct Gemma4Weights {
        // Embedding [VOCAB, HIDDEN] bf16 -- also used as tied lm_head
        embedding: PjrtBufferHandle,
        // Final RMSNorm [HIDDEN] bf16
        final_norm: PjrtBufferHandle,

        // Per-layer weights stored as vectors (not stacked, because sliding/global differ)
        // Each entry is for one layer.
        layer_input_norm: Vec<PjrtBufferHandle>,    // [HIDDEN] bf16
        layer_post_attn_norm: Vec<PjrtBufferHandle>, // [HIDDEN] bf16
        layer_pre_ff_norm: Vec<PjrtBufferHandle>,    // [HIDDEN] bf16
        layer_post_ff_norm: Vec<PjrtBufferHandle>,   // [HIDDEN] bf16
        layer_scalar: Vec<PjrtBufferHandle>,         // [1] bf16

        // Attention: Q, K, V (or K=V for global), O -- sharded across TP devices
        layer_q_proj: Vec<PjrtBufferHandle>,   // transposed to [HIDDEN, q_dim/TP]
        layer_k_proj: Vec<PjrtBufferHandle>,   // transposed to [HIDDEN, k_dim/TP]
        layer_v_proj: Vec<PjrtBufferHandle>,   // transposed to [HIDDEN, v_dim/TP] (= k_proj for global)
        layer_o_proj: Vec<PjrtBufferHandle>,   // transposed to [q_dim/TP, HIDDEN]

        // QK-Norm
        layer_q_norm: Vec<PjrtBufferHandle>,   // [head_dim] bf16
        layer_k_norm: Vec<PjrtBufferHandle>,   // [head_dim] bf16

        // MLP: gate||up fused, down -- sharded across TP
        layer_gate_up: Vec<PjrtBufferHandle>,  // transposed to [HIDDEN, 2*INTERMEDIATE/TP]
        layer_down: Vec<PjrtBufferHandle>,     // transposed to [INTERMEDIATE/TP, HIDDEN]
    }

    fn load_gemma4_weights(client: &PjrtClientHandle, model_dir: &PathBuf) -> Gemma4Weights {
        let tensor_data = load_safetensors_index(model_dir);
        eprintln!("  {} tensors indexed", tensor_data.len());

        // Embedding (tied with lm_head)
        let embedding = upload_bf16(
            client,
            &tensor_data,
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            &[VOCAB as i64, HIDDEN as i64],
        );

        // Final norm
        let final_norm = upload_bf16(
            client,
            &tensor_data,
            "model.language_model.norm.weight",
            "model.norm.weight",
            &[HIDDEN as i64],
        );

        let mut layer_input_norm = Vec::with_capacity(NUM_LAYERS);
        let mut layer_post_attn_norm = Vec::with_capacity(NUM_LAYERS);
        let mut layer_pre_ff_norm = Vec::with_capacity(NUM_LAYERS);
        let mut layer_post_ff_norm = Vec::with_capacity(NUM_LAYERS);
        let mut layer_scalar = Vec::with_capacity(NUM_LAYERS);
        let mut layer_q_proj = Vec::with_capacity(NUM_LAYERS);
        let mut layer_k_proj = Vec::with_capacity(NUM_LAYERS);
        let mut layer_v_proj = Vec::with_capacity(NUM_LAYERS);
        let mut layer_o_proj = Vec::with_capacity(NUM_LAYERS);
        let mut layer_q_norm = Vec::with_capacity(NUM_LAYERS);
        let mut layer_k_norm = Vec::with_capacity(NUM_LAYERS);
        let mut layer_gate_up = Vec::with_capacity(NUM_LAYERS);
        let mut layer_down = Vec::with_capacity(NUM_LAYERS);

        for l in 0..NUM_LAYERS {
            let prefix = find_layer_prefix(&tensor_data, l);
            let global = is_global_layer(l);
            let head_dim = if global { GLOBAL_HEAD_DIM } else { SLIDING_HEAD_DIM };
            let kv_heads = if global { GLOBAL_KV_HEADS } else { SLIDING_KV_HEADS };
            let q_dim = NUM_HEADS * head_dim;
            let k_dim = kv_heads * head_dim;

            // 4 norms
            layer_input_norm.push(upload_bf16_layer(
                client, &tensor_data, &prefix, "input_layernorm.weight", &[HIDDEN as i64],
            ));
            layer_post_attn_norm.push(upload_bf16_layer(
                client, &tensor_data, &prefix, "post_attention_layernorm.weight", &[HIDDEN as i64],
            ));
            layer_pre_ff_norm.push(upload_bf16_layer(
                client, &tensor_data, &prefix, "pre_feedforward_layernorm.weight", &[HIDDEN as i64],
            ));
            layer_post_ff_norm.push(upload_bf16_layer(
                client, &tensor_data, &prefix, "post_feedforward_layernorm.weight", &[HIDDEN as i64],
            ));

            // Layer scalar
            layer_scalar.push(upload_bf16_layer(
                client, &tensor_data, &prefix, "self_attn.layer_scalar", &[1],
            ));

            // QK-norm
            layer_q_norm.push(upload_bf16_layer(
                client, &tensor_data, &prefix, "self_attn.q_norm.weight", &[head_dim as i64],
            ));
            layer_k_norm.push(upload_bf16_layer(
                client, &tensor_data, &prefix, "self_attn.k_norm.weight", &[head_dim as i64],
            ));

            // Attention projections (transpose to [in, out] for matmul)
            // Q: [q_dim, HIDDEN] -> transpose -> [HIDDEN, q_dim]
            // Then shard: [HIDDEN, q_dim/TP] on each device
            let q_name = format!("{}.self_attn.q_proj.weight", prefix);
            let q_transposed = transpose_bf16(&tensor_data, &q_name);
            let q_shard_size = HIDDEN * (q_dim / TP) * 2;
            layer_q_proj.push(upload_sharded_bf16(
                client, &q_transposed, &[HIDDEN as i64, (q_dim / TP) as i64], q_shard_size,
            ));

            let k_name = format!("{}.self_attn.k_proj.weight", prefix);
            let k_transposed = transpose_bf16(&tensor_data, &k_name);
            let k_shard_size = HIDDEN * (k_dim / TP) * 2;
            layer_k_proj.push(upload_sharded_bf16(
                client, &k_transposed, &[HIDDEN as i64, (k_dim / TP) as i64], k_shard_size,
            ));

            // V: for global layers with attention_k_eq_v, V = K
            if global {
                // V doesn't exist; reuse K
                let v_transposed = k_transposed.clone();
                layer_v_proj.push(upload_sharded_bf16(
                    client, &v_transposed, &[HIDDEN as i64, (k_dim / TP) as i64], k_shard_size,
                ));
            } else {
                let v_name = format!("{}.self_attn.v_proj.weight", prefix);
                let v_transposed = transpose_bf16(&tensor_data, &v_name);
                layer_v_proj.push(upload_sharded_bf16(
                    client, &v_transposed, &[HIDDEN as i64, (k_dim / TP) as i64], k_shard_size,
                ));
            }

            // O: [HIDDEN, q_dim] -> transpose -> [q_dim, HIDDEN]
            // Shard along q_dim: [q_dim/TP, HIDDEN]
            let o_name = format!("{}.self_attn.o_proj.weight", prefix);
            let o_transposed = transpose_bf16(&tensor_data, &o_name);
            let o_shard_size = (q_dim / TP) * HIDDEN * 2;
            layer_o_proj.push(upload_sharded_bf16(
                client, &o_transposed, &[(q_dim / TP) as i64, HIDDEN as i64], o_shard_size,
            ));

            // MLP: gate||up fused, sharded
            let gate_name = format!("{}.mlp.gate_proj.weight", prefix);
            let up_name = format!("{}.mlp.up_proj.weight", prefix);
            let gu = concat_and_transpose_bf16(&tensor_data, &[&gate_name, &up_name], HIDDEN);
            let gu_shard_size = HIDDEN * (2 * INTERMEDIATE / TP) * 2;
            layer_gate_up.push(upload_sharded_bf16(
                client, &gu, &[HIDDEN as i64, (2 * INTERMEDIATE / TP) as i64], gu_shard_size,
            ));

            // Down: [HIDDEN, INTERMEDIATE] -> transpose -> [INTERMEDIATE, HIDDEN]
            // Shard: [INTERMEDIATE/TP, HIDDEN]
            let d_name = format!("{}.mlp.down_proj.weight", prefix);
            let d_transposed = transpose_bf16(&tensor_data, &d_name);
            let d_shard_size = (INTERMEDIATE / TP) * HIDDEN * 2;
            layer_down.push(upload_sharded_bf16(
                client, &d_transposed, &[(INTERMEDIATE / TP) as i64, HIDDEN as i64], d_shard_size,
            ));

            if l == 0 || l == NUM_LAYERS - 1 || l == 5 {
                eprintln!("  layer {} ({}) loaded", l, if global { "global" } else { "sliding" });
            } else if l == 1 {
                eprintln!("  ...");
            }
        }

        Gemma4Weights {
            embedding,
            final_norm,
            layer_input_norm,
            layer_post_attn_norm,
            layer_pre_ff_norm,
            layer_post_ff_norm,
            layer_scalar,
            layer_q_proj,
            layer_k_proj,
            layer_v_proj,
            layer_o_proj,
            layer_q_norm,
            layer_k_norm,
            layer_gate_up,
            layer_down,
        }
    }

    /// Find the correct prefix for layer l (handles both
    /// "model.language_model.layers.N" and "model.layers.N")
    fn find_layer_prefix(tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>, l: usize) -> String {
        let prefixed = format!("model.language_model.layers.{}", l);
        let plain = format!("model.layers.{}", l);
        let test_key = format!("{}.input_layernorm.weight", prefixed);
        if tensors.contains_key(&test_key) {
            prefixed
        } else {
            plain
        }
    }

    // -----------------------------------------------------------------------
    // RoPE precomputation
    // -----------------------------------------------------------------------

    struct RopeTables {
        sliding_cos: PjrtBufferHandle, // [max_pos, SLIDING_HEAD_DIM/2] f32
        sliding_sin: PjrtBufferHandle,
        global_cos: PjrtBufferHandle,  // [max_pos, GLOBAL_HEAD_DIM/2] f32
        global_sin: PjrtBufferHandle,
    }

    fn precompute_rope_tables(client: &PjrtClientHandle, max_ctx: usize) -> RopeTables {
        let max_pos = max_ctx;

        // Sliding RoPE: theta=10k, full rotation, rot_dim = SLIDING_HEAD_DIM = 256
        let sliding_half = SLIDING_HEAD_DIM / 2; // 128
        let mut s_cos = vec![0f32; max_pos * sliding_half];
        let mut s_sin = vec![0f32; max_pos * sliding_half];
        for pos in 0..max_pos {
            for i in 0..sliding_half {
                let freq = 1.0 / ROPE_THETA_SLIDING.powf(2.0 * i as f32 / SLIDING_HEAD_DIM as f32);
                let angle = pos as f32 * freq;
                s_cos[pos * sliding_half + i] = angle.cos();
                s_sin[pos * sliding_half + i] = angle.sin();
            }
        }
        let sliding_cos = client
            .buffer_from_host(
                bytemuck::cast_slice(&s_cos),
                &[max_pos as i64, sliding_half as i64],
                PjrtElementType::F32,
                0,
            )
            .unwrap();
        let sliding_sin = client
            .buffer_from_host(
                bytemuck::cast_slice(&s_sin),
                &[max_pos as i64, sliding_half as i64],
                PjrtElementType::F32,
                0,
            )
            .unwrap();

        // Global RoPE: theta=1M, partial rotation
        // rot_dim = GLOBAL_HEAD_DIM * 0.25 = 128, so half = 64
        let global_rot_dim = (GLOBAL_HEAD_DIM as f32 * ROPE_GLOBAL_FRAC) as usize; // 128
        let global_half = global_rot_dim / 2; // 64
        let mut g_cos = vec![0f32; max_pos * global_half];
        let mut g_sin = vec![0f32; max_pos * global_half];
        for pos in 0..max_pos {
            for i in 0..global_half {
                let freq =
                    1.0 / ROPE_THETA_GLOBAL.powf(2.0 * i as f32 / global_rot_dim as f32);
                let angle = pos as f32 * freq;
                g_cos[pos * global_half + i] = angle.cos();
                g_sin[pos * global_half + i] = angle.sin();
            }
        }
        let global_cos = client
            .buffer_from_host(
                bytemuck::cast_slice(&g_cos),
                &[max_pos as i64, global_half as i64],
                PjrtElementType::F32,
                0,
            )
            .unwrap();
        let global_sin = client
            .buffer_from_host(
                bytemuck::cast_slice(&g_sin),
                &[max_pos as i64, global_half as i64],
                PjrtElementType::F32,
                0,
            )
            .unwrap();

        RopeTables {
            sliding_cos,
            sliding_sin,
            global_cos,
            global_sin,
        }
    }

    // -----------------------------------------------------------------------
    // KV cache allocation
    // -----------------------------------------------------------------------

    struct KvCaches {
        // Per-layer KV caches, sharded across TP devices
        // Sliding: [max_ctx, SLIDING_KV_HEADS/TP, SLIDING_HEAD_DIM] bf16
        // Global:  [max_ctx, GLOBAL_KV_HEADS/TP, GLOBAL_HEAD_DIM] bf16
        k_caches: Vec<PjrtBufferHandle>,
        v_caches: Vec<PjrtBufferHandle>,
    }

    fn allocate_kv_caches(client: &PjrtClientHandle, max_ctx: usize) -> KvCaches {
        let mut k_caches = Vec::with_capacity(NUM_LAYERS);
        let mut v_caches = Vec::with_capacity(NUM_LAYERS);

        for l in 0..NUM_LAYERS {
            let global = is_global_layer(l);
            let head_dim = if global { GLOBAL_HEAD_DIM } else { SLIDING_HEAD_DIM };
            let kv_heads = if global { GLOBAL_KV_HEADS } else { SLIDING_KV_HEADS };
            let kv_heads_per_dev = kv_heads / TP;

            let shape = [max_ctx as i64, kv_heads_per_dev as i64, head_dim as i64];
            let nbytes = max_ctx * kv_heads_per_dev * head_dim * 2; // bf16
            let zeros = vec![0u8; nbytes];

            let k = client
                .buffer_from_host(&zeros, &shape, PjrtElementType::BF16, 0)
                .unwrap();
            let v = client
                .buffer_from_host(&zeros, &shape, PjrtElementType::BF16, 0)
                .unwrap();
            k_caches.push(k);
            v_caches.push(v);
        }

        KvCaches { k_caches, v_caches }
    }

    // -----------------------------------------------------------------------
    // Input assembly
    // -----------------------------------------------------------------------

    fn build_inputs<'a>(
        tok_buf: &'a PjrtBufferHandle,
        pos_buf: &'a PjrtBufferHandle,
        ctx_buf: &'a PjrtBufferHandle,
        weights: &'a Gemma4Weights,
        kv_caches: &'a KvCaches,
        rope: &'a RopeTables,
    ) -> Vec<&'a PjrtBufferHandle> {
        // Order must match the StableHLO artifact's parameter convention:
        //   token_id, position, context_len,
        //   embedding, final_norm,
        //   rope_sliding_cos, rope_sliding_sin, rope_global_cos, rope_global_sin,
        //   per-layer weights (60x): [input_norm, post_attn_norm, pre_ff_norm, post_ff_norm,
        //     scalar, q_norm, k_norm, q, k, v, o, gate_up, down],
        //   per-layer KV caches (60x): [k_cache, v_cache]
        let mut inputs: Vec<&PjrtBufferHandle> = Vec::with_capacity(
            3 + 2 + 4 + NUM_LAYERS * 13 + NUM_LAYERS * 2,
        );

        // Scalars
        inputs.push(tok_buf);
        inputs.push(pos_buf);
        inputs.push(ctx_buf);

        // Global weights
        inputs.push(&weights.embedding);
        inputs.push(&weights.final_norm);

        // RoPE tables
        inputs.push(&rope.sliding_cos);
        inputs.push(&rope.sliding_sin);
        inputs.push(&rope.global_cos);
        inputs.push(&rope.global_sin);

        // Per-layer weights
        for l in 0..NUM_LAYERS {
            inputs.push(&weights.layer_input_norm[l]);
            inputs.push(&weights.layer_post_attn_norm[l]);
            inputs.push(&weights.layer_pre_ff_norm[l]);
            inputs.push(&weights.layer_post_ff_norm[l]);
            inputs.push(&weights.layer_scalar[l]);
            inputs.push(&weights.layer_q_norm[l]);
            inputs.push(&weights.layer_k_norm[l]);
            inputs.push(&weights.layer_q_proj[l]);
            inputs.push(&weights.layer_k_proj[l]);
            inputs.push(&weights.layer_v_proj[l]);
            inputs.push(&weights.layer_o_proj[l]);
            inputs.push(&weights.layer_gate_up[l]);
            inputs.push(&weights.layer_down[l]);
        }

        // Per-layer KV caches
        for l in 0..NUM_LAYERS {
            inputs.push(&kv_caches.k_caches[l]);
            inputs.push(&kv_caches.v_caches[l]);
        }

        inputs
    }

    fn update_kv_caches(
        kv_caches: &mut KvCaches,
        outs: &mut std::vec::IntoIter<PjrtBufferHandle>,
    ) {
        // The artifact outputs updated KV caches in layer order: k0, v0, k1, v1, ...
        for l in 0..NUM_LAYERS {
            if let Some(k) = outs.next() {
                kv_caches.k_caches[l] = k;
            }
            if let Some(v) = outs.next() {
                kv_caches.v_caches[l] = v;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Safetensors loading utilities
    // -----------------------------------------------------------------------

    fn load_safetensors_index(
        model_dir: &PathBuf,
    ) -> BTreeMap<String, (Vec<usize>, Vec<u8>, String)> {
        let idx_path = model_dir.join("model.safetensors.index.json");
        let single_path = model_dir.join("model.safetensors");

        let shard_paths: Vec<PathBuf> = if idx_path.exists() {
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
            panic!("no safetensors found in {:?}", model_dir);
        };

        let mut tensor_data: BTreeMap<String, (Vec<usize>, Vec<u8>, String)> = BTreeMap::new();
        for sp in &shard_paths {
            let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(sp).unwrap()).unwrap() };
            let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value =
                serde_json::from_slice(&mmap[8..8 + header_len]).unwrap();
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
                tensor_data.insert(name.clone(), (shape, bytes, dtype));
            }
        }
        tensor_data
    }

    /// Upload a tensor as bf16, trying primary name then fallback name
    fn upload_bf16(
        client: &PjrtClientHandle,
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        primary: &str,
        fallback: &str,
        shape: &[i64],
    ) -> PjrtBufferHandle {
        let name = if tensors.contains_key(primary) {
            primary
        } else {
            fallback
        };
        let (_, bf16) = get_bf16(tensors, name);
        client
            .buffer_from_host(&bf16, shape, PjrtElementType::BF16, 0)
            .unwrap_or_else(|e| panic!("upload {name}: {e}"))
    }

    fn upload_bf16_layer(
        client: &PjrtClientHandle,
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        prefix: &str,
        suffix: &str,
        shape: &[i64],
    ) -> PjrtBufferHandle {
        let name = format!("{}.{}", prefix, suffix);
        let (_, bf16) = get_bf16(tensors, &name);
        client
            .buffer_from_host(&bf16, shape, PjrtElementType::BF16, 0)
            .unwrap_or_else(|e| panic!("upload {name}: {e}"))
    }

    /// Upload bf16 data replicated to device 0 (TP sharding handled by XLA)
    fn upload_sharded_bf16(
        client: &PjrtClientHandle,
        data: &[u8],
        shape: &[i64],
        _expected_shard_size: usize,
    ) -> PjrtBufferHandle {
        // Upload full tensor to device 0; the compiled artifact's sharding
        // annotations (from compile_options_tp4.pb) handle the TP split
        client
            .buffer_from_host(data, shape, PjrtElementType::BF16, 0)
            .unwrap()
    }

    // -----------------------------------------------------------------------
    // Tensor conversion utilities
    // -----------------------------------------------------------------------

    fn get_bf16(
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        name: &str,
    ) -> (Vec<usize>, Vec<u8>) {
        let (shape, bytes, dtype) = tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor: {name}"));
        let bf16 = match dtype.as_str() {
            "BF16" | "bf16" => bytes.clone(),
            "F16" | "f16" => f16_bytes_to_bf16(bytes),
            "F32" | "f32" => f32_bytes_to_bf16(bytes),
            other => panic!("unsupported dtype {other} for {name}"),
        };
        (shape.clone(), bf16)
    }

    fn transpose_2d_bf16(data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let mut out = vec![0u8; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = (r * cols + c) * 2;
                let dst = (c * rows + r) * 2;
                out[dst] = data[src];
                out[dst + 1] = data[src + 1];
            }
        }
        out
    }

    fn transpose_bf16(
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        name: &str,
    ) -> Vec<u8> {
        let (shape, bf16) = get_bf16(tensors, name);
        assert_eq!(shape.len(), 2, "transpose requires 2D tensor: {name}");
        transpose_2d_bf16(&bf16, shape[0], shape[1])
    }

    fn concat_and_transpose_bf16(
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        names: &[&str],
        inner_dim: usize,
    ) -> Vec<u8> {
        let mut total_out = 0usize;
        let mut parts = Vec::new();
        for name in names {
            let (shape, bf16) = get_bf16(tensors, name);
            assert_eq!(shape.len(), 2);
            assert_eq!(shape[1], inner_dim, "inner dim mismatch for {name}");
            total_out += shape[0];
            parts.push(bf16);
        }
        let mut cat = Vec::with_capacity(total_out * inner_dim * 2);
        for data in &parts {
            cat.extend_from_slice(data);
        }
        transpose_2d_bf16(&cat, total_out, inner_dim)
    }

    fn f32_bytes_to_bf16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let v = f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
            out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        out
    }

    fn f16_bytes_to_bf16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            let v = half::f16::from_bits(bits).to_f32();
            out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        out
    }
}
