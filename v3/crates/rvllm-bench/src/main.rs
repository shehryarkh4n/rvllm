//! rvllm-bench: loads a model + kernels + cutlass .so + fa3 .so, runs
//! `iters` decode-step forwards on a fixed-batch bucket, and reports
//! tokens/sec.
//!
//! Env vars (all required for cuda path):
//!   RVLLM_MODEL_DIR   = HF snapshot dir with config.json + safetensors
//!   RVLLM_KERNELS_DIR = dir with manifest.json + compiled PTX
//!   RVLLM_CUTLASS_SO  = path to libcutlass_kernels.so
//!   RVLLM_FA3_SO      = path to libfa3_kernels.so
//!   RVLLM_POLICY      = path to policy.json
//!   RVLLM_BATCH       = batch size (default 128)
//!   RVLLM_ITERS       = decode-step iterations (default 100)
//!   RVLLM_WARMUP      = warmup iterations (default 10)
//!
//! Prints one JSON line per run: {batch, iters, tok_per_sec, ms_per_step}.

use std::path::PathBuf;
use std::time::Instant;

use rvllm_runtime::{Bringup, EnginePaths};

fn env_path(k: &str) -> Result<PathBuf, String> {
    std::env::var(k)
        .map_err(|_| format!("missing env var: {k}"))
        .map(PathBuf::from)
}

fn env_u32(k: &str, default: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let result = run();
    if let Err(e) = result {
        eprintln!("rvllm-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let paths = EnginePaths {
        model_dir: env_path("RVLLM_MODEL_DIR")?,
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path("RVLLM_CUTLASS_SO")?,
        fa3_so: env_path("RVLLM_FA3_SO")?,
        policy_json: env_path("RVLLM_POLICY")?,
    };
    let batch = env_u32("RVLLM_BATCH", 128);
    let iters = env_u32("RVLLM_ITERS", 100);
    let warmup = env_u32("RVLLM_WARMUP", 10);

    // Arena budget: model (~16 GB fp8) + kv (~8 GB) + scratch/workspace (~4 GB).
    let arena_bytes: usize = 32 * 1024 * 1024 * 1024;

    eprintln!("== rvllm-bench v3 ==");
    eprintln!("model_dir   = {}", paths.model_dir.display());
    eprintln!("kernels_dir = {}", paths.kernels_dir.display());
    eprintln!("batch       = {batch}");
    eprintln!("iters       = {iters} (warmup {warmup})");

    let t0 = Instant::now();
    let br = Bringup::load(paths, arena_bytes).map_err(|e| format!("bringup: {e}"))?;
    eprintln!(
        "bringup: {:.2}s | arch layers={} hidden={} heads={} kv_heads={}",
        t0.elapsed().as_secs_f64(),
        br.arch.num_hidden_layers,
        br.arch.hidden_size,
        br.arch.num_attention_heads,
        br.arch.num_key_value_heads,
    );
    eprintln!("arena used = {} MiB", br.arena.used() / (1024 * 1024));

    let result = unsafe { br.run_bench(batch, iters, warmup) }
        .map_err(|e| format!("run_bench: {e}"))?;

    let tok_per_sec = if result.total_ns > 0 {
        (result.iters as f64 * result.num_seqs as f64) * 1.0e9 / result.total_ns as f64
    } else {
        0.0
    };
    let ms_per_step = result.ns_per_step as f64 / 1.0e6;

    eprintln!(
        "bench: batch={} iters={} -> {:.0} tok/s ({:.3} ms/step)",
        batch, iters, tok_per_sec, ms_per_step
    );
    println!(
        "{{\"batch\":{},\"iters\":{},\"tok_per_sec\":{:.1},\"ms_per_step\":{:.4}}}",
        batch, iters, tok_per_sec, ms_per_step
    );
    Ok(())
}
