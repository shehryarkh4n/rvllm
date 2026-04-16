//! Standalone v2 benchmark binary.
//!
//! Usage:
//!   cargo run -p rvllm-v2 --features cuda --bin bench -- \
//!     --model meta-llama/Llama-3-8B \
//!     --n 1,32,64,96,128 \
//!     --output-len 512

use std::time::Instant;

use clap::Parser;
use rvllm_core::prelude::SamplingParams;

use rvllm_v2::engine::StepTimings;
use rvllm_v2::integration::{init_engine, V2EngineConfig};

const BENCH_PROMPTS: &[&str] = &[
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list of integers.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Write a short story about a robot learning to paint.",
    "Explain how a transformer neural network works.",
    "What are the advantages of Rust over C++?",
    "Describe the water cycle in detail.",
    "Write a haiku about machine learning.",
    "Explain the concept of recursion with an example.",
    "What is the difference between a stack and a queue?",
    "Describe how HTTPS encryption works.",
    "Write a SQL query to find duplicate records in a table.",
    "Explain the CAP theorem in distributed systems.",
    "What are the main principles of object-oriented programming?",
    "Describe the architecture of a modern CPU.",
    "Write a regular expression to validate email addresses.",
    "Explain how garbage collection works in Java.",
    "What is the difference between concurrency and parallelism?",
    "Describe the MapReduce programming model.",
    "Explain how a B-tree index works in databases.",
    "What are the trade-offs between microservices and monoliths?",
    "Describe the process of DNS resolution.",
    "Write pseudocode for the A* pathfinding algorithm.",
];

#[derive(Parser)]
#[command(name = "rvllm-v2-bench", about = "v2 engine throughput benchmark")]
struct Cli {
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "1,32,64,96,128")]
    n: String,

    #[arg(long, default_value_t = 512)]
    output_len: usize,

    #[arg(long, default_value_t = 3)]
    iters: usize,

    #[arg(long)]
    num_gpu_blocks: Option<usize>,

    #[arg(long)]
    num_cpu_blocks: Option<usize>,

    #[arg(long, default_value_t = 0.90)]
    gpu_memory_utilization: f32,

    #[arg(long, default_value_t = 0.0)]
    gpu_memory_reserve_gb: f32,

    #[arg(long)]
    max_model_len: Option<usize>,

    #[arg(long)]
    json: bool,

    #[arg(long)]
    fp8: bool,

    #[arg(long, help = "Enable CUPTI GPU kernel profiling and CPU timing breakdown")]
    profile: bool,
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();

    let batch_sizes: Vec<usize> = cli
        .n
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let iters = cli.iters.max(1);

    // Profile mode: disable CUDA graphs so CUPTI can see individual kernel launches
    if cli.profile {
        std::env::set_var("RVLLM_NO_GRAPH", "1");
        eprintln!("Profile mode: CUDA graphs disabled for CUPTI visibility");
    }

    if batch_sizes.len() > 1 || iters > 1 {
        eprintln!("rvLLM v2 benchmark -- direct engine, no HTTP");
        eprintln!(
            "model: {}  output_len: {}  iters: {}",
            cli.model, cli.output_len, iters
        );
    }

    let mut json_results = Vec::new();

    // Create engine ONCE -- max_num_seqs=256 covers all batch sizes.
    // CUDA graph captures happen lazily per batch size bucket.
    let max_batch = *batch_sizes.iter().max().unwrap_or(&256);
    let config = V2EngineConfig {
        num_gpu_blocks_override: cli.num_gpu_blocks,
        num_cpu_blocks_override: cli.num_cpu_blocks,
        gpu_memory_utilization: cli.gpu_memory_utilization,
        gpu_memory_reserve_gb: cli.gpu_memory_reserve_gb,
        max_model_len: cli.max_model_len.unwrap_or(2048),
        max_num_seqs: max_batch.max(256),
        fp8_weights: cli.fp8,
        ..V2EngineConfig::from_model_path(&cli.model)
    };

    let mut engine = init_engine(&config);

    for &batch in &batch_sizes {
        // Warmup: use same output_len as bench to prime identical kernel paths
        let warmup_params = SamplingParams {
            temperature: 0.0,
            max_tokens: cli.output_len,
            ignore_eos: true,
            ..Default::default()
        };
        for i in 0..batch {
            let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()].to_string();
            engine.add_request(prompt, warmup_params.clone())?;
        }
        while engine.has_pending_work() {
            engine.step().map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        engine.sync().map_err(|e| anyhow::anyhow!("{e}"))?;

        // Profile run: CUPTI GPU profiling + CPU timing (separate from throughput measurement)
        if cli.profile {
            run_profiled_iteration(&mut engine, batch, &cli)?;
        }

        // Bench runs
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: cli.output_len,
            ignore_eos: true,
            ..Default::default()
        };

        let mut tps_samples = Vec::with_capacity(iters);

        for _ in 0..iters {
            for i in 0..batch {
                let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()].to_string();
                engine
                    .add_request(prompt, params.clone())
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }

            let t0 = Instant::now();
            let mut total_tokens = 0usize;

            while engine.has_pending_work() {
                let outputs = engine.step_pipelined().map_err(|e| anyhow::anyhow!("{e}"))?;
                total_tokens += outputs.len();
            }
            // Flush the last pipelined step
            let final_outputs = engine.step_pipelined_flush().map_err(|e| anyhow::anyhow!("{e}"))?;
            total_tokens += final_outputs.len();

            engine.sync().map_err(|e| anyhow::anyhow!("{e}"))?;
            let elapsed = t0.elapsed();
            let tps = total_tokens as f64 / elapsed.as_secs_f64();
            tps_samples.push(tps);
        }

        let mean_tps = tps_samples.iter().sum::<f64>() / tps_samples.len() as f64;
        let min_tps = tps_samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_tps = tps_samples
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let stdev = if tps_samples.len() > 1 {
            let variance =
                tps_samples.iter().map(|&x| (x - mean_tps).powi(2)).sum::<f64>()
                    / (tps_samples.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        if cli.json {
            json_results.push(serde_json::json!({
                "n": batch,
                "iters": iters,
                "mean_tok_per_sec": (mean_tps * 10.0).round() / 10.0,
                "min_tok_per_sec": (min_tps * 10.0).round() / 10.0,
                "max_tok_per_sec": (max_tps * 10.0).round() / 10.0,
                "stdev": (stdev * 10.0).round() / 10.0,
                "samples": tps_samples.iter().map(|&x| (x * 10.0).round() / 10.0).collect::<Vec<_>>(),
            }));
        } else {
            eprintln!(
                "N={:<4} {:>8.1} tok/s  (min={:.1} max={:.1} stdev={:.1}, {} iters)",
                batch, mean_tps, min_tps, max_tps, stdev, iters,
            );
        }
    }

    if cli.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "engine": "rvllm-v2",
                "model": cli.model,
                "output_len": cli.output_len,
                "iters": iters,
                "results": json_results,
            }))?
        );
    }

    Ok(())
}

fn run_profiled_iteration(
    engine: &mut rvllm_v2::integration::ConcreteEngine,
    batch: usize,
    cli: &Cli,
) -> anyhow::Result<()> {
    eprintln!("\n--- Profile run: N={batch} ---");

    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: cli.output_len,
        ignore_eos: true,
        ..Default::default()
    };

    for i in 0..batch {
        let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()].to_string();
        engine
            .add_request(prompt, params.clone())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }

    // Try CUPTI profiling
    let mut profiler = match rvllm_autotune::profiler::CuptiProfiler::new() {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("CUPTI profiler init failed (will skip GPU profiling): {e}");
            None
        }
    };

    if let Some(ref mut p) = profiler {
        if let Err(e) = p.start() {
            eprintln!("CUPTI start failed: {e}");
            profiler = None;
        }
    }

    // Run one full iteration with CPU timing
    let mut total_tokens = 0usize;
    let mut step_count = 0u64;
    let mut acc_timings = StepTimings::default();

    while engine.has_pending_work() {
        let (outputs, timings) = engine
            .step_profiled()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        total_tokens += outputs.len();
        step_count += 1;
        acc_timings.scheduler_us += timings.scheduler_us;
        acc_timings.forward_us += timings.forward_us;
        acc_timings.output_us += timings.output_us;
        acc_timings.total_us += timings.total_us;
    }

    engine.sync().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Stop CUPTI and print GPU kernel breakdown
    if let Some(ref mut p) = profiler {
        if let Err(e) = p.stop() {
            eprintln!("CUPTI stop failed: {e}");
        } else {
            eprintln!("\nGPU Kernel Breakdown (N={batch}, {total_tokens} tokens, {step_count} steps):");
            p.print_summary(25);
        }
    }

    // Print CPU timing breakdown
    if step_count > 0 {
        eprintln!("\nCPU Timing (N={batch}, {step_count} steps):");
        eprintln!(
            "  Scheduler:     {:.2} ms/step",
            acc_timings.scheduler_us as f64 / step_count as f64 / 1000.0
        );
        eprintln!(
            "  Forward (GPU): {:.2} ms/step",
            acc_timings.forward_us as f64 / step_count as f64 / 1000.0
        );
        eprintln!(
            "  Output proc:   {:.2} ms/step",
            acc_timings.output_us as f64 / step_count as f64 / 1000.0
        );
        eprintln!(
            "  Total:         {:.2} ms/step",
            acc_timings.total_us as f64 / step_count as f64 / 1000.0
        );
        eprintln!(
            "  Step rate:     {:.1} steps/sec",
            step_count as f64 / (acc_timings.total_us as f64 / 1_000_000.0)
        );
    }

    eprintln!("--- End profile run ---\n");
    Ok(())
}
