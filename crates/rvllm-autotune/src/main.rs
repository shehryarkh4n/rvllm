mod ffn;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "rvllm-autotune")]
#[command(about = "Autotuning and sweep tools for rvLLM")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Sweep FFN runtime candidates across decode shapes with correctness gating.
    FfnSweep(ffn::FfnSweepArgs),
    /// Sweep Hopper gate-aux CUTLASS build configs across decode shapes.
    FfnKernelSweep(ffn::FfnKernelSweepArgs),
}

fn main() -> Result<(), String> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .compact()
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::FfnSweep(args) => ffn::run_ffn_sweep(&args),
        Command::FfnKernelSweep(args) => ffn::run_ffn_kernel_sweep(&args),
    }
}
