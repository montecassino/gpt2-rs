use std::env;

mod gpt;
mod train;
use anyhow::Result;

// If error is  nvcc cannot target gpu arch xxx. Available nvcc targets are [50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90].
// then do export CUDA_COMPUTE_CAP=<Available nvcc targets>
// cargo run --features cuda -- train

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("train") => train::train_gpt_simple()?,
        Some("generate") => gpt::run_generate_text_simple()?,
        _ => {
            println!("Usage: cargo run --features cuda -- <train|generate>");
        }
    }

    Ok(())
}
