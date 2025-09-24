# GPT-2 Implementation in Rust

This project is a Rust implementation of the GPT-2 language model, designed for training and inference. It provides a foundation for understanding transformer architectures and demonstrates how to build language models from scratch using Rust.

## Project Structure

```
gpt2-rs/
├── Cargo.toml          # Project configuration and dependencies
├── src/
│   ├── main.rs         # Entry point of the application
│   ├── gpt.rs          # Core GPT-2 model implementation
│   └── train.rs        # Training utilities and logic
├── data/
│   └── the-verdict.txt # Sample training data
└── .gitignore          # Git ignore rules
```

## Features

- **GPT-2 Model Implementation**: Core transformer architecture (decoder only) with attention mechanisms
- **Training Capabilities**: Training functionality for language models
- **Inference Support**: Ability to generate text using trained models
- **Rust-based**: Fast, memory-safe implementation using Rust
- **CUDA Support**: Optional GPU acceleration through CUDA

## How to Run

### Prerequisites
- Rust (1.70 or later)
- Cargo (comes with Rust)
- For CUDA support: NVIDIA CUDA toolkit and compatible GPU

### Building and Running

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd gpt2-rs
   ```

2. **Build the project**:
   ```bash
   cargo build
   ```

3. **Run the application**:
   ```bash
   cargo run
   ```

### Training

To train the model, you can use:
```bash
cargo run --features cuda -- train
```

The training data is located in `data/the-verdict.txt`.

### Inference

To generate text using a trained model:
```bash
cargo run --features cuda -- generate
```

## Configuration

The project uses default hyperparameters for GPT-2. You can modify these in the respective source files:
- `src/gpt.rs` - Model architecture and parameters
- `src/train.rs` - Training configuration
- `src/main.rs` - Main application logic

## Data Format

The training data should be in text format. The project currently uses `data/the-verdict.txt` as sample data, but you can replace it with your own dataset.

## Model Variants

The implementation supports different GPT-2 model sizes:
- `gpt2_124m()` - 124M parameters (default)
- `gpt2_medium()` - 355M parameters
- `gpt2_large()` - 762M parameters
- `gpt2_xlarge()` - 1542M parameters

## Contributing

This is a learning project demonstrating GPT-2 implementation in Rust. Contributions are welcome for:
- Improving model performance
- Adding new features
- Optimizing code
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Note

This implementation is educational and demonstrates core concepts of transformer models. For production use, consider using optimized libraries like Hugging Face Transformers or similar established implementations.

## Technical Details

This implementation uses:
- [Candle](https://github.com/huggingface/candle) - A fast machine learning framework in Rust
- [tiktoken-rs](https://github.com/gezakovacs/tiktoken-rs) - Rust implementation of OpenAI's tiktoken tokenizer
- CUDA support for GPU acceleration (optional)

The model architecture includes:
- Multi-head attention mechanism
- Layer normalization
- Feed-forward networks with GELU activation
- Positional embeddings
- Dropout for regularization

The training process uses AdamW optimizer with cross-entropy loss.
