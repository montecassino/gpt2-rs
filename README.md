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


## How to Run

### Prerequisites
- Rust (1.70 or later)
- Cargo (comes with Rust)

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

## Configuration

The project uses default hyperparameters for GPT-2. You can modify these in the respective source files:
- `src/gpt.rs` - Model architecture and parameters
- `src/train.rs` - Training configuration
- `src/main.rs` - Main application logic

## Data Format

The training data should be in text format. The project currently uses `data/the-verdict.txt` as sample data, but you can replace it with your own dataset.

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