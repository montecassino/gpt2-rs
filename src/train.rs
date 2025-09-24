use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, ParamsAdamW, loss::cross_entropy};
use candle_nn::{VarBuilder, VarMap, ModuleT, AdamW};
use tiktoken_rs::get_bpe_from_model;
use anyhow::Result;

use crate::gpt::{generate_text_simple, Config, GPTModel};

pub fn train_gpt_simple() -> Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;
    println!("Device used: {:?}", dev);
    let cfg = Config::gpt_sm_test();
    let batch_size = 1;
    let sequence_length = 256;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = GPTModel::new(cfg, &vb)?;

    // 2. Efficient Data Loading and Preparation
    let tokenizer = get_bpe_from_model("gpt2")
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let file_content = std::fs::read_to_string("data/the-verdict.txt")
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let all_tokens: Vec<u32> = tokenizer
        .encode_with_special_tokens(&file_content)
        .into_iter()
        .map(|x| x as u32)
        .collect();

    let total_batches = (all_tokens.len() / (batch_size * sequence_length)).max(1);
    let mut prebaked_batches = Vec::with_capacity(total_batches);
    for batch_idx in 0..total_batches {
        let start_idx = batch_idx * batch_size * sequence_length;
        if start_idx + batch_size * sequence_length >= all_tokens.len() {
            break;
        }
        let (input_batch, target_batch) = create_simple_batch(
            &all_tokens,
            batch_size,
            sequence_length,
            start_idx,
            &dev,
        )?;
        prebaked_batches.push((input_batch, target_batch));
    }
    println!("Pre-computed {} batches ready for training", prebaked_batches.len());

    // 3. Model and Optimizer Setup
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 0.0004,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;

    let num_epochs = 25; 
    println!("Starting training...");
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut num_steps = 0;

        for (input_batch, target_batch) in prebaked_batches.iter() {
            let logits = model.forward_t(input_batch, true)?;
            let batch_size = input_batch.dims()[0];
            let seq_len = input_batch.dims()[1];
            let vocab_size = cfg.vocab_size;

            let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
            let targets_flat = target_batch.reshape((batch_size * seq_len,))?;
            
            let loss = cross_entropy(&logits_flat, &targets_flat)?;
            
            optimizer.backward_step(&loss)?;
            
            total_loss += loss.to_scalar::<f32>()?;
            num_steps += 1;
        }
        let avg_loss = if num_steps > 0 { total_loss / num_steps as f32 } else { 0.0 };
        println!("Epoch {} finished, avg_loss: {:.4}", epoch + 1, avg_loss);

        quick_generation_test(&model, &all_tokens, &tokenizer, &dev, sequence_length, &cfg)?;
    }

    println!("Training completed!");

    let custom_prompt = "Every breath you";
    let custom_tokens: Vec<u32> = tokenizer
        .encode_with_special_tokens(custom_prompt)
        .into_iter()
        .map(|x| x as u32)
        .collect();

    let custom_input = Tensor::from_vec(custom_tokens.clone(), (1, custom_tokens.len()), &dev)?;
    println!("--- Custom generation for: \"{}\" ---", custom_prompt);

    let custom_generated = generate_text_simple(&model, custom_input, 100, cfg.context_length)?;
    let custom_generated_vec = custom_generated
        .reshape(custom_generated.dims()[1])?
        .to_vec1::<u32>()?;
    let custom_text = tokenizer.decode(custom_generated_vec)?;
    println!("{}", custom_text);
    println!("-----------------------------------");

    Ok(())
}

fn create_simple_batch(
    tokens: &[u32],
    batch_size: usize,
    sequence_length: usize,
    start_idx: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut input_batch = Vec::new();
    let mut target_batch = Vec::new();

    for i in 0..batch_size {
        let idx = start_idx + i * sequence_length;
        input_batch.extend_from_slice(&tokens[idx..idx + sequence_length]);
        target_batch.extend_from_slice(&tokens[idx + 1..idx + 1 + sequence_length]);
    }

    let input_tensor = Tensor::from_vec(input_batch, (batch_size, sequence_length), device)?;
    let target_tensor = Tensor::from_vec(target_batch, (batch_size, sequence_length), device)?;

    Ok((input_tensor, target_tensor))
}

fn quick_generation_test(
    model: &GPTModel,
    tokens: &[u32],
    tokenizer: &tiktoken_rs::CoreBPE,
    device: &Device,
    seq_len: usize,
    cfg: &Config,
) -> Result<()> {
    let sample_len = 20;
    let sample_tokens: Vec<u32> = tokens[0..sample_len].to_vec();
    let sample_input = Tensor::from_vec(sample_tokens, (1, sample_len), device)?;
    
    println!("--- Generating text... ---");
    let generated_tokens = generate_text_simple(model, sample_input, 100, cfg.context_length)?;
    
    let generated_tokens_vec = generated_tokens.reshape(generated_tokens.dims()[1])?.to_vec1::<u32>()?;
    let text = tokenizer.decode(generated_tokens_vec)?;
    println!("{}", text);
    println!("--------------------------");
    
    Ok(())
}