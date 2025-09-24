use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Dropout, Embedding, Linear, Module, ModuleT, VarBuilder, VarMap, embedding, linear_b,
    ops::softmax,
};
use tiktoken_rs::get_bpe_from_model;

const EPS: f32 = 1e-5;

pub fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let arange = Tensor::arange(0f32, size as f32, device)?; // [0, 1, ..., size-1]
    let row = arange.unsqueeze(0)?; // shape (1, size)
    let col = arange.unsqueeze(1)?; // shape (size, 1)
    let mask = col.broadcast_sub(&row)?.gt(0.0)?; // j > i
    Ok(mask.to_dtype(candle_core::DType::U32)?)
}

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

impl Config {
    fn new(emb_dim: usize, n_heads: usize, n_layers: usize) -> Self {
        Self {
            vocab_size: 50_257,
            context_length: 1_024,
            emb_dim,
            n_heads,
            n_layers,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }

    pub fn gpt2_124m() -> Self {
        Self::new(768, 12, 12)
    }
    pub fn gpt2_medium() -> Self {
        Self::new(1024, 16, 24)
    }
    pub fn gpt2_large() -> Self {
        Self::new(1280, 20, 36)
    }
    pub fn gpt2_xlarge() -> Self {
        Self::new(1600, 25, 48)
    }

    pub fn gpt_sm_test() -> Self {
        Self {
            vocab_size: 50_257,
            context_length: 256,
            emb_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }
}

pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    dim_out: usize,
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    out_proj: Linear,
    dropout_prob: f32,
    dropout: Dropout,
    scaling: f64,
}

impl MultiHeadAttention {
    /// dim_in = number of 'features' of input matrix
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        dropout_prob: f32,
        num_heads: usize,
        qkv_bias: bool,
        vb: &VarBuilder,
    ) -> Result<Self> {
        if dim_out % num_heads != 0 {
            println!("Num Heads: {:?}, Dim Out: {:?}", num_heads, dim_out);
            panic!("Number of heads must be divisible by dim out")
        }

        let head_dim = dim_out / num_heads;

        let w_query = linear_b(dim_in, dim_out, qkv_bias, vb.pp("w_query"))?;
        let w_key = linear_b(dim_in, dim_out, qkv_bias, vb.pp("w_key"))?;
        let w_value: Linear = linear_b(dim_in, dim_out, qkv_bias, vb.pp("w_value"))?;

        let out_proj = linear_b(dim_out, dim_out, qkv_bias, vb.pp("out_proj"))?;

        // denominator
        let power = 0.5;
        let scaling = 1. / (head_dim as f64).powf(power);

        let dropout = Dropout::new(dropout_prob);

        let mha = Self {
            dim_out,
            head_dim,
            num_heads,
            dropout,
            dropout_prob,
            w_key,
            w_value,
            w_query,
            scaling,
            out_proj,
        };
        Ok(mha)
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b, num_tokens, _d_in) = xs.dims3()?;

        let query = self.w_query.forward_t(xs, train)?;
        let key = self.w_key.forward_t(xs, train)?;
        let values = self.w_value.forward_t(xs, train)?;

        // split matrices into multiple heads
        let query = query
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = key
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let values = values
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let key_t = &key.transpose(D::Minus2, D::Minus1)?;

        let attention_scores = query.matmul(key_t)?;

        let mask = get_mask(num_tokens, xs.device())?;
        let masked_tensor = masked_fill(
            &attention_scores,
            &mask.broadcast_left((b, self.num_heads)).unwrap(),
            f32::NEG_INFINITY,
        )?;

        // probabilistic set of numbers
        let mut attention_weights = softmax(&(masked_tensor * self.scaling)?, D::Minus1)?;

        attention_weights = self.dropout.forward(&attention_weights, train)?;

        let context_vec = attention_weights.matmul(&values)?.transpose(1, 2)?;
        let context_vec = context_vec
            .reshape((b, num_tokens, self.dim_out))?
            .contiguous()?;

        self.out_proj.forward_t(&context_vec, train)
    }
}

pub struct GELU;

impl Module for GELU {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<Tensor> {
        let x3 = xs.mul(xs)?.mul(xs)?; // x^3
        let inner = xs + &(x3 * 0.044715f64)?;
        let scaled = (2.0 / std::f64::consts::PI).powf(0.5) * inner?;
        let tanh_term = scaled?.tanh();
        let ones = Tensor::ones((1,), candle_core::DType::F32, xs.device())?;
        let factor = tanh_term?.broadcast_add(&ones)?; // 1+tanh(...)
        let result = 0.5f64 * xs * factor;
        Ok(result?)
    }
}

// Fixed sandwich layering - Linear - GELU - Linear
pub struct FeedForward {
    top: Linear,
    bottom: Linear,
}

impl FeedForward {
    pub fn new(cfg: Config, vb: &VarBuilder) -> Result<Self> {
        let expansion_factor = 4_usize;
        let hidden_dim = expansion_factor * cfg.emb_dim;
        let top = linear_b(cfg.emb_dim, hidden_dim, true, vb.pp("ff_top"))?;
        let bottom = linear_b(hidden_dim, cfg.emb_dim, true, vb.pp("ff_bottom"))?;
        let ff = Self { top, bottom };

        Ok(ff)
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.top.forward(xs)?;
        let xs = GELU.forward(&xs)?;
        let xs = self.bottom.forward(&xs)?;

        Ok(xs)
    }
}

pub struct LayerNorm {
    eps: f32,
    scale: Tensor,
    shift: Tensor,
}

impl LayerNorm {
    pub fn new(emb_dim: usize, vb: &VarBuilder) -> Result<Self> {
        let scale = vb.get_with_hints(emb_dim, "scale", candle_nn::Init::Const(1.))?;
        let shift = vb.get_with_hints(emb_dim, "shift", candle_nn::Init::Const(0.))?;

        let layer_norm = Self {
            eps: EPS,
            scale,
            shift,
        };

        Ok(layer_norm)
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = xs.mean_keepdim(D::Minus1)?;
        let variance = xs.var_keepdim(D::Minus1)?;
        // prevent division by zero
        let eps = Tensor::new(&[self.eps], xs.device())?;

        // (x - mean) / (var + epsilon)
        let dividend = xs.broadcast_sub(&mean)?;
        let divisor = variance.broadcast_add(&eps)?;
        let divisor_sqrt = divisor.sqrt()?;
        let normed = dividend.broadcast_div(&divisor_sqrt)?;

        let scaled_shifted = normed
            .broadcast_mul(&self.scale)?
            .broadcast_add(&self.shift)?;

        Ok(scaled_shifted)
    }
}

pub struct TransformerBlock {
    mha: MultiHeadAttention,
    ff: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(cfg: Config, vb: &VarBuilder) -> Result<Self> {
        let mha = MultiHeadAttention::new(
            cfg.emb_dim,
            cfg.emb_dim,
            cfg.drop_rate,
            cfg.n_heads,
            cfg.qkv_bias,
            &vb.pp("mha"),
        )?;

        let ff = FeedForward::new(cfg, &vb.pp("ff"))?;

        let layer_norm1 = LayerNorm::new(cfg.emb_dim, &vb.pp("layer_norm1"))?;
        let layer_norm2 = LayerNorm::new(cfg.emb_dim, &vb.pp("layer_norm2"))?;
        let dropout = Dropout::new(cfg.drop_rate);
        Ok(Self {
            mha,
            ff,
            layer_norm1,
            layer_norm2,
            dropout,
        })
    }
}

impl ModuleT for TransformerBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let shortcut = xs.clone();
        let mut x = xs.to_owned();
        x = self.layer_norm1.forward(&x)?;
        x = self.mha.forward(&x, train)?;
        x = self.dropout.forward(&x, train)?;
        x = (x + shortcut)?;

        let shortcut = x.clone();
        x = self.layer_norm2.forward(&x)?;
        x = self.ff.forward(&x)?;
        x = self.dropout.forward(&x, train)?;
        x = (x + shortcut)?;
        Ok(x)
    }
}

pub struct TransformerBlocks {
    blocks: Vec<TransformerBlock>,
}

impl TransformerBlocks {
    pub fn new() -> Self {
        Self { blocks: vec![] }
    }

    pub fn add_block(mut self, layer: TransformerBlock) -> Self {
        self.blocks.push(layer);
        self
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}

impl ModuleT for TransformerBlocks {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.blocks.iter() {
            xs = layer.forward_t(&xs, train)?;
        }

        Ok(xs)
    }
}

pub trait GPT {
    fn context_size(&self) -> usize;
}

pub struct GPTModel {
    token_emb: Embedding,
    pos_emb: Embedding,
    dropout: Dropout,
    transformer_blocks: TransformerBlocks,
    final_layer_norm: LayerNorm,
    linear_output_layer: Linear,
}

impl GPTModel {
    pub fn new(cfg: Config, vb: &VarBuilder) -> Result<Self> {
        let token_emb: Embedding = embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("token_emb"))?;
        let pos_emb = embedding(cfg.context_length, cfg.emb_dim, vb.pp("pos_emb"))?;
        let dropout = Dropout::new(cfg.drop_rate);

        let mut transformer_blocks = TransformerBlocks::new();

        for idx in 0..cfg.n_layers {
            let storage_name = format!("transformer_block_{}", idx);
            let block = TransformerBlock::new(cfg, &vb.pp(storage_name))?;
            transformer_blocks = transformer_blocks.add_block(block);
        }

        let final_layer_norm = LayerNorm::new(cfg.emb_dim, &vb.pp("final_layer_norm"))?;
        let linear_output_layer = linear_b(
            cfg.emb_dim,
            cfg.vocab_size,
            false,
            vb.pp("linear_output_layer"),
        )?;

        let gpt = Self {
            linear_output_layer,
            final_layer_norm,
            transformer_blocks,
            token_emb,
            pos_emb,
            dropout,
        };

        Ok(gpt)
    }

    pub fn set_linear_output_layer(&mut self, new_linear_output_layer: Linear) {
        self.linear_output_layer = new_linear_output_layer
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs, true)
    }
}

impl ModuleT for GPTModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (_batch_size, seq_len) = xs.dims2()?;
        let token_emb = self.token_emb.forward(xs)?;
        let pos_ids = Tensor::arange(0, seq_len as u32, xs.device())?;
        let pos_embed = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;

        let mut x = token_emb.broadcast_add(&pos_embed)?;
        x = self.dropout.forward(&x, train)?;
        x = self.transformer_blocks.forward_t(&x, train)?;
        x = self.final_layer_norm.forward(&x)?;

        let result = self.linear_output_layer.forward(&x)?;

        Ok(result)
    }
}

impl GPT for GPTModel {
    fn context_size(&self) -> usize {
        self.pos_emb.embeddings().dims()[0]
    }
}

pub fn generate_text_simple<M: GPT + ModuleT>(
    model: &M,
    mut tokens: Tensor,    // starting sequence of token IDs
    max_new_tokens: usize, // how many tokens to generate
    context_size: usize,   // window size (for GPT context)
) -> Result<Tensor> {
    for _ in 0..max_new_tokens {
        // Current sequence length (batch_size, seq_len)
        let (_batch, seq_len) = tokens.dims2()?;

        // Truncate to the last `context_size` tokens
        let start = seq_len.saturating_sub(context_size);
        let context = tokens.i((.., start..seq_len))?;

        // Forward pass → get logits
        let logits = model.forward_t(&context, false)?;

        // Grab logits for the last token position
        // batch = its means how many sequence of words or sentences are you feeding
        // seq_len = number of "words", "vocab" is the number of words in vocab (e.g tiktoken has 50k words / chars inside it)
        let (_batch, seq_len, _vocab) = logits.dims3()?;
        let last_logits = logits.i((.., seq_len - 1, ..))?;

        // Convert to probabilities and pick the most likely token
        let probs = softmax(&last_logits, 1)?;
        let next_token = probs.argmax_keepdim(D::Minus1)?;

        // Append new token to the sequence
        tokens = Tensor::cat(&[&tokens, &next_token], D::Minus1)?;
    }

    Ok(tokens)
}

pub fn run_generate_text_simple() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    let start_context = "Every breath you take";
    let tokenizer =
        get_bpe_from_model("gpt2").map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let encoded = tokenizer.encode_with_special_tokens(start_context);
    let num_tokens = encoded.len();
    println!("encoded: {:?}", encoded);
    let encoded_tensor = Tensor::from_vec(encoded, (1_usize, num_tokens), &dev)?;
    println!("encoded_tensor.shape {:?}", encoded_tensor);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = Config::gpt2_124m();
    let model = GPTModel::new(cfg, &vb)?;

    let out = generate_text_simple(&model, encoded_tensor, 6_usize, cfg.context_length)?;
    println!("Output: {:?}", out.to_vec2::<u32>());
    println!("Output length: {}", out.dims()[1]);

    let decoded_text = tokenizer.decode(out.reshape(out.dims()[1])?.to_vec1::<u32>()?);
    println!("Decoded Text: {:?}", decoded_text);
    Ok(())
}
