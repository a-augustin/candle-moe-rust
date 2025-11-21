use candle_moe_rust::qwen3_moe_optimized;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;

fn main() -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    
    let cfg = qwen3_moe_optimized::Config {
        vocab_size: 1000,
        hidden_size: 1024,
        intermediate_size: 4096,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        head_dim: 128,
        attention_bias: false,
        num_key_value_heads: 8,
        max_position_embeddings: 4096,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
        // MoE specific configuration
        decoder_sparse_step: 1,
        moe_intermediate_size: 4096,
        num_experts_per_tok: 2,
        num_experts: 8,
        norm_topk_prob: false,
    };

    let vb = VarBuilder::zeros(DType::F32, &dev);

    // Instantiate base model
    let mut model = qwen3_moe_optimized::ModelForCausalLM::new(&cfg, vb.clone())?;

    // Fake input
    let input = Tensor::zeros((1, 1), DType::U32, &dev)?;
    let out = model.forward(&input, 0)?;
    
    println!("Output: {:?}", out);

    Ok(())
}