#![allow(dead_code)]

use crate::qwen3_base::{
    Config as Qwen3Config,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
};
use candle_transformers::models::{
    with_tracing::{linear_no_bias, Linear, RmsNorm},
};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;
use candle_transformers::models::deepseek2::NonZeroOp;
use candle_transformers::models::deepseek2::TopKLastDimOp;
use candle_core::IndexOp;


#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
    // MoE specific configuration
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    pub norm_topk_prob: bool,
}

impl From<&Config> for Qwen3Config {
    fn from(val: &Config) -> Self {
        Qwen3Config {
            vocab_size: val.vocab_size,
            hidden_size: val.hidden_size,
            intermediate_size: val.intermediate_size,
            num_hidden_layers: val.num_hidden_layers,
            num_attention_heads: val.num_attention_heads,
            head_dim: val.head_dim,
            attention_bias: val.attention_bias,
            num_key_value_heads: val.num_key_value_heads,
            max_position_embeddings: val.max_position_embeddings,
            sliding_window: val.sliding_window,
            max_window_layers: val.max_window_layers,
            tie_word_embeddings: val.tie_word_embeddings,
            rope_theta: val.rope_theta,
            rms_norm_eps: val.rms_norm_eps,
            use_sliding_window: val.use_sliding_window,
            hidden_act: val.hidden_act,
        }
    }
}

#[derive(Debug, Clone)]
struct Qwen3MLPExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLPExpert {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.moe_intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLPExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// Qwen3 Sparse MoE Block implementation
#[derive(Debug, Clone)]
struct Qwen3SparseMoeBlock {
    gate: Linear,
    experts: Vec<Qwen3MLPExpert>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl Qwen3SparseMoeBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate = linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(cfg.num_experts);
        let vb_e = vb.pp("experts");
        for idx in 0..cfg.num_experts {
            let expert = Qwen3MLPExpert::new(cfg, vb_e.pp(idx))?;
            experts.push(expert)
        }
        Ok(Self {
            gate,
            experts,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }
}
// OPTIMIZATION: GPU qwen3 MoE forward
impl Module for Qwen3SparseMoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let total_tokens = b_size * seq_len;

        if total_tokens == 0 {
            return Ok(xs.clone());
        }

        let xs_flat = xs.reshape((total_tokens, hidden_dim))?;
        let router_logits = xs_flat.apply(&self.gate)?; // (T, E)

        let routing_weights = candle_nn::ops::softmax(&router_logits, D::Minus1)?; // (T, E)

        // === Fixed topk: uses TopKOutput struct ===
        let topk_res = routing_weights.topk(self.num_experts_per_tok)?; // TopKOutput
        let mut topk_weights = topk_res.values;   // (T, K)
        let topk_indices = topk_res.indices;      // (T, K)

        if self.norm_topk_prob {
            let sum_w = topk_weights.sum_keepdim(D::Minus1)?; // (T, 1)
            topk_weights = topk_weights.broadcast_div(&sum_w)?;
        }

        let k = self.num_experts_per_tok;
        let device = xs.device();

        // Token indices repeated K times: (T, K)
        let token_idx = Tensor::arange(0u32, total_tokens as u32, device)?; // (T,)
        let token_idx_k = token_idx.unsqueeze(1)?.expand((total_tokens, k))?; // (T, K)

        // Flatten everything → (T×K, ...)
        let dispatched_token_ids = token_idx_k.flatten(0, 1)?;                // (T×K,)
        let dispatched_weights   = topk_weights.flatten(0, 1)?.unsqueeze(1)?;  // (T×K, 1)
        let dispatched_experts   = topk_indices.flatten(0, 1)?;                // (T×K,)
        let dispatched_xs        = xs_flat.index_select(&dispatched_token_ids, 0)?; // (T×K, H)

        let mut output = Tensor::zeros((total_tokens, hidden_dim), xs.dtype(), device)?;

        for expert_id in 0..self.experts.len() {
            let expert = &self.experts[expert_id];

            let mask = dispatched_experts.eq(expert_id as u32)?; // (T×K,) bool
            let selected_idx = mask.nonzero()?.squeeze(0)?;     // (N_sel,)

            if selected_idx.elem_count() == 0 {
                continue;
            }

            let batch_xs      = dispatched_xs.i(&selected_idx)?;      // (N_sel, H)
            let batch_weights = dispatched_weights.i(&selected_idx)?; // (N_sel, 1)
            let batch_tokens  = dispatched_token_ids.i(&selected_idx)?; // (N_sel,)

            let expert_out = expert.forward(&batch_xs)?;              // (N_sel, H)
            let expert_out = expert_out.broadcast_mul(&batch_weights)?; // apply weight

            output = output.index_add(&batch_tokens, &expert_out, 0)?;
        }

        output.reshape((b_size, seq_len, hidden_dim))
    }
}

// MLP or MoE decision enum
#[derive(Debug, Clone)]
enum Qwen3FeedForward {
    Mlp(Qwen3MLP),
    MoE(Qwen3SparseMoeBlock),
}

impl Module for Qwen3FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::MoE(m) => m.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    feed_forward: Qwen3FeedForward,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(
        layer_idx: usize,
        cfg: &Config,
        rotary: Arc<Qwen3RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Qwen3Attention::new(&cfg.into(), rotary, vb.pp("self_attn"))?;

        // Decide whether to use MoE or regular MLP based on layer_idx and decoder_sparse_step
        let feed_forward =
            if cfg.num_experts > 0 && (layer_idx + 1).is_multiple_of(cfg.decoder_sparse_step) {
                Qwen3FeedForward::MoE(Qwen3SparseMoeBlock::new(cfg, vb.pp("mlp"))?)
            } else {
                Qwen3FeedForward::Mlp(Qwen3MLP::new(&cfg.into(), vb.pp("mlp"))?)
            };

        let ln1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            feed_forward,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.feed_forward)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(
            vb.dtype(),
            &cfg.into(),
            vb.device(),
        )?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(i, cfg, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}