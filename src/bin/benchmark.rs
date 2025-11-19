use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use hf_hub;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use candle_core::IndexOp;
use tokenizers::Tokenizer;

// Import from parent crate library
use candle_moe_rust::{qwen3_moe, qwen3_moe_optimized};

// Wrapper enum to handle both model types
enum Model {
    Original(qwen3_moe::ModelForCausalLM),
    Optimized(qwen3_moe_optimized::ModelForCausalLM),
}

impl Model {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Model::Original(m) => m.forward(input, offset),
            Model::Optimized(m) => m.forward(input, offset),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Model::Original(m) => m.clear_kv_cache(),
            Model::Optimized(m) => m.clear_kv_cache(),
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "Qwen/Qwen3-30B-A3B")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, default_value_t = 512)]
    seq_len: usize,

    #[arg(long, default_value_t = 64)]
    gen_len: usize,

    #[arg(long)]
    use_original: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_cuda(0)?;
    println!("Using device: {:?}", device);

    // ========== TEMPORARY: Using fake config for testing ==========
    // TODO: Uncomment the HF loading code below when ready to use real model weights
    
    // Create a small test config for benchmarking
    println!("Creating test configuration (no real model weights)...");
    let config_optimized = qwen3_moe_optimized::Config {
        vocab_size: 1000,
        hidden_size: 512,
        intermediate_size: 2048,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        head_dim: 64,
        attention_bias: false,
        num_key_value_heads: 8,
        max_position_embeddings: args.seq_len + args.gen_len,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
        decoder_sparse_step: 1,
        moe_intermediate_size: 2048,
        num_experts_per_tok: 2,
        num_experts: 4,
        norm_topk_prob: false,
    };

    println!(
        "Test config: experts={}, top_k={}, hidden_size={}",
        config_optimized.num_experts, 
        config_optimized.num_experts_per_tok,
        config_optimized.hidden_size
    );

    // Use random initialization instead of loading weights
    let vb = VarBuilder::zeros(DType::BF16, &device);
    
    // ========== COMMENTED OUT: Real HF model loading ==========
    // Uncomment this section to load actual model weights from Hugging Face
    /*
    // HF hub
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        args.model_id.clone(),
        hf_hub::RepoType::Model,
        args.revision.clone(),
    ));
    
    // --- Load safetensor shards ---
    let mut weight_files = vec![];

    // Try various common patterns for safetensors files
    println!("Searching for safetensors files...");
    
    // Pattern 1: model-00001-of-00010.safetensors (try up to 50 shards)
    for total in [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50].iter() {
        let mut found = vec![];
        for i in 1..=*total {
            let filename = format!("model-{:05}-of-{:05}.safetensors", i, total);
            if let Ok(path) = repo.get(&filename) {
                found.push(path);
                println!("  Found: {}", filename);
            }
        }
        if !found.is_empty() && found.len() == *total {
            weight_files = found;
            break;
        }
    }
    
    // Pattern 2: model.safetensors.index.json with actual shard names
    if weight_files.is_empty() {
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            if let Ok(index_content) = std::fs::read_to_string(&index_path) {
                if let Ok(index_json) = serde_json::from_str::<serde_json::Value>(&index_content) {
                    if let Some(weight_map) = index_json.get("weight_map").and_then(|v| v.as_object()) {
                        let mut shard_names: Vec<String> = weight_map.values()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect();
                        shard_names.sort();
                        shard_names.dedup();
                        
                        for shard_name in shard_names {
                            if let Ok(path) = repo.get(&shard_name) {
                                weight_files.push(path);
                                println!("  Found: {}", shard_name);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Pattern 3: Single model.safetensors
    if weight_files.is_empty() {
        if let Ok(path) = repo.get("model.safetensors") {
            weight_files.push(path);
            println!("  Found: model.safetensors");
        }
    }

    if weight_files.is_empty() {
        anyhow::bail!("No safetensors found in HF repo. Model may require authentication or doesn't exist.");
    }

    println!("Found {} safetensor shards.", weight_files.len());

    // --- Load config ---
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    let config_optimized: qwen3_moe_optimized::Config =
        serde_json::from_slice(&std::fs::read(&config_path)?)?;

    println!(
        "Loaded config: experts={}, top_k={}",
        config_optimized.num_experts, config_optimized.num_experts_per_tok
    );

    // --- Load tokenizer ---
    let tokenizer = Tokenizer::from_file(
        tokenizer_path.to_string_lossy().to_string()
    ).map_err(|e| anyhow::anyhow!("Tokenizer load failed: {}", e))?;

    // --- Load weights ---
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&weight_files, DType::BF16, &device)?
    };
    */
    // ========== END COMMENTED SECTION ==========

    let start = Instant::now();
    // --- Build model ---
    let mut model = if args.use_original {
        println!("Using ORIGINAL MoE backend");

        let config_orig = qwen3_moe::Config {
            vocab_size: config_optimized.vocab_size,
            hidden_size: config_optimized.hidden_size,
            intermediate_size: config_optimized.intermediate_size,
            num_hidden_layers: config_optimized.num_hidden_layers,
            num_attention_heads: config_optimized.num_attention_heads,
            head_dim: config_optimized.head_dim,
            attention_bias: config_optimized.attention_bias,
            num_key_value_heads: config_optimized.num_key_value_heads,
            max_position_embeddings: config_optimized.max_position_embeddings,
            sliding_window: config_optimized.sliding_window,
            max_window_layers: config_optimized.max_window_layers,
            tie_word_embeddings: config_optimized.tie_word_embeddings,
            rope_theta: config_optimized.rope_theta,
            rms_norm_eps: config_optimized.rms_norm_eps,
            use_sliding_window: config_optimized.use_sliding_window,
            hidden_act: config_optimized.hidden_act,
            decoder_sparse_step: config_optimized.decoder_sparse_step,
            moe_intermediate_size: config_optimized.moe_intermediate_size,
            num_experts_per_tok: config_optimized.num_experts_per_tok,
            num_experts: config_optimized.num_experts,
            norm_topk_prob: config_optimized.norm_topk_prob,
        };

        Model::Original(qwen3_moe::ModelForCausalLM::new(&config_orig, vb.clone())?)
    } else {
        println!("Using OPTIMIZED MoE backend");
        Model::Optimized(qwen3_moe_optimized::ModelForCausalLM::new(&config_optimized, vb.clone())?)
    };

    device.synchronize()?;
    println!("Model loaded in {:?}", start.elapsed());

    // --- Warmup ---
    println!("Warming up kernels...");
    for _ in 0..3 {
        let dummy_tokens = vec![0u32; 16];
        let dummy = Tensor::new(&dummy_tokens[..], &device)?.unsqueeze(0)?;
        let _ = model.forward(&dummy, 0)?;
        model.clear_kv_cache();
    }

    // --- PREFILL ---
    println!("\n=== PREFILL {} tokens ===", args.seq_len);

    // Generate fake tokens for testing (just repeat 0..vocab_size)
    let base: Vec<u32> = (0..config_optimized.vocab_size as u32).cycle()
        .take(args.seq_len)
        .collect();

    let input = Tensor::new(&base[..], &device)?.unsqueeze(0)?;

    let start = Instant::now();
    let _ = model.forward(&input, 0)?;
    device.synchronize()?;
    let prefill_t = start.elapsed();

    let prefill_tok_s = args.seq_len as f64 / prefill_t.as_secs_f64();
    let prefill_latency_per_1k =
        (prefill_t.as_secs_f64() * 1000.0) / (args.seq_len as f64 / 1000.0);

    println!(
        "Prefill Throughput:  {:.1} tok/s\nPrefill Latency:     {:?}\nLatency / 1K tokens: {:.2} ms",
        prefill_tok_s,
        prefill_t,
        prefill_latency_per_1k
    );

    // --- DECODE ---
    println!("\n=== DECODE {} tokens ===", args.gen_len);

    model.clear_kv_cache();
    let mut offset = args.seq_len;

    let mut last_token = input.narrow(1, offset - 1, 1)?;
    let mut logits_processor = LogitsProcessor::new(1234, Some(1.0), Some(1.0));

    let pb = ProgressBar::new(args.gen_len as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar} {pos}/{len} tok")
            .unwrap(),
    );

    let start = Instant::now();
    for _ in 0..args.gen_len {
        let logits = model.forward(&last_token, offset)?;
        let next = logits_processor.sample(&logits.squeeze(0)?)?;
        last_token = Tensor::new(&[next], &device)?.unsqueeze(0)?;
        offset += 1;
        pb.inc(1);
    }
    device.synchronize()?;
    let decode_t = start.elapsed();
    pb.finish();

    let decode_tok_s = args.gen_len as f64 / decode_t.as_secs_f64();
    let decode_lat_ms = decode_t.as_secs_f64() * 1000.0 / args.gen_len as f64;

    println!(
        "Decode Throughput: {:.1} tok/s\nDecode Latency:    {:?}\nLatency / Token:   {:.3} ms/token",
        decode_tok_s,
        decode_t,
        decode_lat_ms
    );

    // --- SUMMARY ---
    println!("\n=== BENCHMARK SUMMARY ===");
    println!("Model: {}", args.model_id);
    println!(
        "Backend: {}",
        if args.use_original {
            "Original (slow)"
        } else {
            "Optimized (fast)"
        }
    );

    println!("--- Prefill ---");
    println!("Throughput: {:.1} tok/s", prefill_tok_s);
    println!("Total time: {:?}", prefill_t);

    println!("--- Decode ---");
    println!("Throughput: {:.1} tok/s", decode_tok_s);
    println!("Avg Latency: {:.3} ms/token", decode_lat_ms);

    Ok(())
}
