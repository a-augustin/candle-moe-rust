# candle-moe-rust
Optimizing the Qwen3 MoE model in Rust 

## Build the Project:
```bash
cargo build --features cuda
```

```bash
## To run baseline model:
cargo run --features cuda --bin benchmark -- --use-original --seq-len 8192 --gen-len 512
```

```bash
## To run optimized model:
cargo run --features cuda --bin benchmark -- --seq-len 8192 --gen-len 512
```