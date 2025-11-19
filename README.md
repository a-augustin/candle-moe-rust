# candle-moe-rust
Optimizing the Qwen3 MoE model in Rust 

Using device: Cuda(CudaDevice(DeviceId(1)))
Creating test configuration (no real model weights)...
Test config: experts=4, top_k=2, hidden_size=512
Using OPTIMIZED MoE backend
Model loaded in 85.470417ms
Warming up kernels...

=== PREFILL 8192 tokens ===
Prefill Throughput:  10427.7 tok/s
Prefill Latency:     785.60217ms
Latency / 1K tokens: 95.90 ms

=== DECODE 512 tokens ===
Decode Throughput: 451.1 tok/s
Decode Latency:    1.134940694s
Latency / Token:   2.217 ms/token

=== BENCHMARK SUMMARY ===
Model: Qwen/Qwen3-30B-A3B
Backend: Optimized (fast)
--- Prefill ---
Throughput: 10427.7 tok/s
Total time: 785.60217ms
--- Decode ---
Throughput: 451.1 tok/s
Avg Latency: 2.217 ms/token