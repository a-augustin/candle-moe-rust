cargo clean
cargo build --release --features cuda
echo "Running benchmark with original model..." > benchmark.log
./target/release/benchmark --use-original --seq-len 8192 --gen-len 512 >> benchmark.log
echo "" >> benchmark.log
echo "Running benchmark with optimized model..." >> benchmark.log
./target/release/benchmark --seq-len 8192 --gen-len 512 >> benchmark.log