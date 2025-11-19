use std::env;

fn main() {
    // Check if cuda feature is enabled by looking at CARGO_FEATURE_CUDA env var
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    
    if cuda_enabled {
        println!("cargo:warning=Building CUDA kernels to PTX...");
        println!("cargo:rerun-if-changed=src/cuda/moe_gemm.cu");
        
        let cuda_path = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        
        let out_dir = env::var("OUT_DIR").unwrap();
        let cuda_src = "src/cuda/moe_gemm.cu";
        let ptx_out = format!("{}/moe_gemm.ptx", out_dir);
        
        println!("cargo:warning=Compiling CUDA kernel to PTX: {}", cuda_src);
        
        // Compile CUDA kernel to PTX
        let status = std::process::Command::new("nvcc")
            .args(&[
                "--ptx",
                cuda_src,
                "-o", &ptx_out,
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-I", &format!("{}/include", cuda_path),
            ])
            .status()
            .expect("Failed to compile CUDA kernel to PTX. Make sure nvcc is in PATH.");
        
        if !status.success() {
            panic!("CUDA PTX compilation failed");
        }
        
        println!("cargo:warning=PTX generated successfully at: {}", ptx_out);
    } else {
        println!("cargo:warning=CUDA feature not enabled, skipping kernel compilation");
    }
}
