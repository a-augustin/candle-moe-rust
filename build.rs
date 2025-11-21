use std::env;
use std::path::PathBuf;

fn main() {
    // Check if cuda feature is enabled by looking at CARGO_FEATURE_CUDA env var
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    
    if cuda_enabled {
        println!("cargo:warning=Building CUDA support with PTX compilation...");
        
        let cuda_path = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        
        // Link against cuBLAS and CUDA runtime
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
        
        // Compile the CUDA .cu file to PTX using nvcc
        println!("cargo:rerun-if-changed=src/cuda/moe_gemm.cu");
        
        let out_dir = env::var("OUT_DIR").unwrap();
        let cuda_src = "src/cuda/moe_gemm.cu";
        let ptx_out = PathBuf::from(&out_dir).join("moe_gemm.ptx");
        
        println!("cargo:warning=Compiling CUDA kernel to PTX: {}", cuda_src);
        println!("cargo:warning=Output PTX file: {}", ptx_out.display());
        
        // Compile to PTX with optimizations
        // PTX mode only supports single architecture target
        // Use compute_90 for maximum forward compatibility (Hopper+)
        // The PTX will be JIT-compiled to the actual GPU architecture at runtime
        let status = std::process::Command::new("nvcc")
            .args(&[
                "--ptx",                           // Generate PTX instead of binary
                cuda_src,
                "-o", ptx_out.to_str().unwrap(),
                "-O3",                             // Maximum optimization
                "-std=c++17",                      // C++17 for modern features
                "--use_fast_math",                 // Enable fast math operations
                "--expt-relaxed-constexpr",        // Relaxed constexpr rules
                "-I", &format!("{}/include", cuda_path),
                // Single architecture for PTX - use compute_80 for Ampere+ compatibility
                // PTX is forward compatible and will work on newer architectures
                "-arch=compute_80",                // Ampere and newer (A100, RTX 3090+, H100, etc.)
            ])
            .status()
            .expect("Failed to compile CUDA kernel. Make sure nvcc is in PATH.");
        
        if !status.success() {
            panic!("CUDA PTX compilation failed");
        }
        
        // Verify PTX file was created
        if !ptx_out.exists() {
            panic!("PTX file was not created at: {}", ptx_out.display());
        }
        
        println!("cargo:warning=CUDA PTX compilation completed successfully");
        println!("cargo:warning=PTX file size: {} bytes", 
                 std::fs::metadata(&ptx_out).unwrap().len());
    } else {
        println!("cargo:warning=CUDA feature not enabled, skipping kernel compilation");
    }
}
