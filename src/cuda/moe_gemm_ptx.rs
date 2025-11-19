use candle_core::{CudaDevice, Result, Tensor, Device, DType, IndexOp};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_gemm.ptx"));

/// Grouped GEMM operation for MoE experts using PTX module loading
#[derive(Debug)]
pub struct MoeGroupedGemmPTX {
    device: CudaDevice,
}

impl MoeGroupedGemmPTX {
    pub fn new(device: &CudaDevice, _use_fp16: bool) -> Result<Self> {
        // Pre-load the PTX module during initialization
        device.get_or_load_custom_func("silu_mul_kernel_fp16", "moe_gemm", PTX)?;
        
        Ok(Self {
            device: device.clone(),
        })
    }

    /// Execute grouped GEMM for all experts using Candle's matmul + custom PTX SiLU kernel
    pub fn forward(
        &self,
        inputs: &[Tensor],
        gate_weights: &[Tensor],
        up_weights: &[Tensor],
        down_weights: &[Tensor],
        token_indices: &[Vec<u32>],
        routing_weights: &[Vec<f32>],
        total_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Tensor> {
        let device = &self.device;
        let dtype = inputs[0].dtype();
        
        // Count tokens per expert
        let num_experts = inputs.len();
        let mut tokens_per_expert = vec![0; num_experts];
        for (i, inp) in inputs.iter().enumerate() {
            if inp.dims().len() > 0 && inp.dims()[0] > 0 {
                tokens_per_expert[i] = inp.dims()[0];
            }
        }
        
        // Step 1 & 2: Perform GEMM operations and SiLU activation
        let mut merged_outputs = Vec::new();
        
        for (idx, (input, (gate_w, up_w))) in inputs.iter().zip(gate_weights.iter().zip(up_weights.iter())).enumerate() {
            if tokens_per_expert[idx] == 0 {
                merged_outputs.push(Tensor::zeros(&[0, intermediate_size], dtype, &Device::Cuda(device.clone()))?);
                continue;
            }
            
            // gate_output = input @ gate_weights.t()
            let gate_out = input.matmul(&gate_w.t()?)?;
            
            // up_output = input @ up_weights.t()
            let up_out = input.matmul(&up_w.t()?)?;
            
            // Apply SiLU: merged = silu(gate_out) * up_out using PTX kernel
            let merged = self.apply_silu_kernel(&gate_out, &up_out, tokens_per_expert[idx], intermediate_size)?;
            merged_outputs.push(merged);
        }
        
        // Step 3: Down projection
        let mut final_outputs = Vec::new();
        for (idx, (merged, down_w)) in merged_outputs.iter().zip(down_weights.iter()).enumerate() {
            if tokens_per_expert[idx] == 0 {
                final_outputs.push(Tensor::zeros(&[0, hidden_size], dtype, &Device::Cuda(device.clone()))?);
            } else {
                let out = merged.matmul(&down_w.t()?)?;
                final_outputs.push(out);
            }
        }
        
        // Step 4: Scatter results back with routing weights
        let mut result = Tensor::zeros(&[total_tokens, hidden_size], dtype, &Device::Cuda(device.clone()))?;
        
        for (_expert_idx, (expert_out, (token_idxs, rw))) in final_outputs.iter().zip(token_indices.iter().zip(routing_weights.iter())).enumerate() {
            if token_idxs.is_empty() {
                continue;
            }
            
            // Apply routing weights and accumulate into result
            for (i, &token_idx) in token_idxs.iter().enumerate() {
                let weighted = expert_out.i(i)?.affine(rw[i] as f64, 0.0)?;
                let current = result.i(token_idx as usize)?;
                let updated = (&current + &weighted)?;
                
                // Update result tensor at token_idx - must use 2D range
                result = result.slice_assign(&[token_idx as usize..(token_idx as usize + 1), 0..hidden_size], &updated.unsqueeze(0)?)?;
            }
        }
        
        Ok(result)
    }
    
    fn apply_silu_kernel(
        &self,
        gate: &Tensor,
        up: &Tensor,
        num_tokens: usize,
        intermediate_size: usize,
    ) -> Result<Tensor> {
        let device = &self.device;
        let dtype = gate.dtype();
        
        // Load PTX function based on dtype
        let kernel_name = match dtype {
            DType::F16 => "silu_mul_kernel_fp16",
            DType::BF16 | DType::F32 | DType::F64 => "silu_mul_kernel_fp32",
            _ => candle_core::bail!("Unsupported dtype for SiLU kernel: {:?}", dtype),
        };
        
        let func = device.get_or_load_custom_func(kernel_name, "moe_gemm", PTX)?;
        
        // Convert to F32 if BF16
        let (gate_comp, up_comp) = if dtype == DType::BF16 {
            (gate.to_dtype(DType::F32)?, up.to_dtype(DType::F32)?)
        } else {
            (gate.clone(), up.clone())
        };
        
        // Allocate output tensor in the working dtype
        let working_dtype = if dtype == DType::BF16 { DType::F32 } else { dtype };
        let output = Tensor::zeros(&[num_tokens, intermediate_size], working_dtype, &Device::Cuda(device.clone()))?;
        
        // Get storage and extract slices
        match working_dtype {
            DType::F16 => {
                let (gate_storage, _) = gate_comp.storage_and_layout();
                let (up_storage, _) = up_comp.storage_and_layout();
                let (out_storage, _) = output.storage_and_layout();
                
                let (gate_slice, up_slice, out_slice) = match (&*gate_storage, &*up_storage, &*out_storage) {
                    (candle_core::Storage::Cuda(g_cuda), candle_core::Storage::Cuda(u_cuda), candle_core::Storage::Cuda(o_cuda)) => {
                        (g_cuda.as_cuda_slice::<f16>()?, u_cuda.as_cuda_slice::<f16>()?, o_cuda.as_cuda_slice::<f16>()?)
                    }
                    _ => candle_core::bail!("Expected CUDA storage for all tensors")
                };
                
                let total_elements = num_tokens * intermediate_size;
                let threads = 256;
                let blocks = (total_elements + threads - 1) / threads;
                let cfg = LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                
                let mut builder = func.builder();
                let num_tokens_i32 = num_tokens as i32;
                let intermediate_size_i32 = intermediate_size as i32;
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(out_slice);
                builder.arg(&num_tokens_i32);
                builder.arg(&intermediate_size_i32);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            }
            DType::F32 | DType::F64 | DType::BF16 => {
                let (gate_storage, _) = gate_comp.storage_and_layout();
                let (up_storage, _) = up_comp.storage_and_layout();
                let (out_storage, _) = output.storage_and_layout();
                
                let (gate_slice, up_slice, out_slice) = match (&*gate_storage, &*up_storage, &*out_storage) {
                    (candle_core::Storage::Cuda(g_cuda), candle_core::Storage::Cuda(u_cuda), candle_core::Storage::Cuda(o_cuda)) => {
                        (g_cuda.as_cuda_slice::<f32>()?, u_cuda.as_cuda_slice::<f32>()?, o_cuda.as_cuda_slice::<f32>()?)
                    }
                    _ => candle_core::bail!("Expected CUDA storage for all tensors")
                };
                
                let total_elements = num_tokens * intermediate_size;
                let threads = 256;
                let blocks = (total_elements + threads - 1) / threads;
                let cfg = LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                
                let mut builder = func.builder();
                let num_tokens_i32 = num_tokens as i32;
                let intermediate_size_i32 = intermediate_size as i32;
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(out_slice);
                builder.arg(&num_tokens_i32);
                builder.arg(&intermediate_size_i32);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            }
            _ => candle_core::bail!("Unsupported working dtype"),
        }
        
        // Convert back to BF16 if needed
        if dtype == DType::BF16 {
            output.to_dtype(DType::BF16)
        } else {
            Ok(output)
        }
    }
}
