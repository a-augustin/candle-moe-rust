use candle_core::{CudaDevice, Result, Tensor, DType, Device};
use std::ffi::c_void;
use cudarc::driver::{CudaDevice as CuDevice, DeviceSlice};
use std::sync::Arc;

#[link(name = "moe_kernels", kind = "static")]
unsafe extern "C" {
    fn moe_grouped_gemm_gate_up(
        handle: *mut c_void,
        num_groups: i32,
        tokens_per_group: *const i32,
        input_ptrs: *const *const c_void,
        gate_weight_ptrs: *const *const c_void,
        up_weight_ptrs: *const *const c_void,
        gate_outputs: *mut *mut c_void,
        up_outputs: *mut *mut c_void,
        hidden_size: i32,
        intermediate_size: i32,
        compute_type: i32,
        use_fp16: bool,
    ) -> i32;

    fn moe_silu_mul(
        gate_outputs: *mut *mut c_void,
        up_outputs: *mut *mut c_void,
        merged_outputs: *mut *mut c_void,
        tokens_per_group: *const i32,
        num_groups: i32,
        intermediate_size: i32,
        use_fp16: bool,
    ) -> i32;

    fn moe_grouped_gemm_down(
        handle: *mut c_void,
        num_groups: i32,
        tokens_per_group: *const i32,
        merged_ptrs: *const *const c_void,
        down_weight_ptrs: *const *const c_void,
        outputs: *mut *mut c_void,
        hidden_size: i32,
        intermediate_size: i32,
        compute_type: i32,
        use_fp16: bool,
    ) -> i32;

    fn moe_scatter_weighted(
        expert_outputs: *mut *mut c_void,
        final_output: *mut c_void,
        token_indices: *const *const i32,
        routing_weights: *const *const f32,
        tokens_per_group: *const i32,
        num_groups: i32,
        hidden_size: i32,
        use_fp16: bool,
    ) -> i32;
}

/// Grouped GEMM operation for MoE experts
/// Efficiently processes multiple experts in parallel using cuBLAS batched operations
#[derive(Debug)]
pub struct MoeGroupedGemm {
    device: CudaDevice,
    use_fp16: bool,
}

impl MoeGroupedGemm {
    pub fn new(device: &CudaDevice, use_fp16: bool) -> Self {
        Self {
            device: device.clone(),
            use_fp16,
        }
    }

    /// Execute grouped GEMM for all experts
    /// 
    /// # Arguments
    /// * `inputs` - Input tensors for each expert [num_experts][tokens_per_expert, hidden_size]
    /// * `gate_weights` - Gate projection weights [num_experts][intermediate_size, hidden_size]
    /// * `up_weights` - Up projection weights [num_experts][intermediate_size, hidden_size]
    /// * `down_weights` - Down projection weights [num_experts][hidden_size, intermediate_size]
    /// * `token_indices` - Original token indices for each expert's tokens
    /// * `routing_weights` - Routing weights for each token-expert pair
    /// * `total_tokens` - Total number of tokens in the batch
    /// 
    /// # Returns
    /// Final output tensor [total_tokens, hidden_size] with expert outputs scattered back
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
        let num_experts = inputs.len();
        assert_eq!(gate_weights.len(), num_experts);
        assert_eq!(up_weights.len(), num_experts);
        assert_eq!(down_weights.len(), num_experts);
        assert_eq!(token_indices.len(), num_experts);
        assert_eq!(routing_weights.len(), num_experts);

        // Get cuBLAS handle - CudaBlas is an Arc, we need to get the raw handle
        // Candle's CudaBlas wraps cuBLAS, but doesn't expose the handle directly
        // For now, use a null handle and let cuBLAS create one internally
        let cublas_handle = std::ptr::null_mut::<c_void>();
        // Prepare pointer arrays
        let mut tokens_per_expert = Vec::with_capacity(num_experts);
        let mut input_ptrs = Vec::with_capacity(num_experts);
        let mut gate_weight_ptrs = Vec::with_capacity(num_experts);
        let mut up_weight_ptrs = Vec::with_capacity(num_experts);
        let mut down_weight_ptrs = Vec::with_capacity(num_experts);

        for i in 0..num_experts {
            let num_tokens = if inputs[i].dims().len() == 2 {
                inputs[i].dims()[0]
            } else {
                0
            };
            tokens_per_expert.push(num_tokens as i32);

            // Get raw device pointers using candle's storage API
            if num_tokens > 0 {
                input_ptrs.push(get_cuda_ptr(&inputs[i])? as *const c_void);
                gate_weight_ptrs.push(get_cuda_ptr(&gate_weights[i])? as *const c_void);
                up_weight_ptrs.push(get_cuda_ptr(&up_weights[i])? as *const c_void);
                down_weight_ptrs.push(get_cuda_ptr(&down_weights[i])? as *const c_void);
            } else {
                input_ptrs.push(std::ptr::null());
                gate_weight_ptrs.push(std::ptr::null());
                up_weight_ptrs.push(std::ptr::null());
                down_weight_ptrs.push(std::ptr::null());
            }
        }

        // Allocate intermediate buffers - store tensors to keep them alive
        let mut gate_output_tensors = Vec::new();
        let mut up_output_tensors = Vec::new();
        let mut merged_output_tensors = Vec::new();
        let mut final_expert_output_tensors = Vec::new();
        
        let mut gate_outputs = Vec::new();
        let mut up_outputs = Vec::new();
        let mut merged_outputs = Vec::new();
        let mut final_expert_outputs = Vec::new();

        for &num_tokens in &tokens_per_expert {
            if num_tokens == 0 {
                gate_outputs.push(std::ptr::null_mut());
                up_outputs.push(std::ptr::null_mut());
                merged_outputs.push(std::ptr::null_mut());
                final_expert_outputs.push(std::ptr::null_mut());
                continue;
            }

            // Create TRULY separate allocations by using add(0) which forces a copy
            // This ensures each tensor gets its own device memory, not shared storage
            let gate_out = Tensor::zeros(
                (num_tokens as usize, intermediate_size),
                inputs[0].dtype(),
                inputs[0].device(),
            )?.broadcast_add(&Tensor::zeros(&[], inputs[0].dtype(), inputs[0].device())?)?;
            
            let up_out = Tensor::zeros(
                (num_tokens as usize, intermediate_size),
                inputs[0].dtype(),
                inputs[0].device(),
            )?.broadcast_add(&Tensor::zeros(&[], inputs[0].dtype(), inputs[0].device())?)?;
            
            let merged_out = Tensor::zeros(
                (num_tokens as usize, intermediate_size),
                inputs[0].dtype(),
                inputs[0].device(),
            )?.broadcast_add(&Tensor::zeros(&[], inputs[0].dtype(), inputs[0].device())?)?;
            
            let final_out = Tensor::zeros(
                (num_tokens as usize, hidden_size),
                inputs[0].dtype(),
                inputs[0].device(),
            )?.broadcast_add(&Tensor::zeros(&[], inputs[0].dtype(), inputs[0].device())?)?;

            gate_outputs.push(get_cuda_ptr_mut(&gate_out)?);
            up_outputs.push(get_cuda_ptr_mut(&up_out)?);
            merged_outputs.push(get_cuda_ptr_mut(&merged_out)?);
            final_expert_outputs.push(get_cuda_ptr_mut(&final_out)?);
            
            gate_output_tensors.push(gate_out);
            up_output_tensors.push(up_out);
            merged_output_tensors.push(merged_out);
            final_expert_output_tensors.push(final_out);
        }

        // Step 1: Gate and Up projections
        unsafe {
            let ret = moe_grouped_gemm_gate_up(
                cublas_handle,
                num_experts as i32,
                tokens_per_expert.as_ptr(),
                input_ptrs.as_ptr(),
                gate_weight_ptrs.as_ptr(),
                up_weight_ptrs.as_ptr(),
                gate_outputs.as_mut_ptr(),
                up_outputs.as_mut_ptr(),
                hidden_size as i32,
                intermediate_size as i32,
                self.compute_type(),
                self.use_fp16,
            );
            if ret != 0 {
                candle_core::bail!("CUDA moe_grouped_gemm_gate_up failed with code {}", ret);
            }
        }

        // Step 2: SiLU activation and element-wise multiplication
        unsafe {
            let ret = moe_silu_mul(
                gate_outputs.as_mut_ptr(),
                up_outputs.as_mut_ptr(),
                merged_outputs.as_mut_ptr(),
                tokens_per_expert.as_ptr(),
                num_experts as i32,
                intermediate_size as i32,
                self.use_fp16,
            );
            if ret != 0 {
                candle_core::bail!("CUDA moe_silu_mul failed with code {}", ret);
            }
        }

        // Step 3: Down projection
        unsafe {
            let ret = moe_grouped_gemm_down(
                cublas_handle,
                num_experts as i32,
                tokens_per_expert.as_ptr(),
                merged_outputs.as_ptr() as *const *const c_void,
                down_weight_ptrs.as_ptr(),
                final_expert_outputs.as_mut_ptr(),
                hidden_size as i32,
                intermediate_size as i32,
                self.compute_type(),
                self.use_fp16,
            );
            if ret != 0 {
                candle_core::bail!("CUDA moe_grouped_gemm_down failed with code {}", ret);
            }
        }

        // Step 4: Scatter weighted outputs back to original positions
        let final_output = Tensor::zeros(
            (total_tokens, hidden_size),
            inputs[0].dtype(),
            inputs[0].device(),
        )?;

        // Prepare token indices and routing weights for CUDA
        let mut token_idx_ptrs = Vec::new();
        let mut routing_weight_ptrs = Vec::new();
        let mut token_idx_device_tensors = Vec::new();
        let mut routing_weight_device_tensors = Vec::new();

        for i in 0..num_experts {
            if tokens_per_expert[i] == 0 {
                token_idx_ptrs.push(std::ptr::null());
                routing_weight_ptrs.push(std::ptr::null());
                continue;
            }

            // Copy to device
            let idx_u32: Vec<u32> = token_indices[i].iter().copied().collect();
            let idx_tensor = Tensor::from_slice(
                &idx_u32,
                token_indices[i].len(),
                inputs[0].device(),
            )?;
            let weight_tensor = Tensor::from_slice(
                &routing_weights[i],
                routing_weights[i].len(),
                inputs[0].device(),
            )?;

            token_idx_ptrs.push(get_cuda_ptr(&idx_tensor)? as *const i32);
            routing_weight_ptrs.push(get_cuda_ptr(&weight_tensor)? as *const f32);
            
            token_idx_device_tensors.push(idx_tensor);
            routing_weight_device_tensors.push(weight_tensor);
        }

        unsafe {
            let ret = moe_scatter_weighted(
                final_expert_outputs.as_mut_ptr(),
                get_cuda_ptr_mut(&final_output)?,
                token_idx_ptrs.as_ptr(),
                routing_weight_ptrs.as_ptr(),
                tokens_per_expert.as_ptr(),
                num_experts as i32,
                hidden_size as i32,
                self.use_fp16,
            );
            if ret != 0 {
                candle_core::bail!("CUDA moe_scatter_weighted failed with code {}", ret);
            }
        }

        Ok(final_output)
    }

    fn compute_type(&self) -> i32 {
        if self.use_fp16 {
            2 // CUDA_R_16F
        } else {
            0 // CUDA_R_32F
        }
    }
}

// Helper to allocate separate CUDA memory and copy tensor data
// This bypasses Candle's shared storage issue
fn tensor_to_device_copy<T: cudarc::driver::DeviceRepr>(
    tensor: &Tensor,
    device: &Arc<CuDevice>,
) -> Result<cudarc::driver::CudaSlice<T>> {
    // Copy tensor data to host first
    let data = tensor.to_vec1::<T>()?;
    // Allocate fresh device memory and copy
    let dev_slice = device.htod_sync_copy(&data)
        .map_err(|e| candle_core::Error::Msg(format!("CUDA copy failed: {:?}", e)))?;
    Ok(dev_slice)
}

// Helper functions to extract CUDA device pointers from Candle tensors
// CRITICAL FIX: as_cuda_slice() returns &CudaSlice<T>, not CudaSlice<T>
// So slice is already a reference - we don't need &slice
fn get_cuda_ptr(tensor: &Tensor) -> Result<*const u8> {
    let (storage, layout) = tensor.storage_and_layout();
    let start_offset = layout.start_offset();
    
    match &*storage {
        candle_core::Storage::Cuda(cuda_storage) => {
            let elem_size = tensor.dtype().size_in_bytes();
            let offset_bytes = start_offset * elem_size;
            
            eprintln!("DEBUG get_cuda_ptr: start_offset={}, elem_size={}, offset_bytes={}", 
                     start_offset, elem_size, offset_bytes);
            
            // Extract device pointer by reading first field of CudaSlice struct
            // CudaSlice<T> = { cu_device_ptr: u64, len: usize, ... }
            // as_cuda_slice() returns &CudaSlice<T>, so slice IS the reference
            let device_ptr: u64 = match tensor.dtype() {
                DType::F32 => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    // slice is &CudaSlice, cast to *const u64 and read first field
                    unsafe {
                        let ptr = slice as *const _ as *const u64;
                        let dev_ptr = std::ptr::read_unaligned(ptr);
                        eprintln!("DEBUG get_cuda_ptr: slice addr={:p}, device_ptr={:#x}", slice, dev_ptr);
                        dev_ptr
                    }
                },
                DType::F16 => {
                    let slice = cuda_storage.as_cuda_slice::<half::f16>()?;
                    unsafe {
                        let ptr = slice as *const _ as *const u64;
                        let dev_ptr = std::ptr::read_unaligned(ptr);
                        eprintln!("DEBUG get_cuda_ptr: slice addr={:p}, device_ptr={:#x}", slice, dev_ptr);
                        dev_ptr
                    }
                },
                DType::BF16 => {
                    let slice = cuda_storage.as_cuda_slice::<half::bf16>()?;
                    unsafe {
                        let ptr = slice as *const _ as *const u64;
                        let dev_ptr = std::ptr::read_unaligned(ptr);
                        eprintln!("DEBUG get_cuda_ptr: slice addr={:p}, device_ptr={:#x}", slice, dev_ptr);
                        dev_ptr
                    }
                },
                DType::U8 => {
                    let slice = cuda_storage.as_cuda_slice::<u8>()?;
                    unsafe {
                        let ptr = slice as *const _ as *const u64;
                        std::ptr::read_unaligned(ptr)
                    }
                },
                DType::U32 => {
                    let slice = cuda_storage.as_cuda_slice::<u32>()?;
                    unsafe {
                        let ptr = slice as *const _ as *const u64;
                        std::ptr::read_unaligned(ptr)
                    }
                },
                DType::I64 => {
                    let slice = cuda_storage.as_cuda_slice::<i64>()?;
                    unsafe {
                        let ptr = slice as *const _ as *const u64;
                        std::ptr::read_unaligned(ptr)
                    }
                },
                _ => candle_core::bail!("Unsupported dtype: {:?}", tensor.dtype()),
            };
            
            Ok((device_ptr as usize + offset_bytes) as *const u8)
        }
        _ => candle_core::bail!("Expected CUDA storage, got CPU storage"),
    }
}

fn get_cuda_ptr_mut(tensor: &Tensor) -> Result<*mut c_void> {
    Ok(get_cuda_ptr(tensor)? as *mut c_void)
}
