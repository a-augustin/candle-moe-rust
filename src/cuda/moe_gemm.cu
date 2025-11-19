#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

extern "C" {

// Structure to hold grouped GEMM parameters for MoE
typedef struct {
    int num_experts;
    int num_tokens_per_expert;
    int hidden_size;
    int intermediate_size;
    const void** expert_weights;  // Array of expert weight pointers
    const void** input_tokens;     // Array of input token pointers per expert
    void** outputs;                // Array of output pointers per expert
    const float* routing_weights;  // Routing weights for each token-expert pair
    cudaDataType_t data_type;
} MoeGemmParams;

/**
 * Grouped GEMM kernel for MoE expert computation
 * Uses cuBLAS batched GEMM operations to efficiently compute multiple experts in parallel
 * 
 * Each expert processes its assigned tokens:
 * output[expert_i] = gate(input[expert_i] @ gate_proj[expert_i]) * (input[expert_i] @ up_proj[expert_i])
 * final_output[expert_i] = output[expert_i] @ down_proj[expert_i]
 */
cudaError_t moe_grouped_gemm_gate_up(
    cublasHandle_t handle,
    int num_groups,
    const int* tokens_per_group,      // Number of tokens for each expert
    const void** input_ptrs,          // Input tensors [tokens, hidden_size] for each expert
    const void** gate_weight_ptrs,    // Gate projection weights [intermediate, hidden] for each expert
    const void** up_weight_ptrs,      // Up projection weights [intermediate, hidden] for each expert
    void** gate_outputs,              // Gate outputs [tokens, intermediate] for each expert
    void** up_outputs,                // Up outputs [tokens, intermediate] for each expert
    int hidden_size,
    int intermediate_size,
    cudaDataType_t compute_type,
    bool use_fp16
) {
    cublasStatus_t status;
    
    // Create cuBLAS handle if not provided
    cublasHandle_t local_handle = handle;
    bool created_handle = false;
    if (local_handle == nullptr) {
        status = cublasCreate(&local_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Failed to create cuBLAS handle: %d\n", status);
            return cudaErrorInitializationError;
        }
        created_handle = true;
    }
    
    printf("CUDA moe_grouped_gemm_gate_up called:\n");
    printf("  num_groups=%d, hidden_size=%d, intermediate_size=%d, use_fp16=%d\n", 
           num_groups, hidden_size, intermediate_size, use_fp16);
    
    // cuBLAS uses column-major, so we need to transpose our operations
    // For C = A @ B^T (row-major), we compute C^T = B @ A^T (col-major)
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasOperation_t no_trans = CUBLAS_OP_N;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_fp16 = __float2half(1.0f);
    half beta_fp16 = __float2half(0.0f);
    
    // Process each expert group
    for (int i = 0; i < num_groups; i++) {
        int m = tokens_per_group[i];
        if (m == 0) continue;
        
        int k = hidden_size;
        int n = intermediate_size;
        
        if (use_fp16) {
            // Gate projection: gate_output = input @ gate_weights^T
            // Shape: [m, k] @ [n, k]^T = [m, n]
            status = cublasGemmEx(
                local_handle,
                CUBLAS_OP_N,  // gate_weights: no transpose (stored transposed)
                CUBLAS_OP_N,  // input: no transpose
                n, m, k,      // n, m, k in column-major
                &alpha_fp16,
                gate_weight_ptrs[i], CUDA_R_16F, n,
                input_ptrs[i], CUDA_R_16F, k,
                &beta_fp16,
                gate_outputs[i], CUDA_R_16F, n,
                compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Gate GEMM failed for expert %d: %d\n", i, status);
                if (created_handle) cublasDestroy(local_handle);
                return cudaErrorUnknown;
            }
            
            // Up projection: up_output = input @ up_weights^T
            status = cublasGemmEx(
                local_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, m, k,
                &alpha_fp16,
                up_weight_ptrs[i], CUDA_R_16F, n,
                input_ptrs[i], CUDA_R_16F, k,
                &beta_fp16,
                up_outputs[i], CUDA_R_16F, n,
                compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Up GEMM failed for expert %d: %d\n", i, status);
                if (created_handle) cublasDestroy(local_handle);
                return cudaErrorUnknown;
            }
        } else {
            // FP32 version
            status = cublasSgemm(
                local_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, m, k,
                &alpha,
                (const float*)gate_weight_ptrs[i], n,
                (const float*)input_ptrs[i], k,
                &beta,
                (float*)gate_outputs[i], n
            );
            if (status != CUBLAS_STATUS_SUCCESS) {
                if (created_handle) cublasDestroy(local_handle);
                return cudaErrorUnknown;
            }
            
            status = cublasSgemm(
                local_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, m, k,
                &alpha,
                (const float*)up_weight_ptrs[i], n,
                (const float*)input_ptrs[i], k,
                &beta,
                (float*)up_outputs[i], n
            );
            if (status != CUBLAS_STATUS_SUCCESS) {
                if (created_handle) cublasDestroy(local_handle);
                return cudaErrorUnknown;
            }
        }
    }
    
    if (created_handle) {
        cublasDestroy(local_handle);
    }
    
    return cudaSuccess;
}

/**
 * SiLU activation and element-wise multiplication kernel
 * output = silu(gate) * up
 */
// SiLU + Mul kernel with column-major input layout from cuBLAS GEMM
// gate and up are column-major: [intermediate_size, num_tokens]
// We need to read them correctly and write row-major output: [num_tokens, intermediate_size]
__global__ void silu_mul_kernel_fp16(
    const half* gate,          // Column-major: [intermediate_size, num_tokens]
    const half* up,            // Column-major: [intermediate_size, num_tokens]
    half* output,              // Row-major: [num_tokens, intermediate_size]
    int num_tokens,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * intermediate_size;
    
    if (idx < total_elements) {
        // Convert linear index to row-major coordinates
        int token_idx = idx / intermediate_size;
        int feat_idx = idx % intermediate_size;
        
        // Read from column-major layout: gate[feat, token] = gate[feat * num_tokens + token]
        int col_major_idx = feat_idx * num_tokens + token_idx;
        
        float g = __half2float(gate[col_major_idx]);
        float u = __half2float(up[col_major_idx]);
        float silu = g / (1.0f + expf(-g));
        
        // Write to row-major layout: output[token, feat] = output[token * intermediate_size + feat]
        output[idx] = __float2half(silu * u);
    }
}

__global__ void silu_mul_kernel_fp32(
    const float* gate,         // Column-major: [intermediate_size, num_tokens]
    const float* up,           // Column-major: [intermediate_size, num_tokens]
    float* output,             // Row-major: [num_tokens, intermediate_size]
    int num_tokens,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * intermediate_size;
    
    if (idx < total_elements) {
        // Convert linear index to row-major coordinates
        int token_idx = idx / intermediate_size;
        int feat_idx = idx % intermediate_size;
        
        // Read from column-major layout
        int col_major_idx = feat_idx * num_tokens + token_idx;
        
        float g = gate[col_major_idx];
        float u = up[col_major_idx];
        float silu = g / (1.0f + expf(-g));
        
        // Write to row-major layout
        output[idx] = silu * u;
    }
}

cudaError_t moe_silu_mul(
    void** gate_outputs,
    void** up_outputs,
    void** merged_outputs,
    const int* tokens_per_group,
    int num_groups,
    int intermediate_size,
    bool use_fp16
) {
    if (!gate_outputs || !up_outputs || !merged_outputs || !tokens_per_group) {
        return cudaErrorInvalidValue;
    }
    
    // Debug: print pointers received
    printf("CUDA kernel moe_silu_mul called:\n");
    printf("  num_groups=%d, intermediate_size=%d, use_fp16=%d\n", num_groups, intermediate_size, use_fp16);
    for (int i = 0; i < num_groups; i++) {
        printf("  Group %d: tokens_per_group=%d, gate=0x%llx, up=0x%llx, merged=0x%llx\n", 
               i, tokens_per_group[i], 
               (unsigned long long)gate_outputs[i],
               (unsigned long long)up_outputs[i],
               (unsigned long long)merged_outputs[i]);
    }
    
    int threads = 256;
    
    for (int i = 0; i < num_groups; i++) {
        int num_tokens = tokens_per_group[i];
        if (num_tokens == 0) continue;
        
        // Safety check - ensure pointers are not null
        if (!gate_outputs[i] || !up_outputs[i] || !merged_outputs[i]) {
            printf("Warning: Skipping expert %d due to null pointer\n", i);
            continue;
        }
        
        int total_elements = num_tokens * intermediate_size;
        int blocks = (total_elements + threads - 1) / threads;
        printf("Launching kernel for group %d: blocks=%d, threads=%d, tokens=%d, features=%d\n", 
               i, blocks, threads, num_tokens, intermediate_size);
        
        if (use_fp16) {
            silu_mul_kernel_fp16<<<blocks, threads>>>(
                (const half*)gate_outputs[i],
                (const half*)up_outputs[i],
                (half*)merged_outputs[i],
                num_tokens,
                intermediate_size
            );
        } else {
            silu_mul_kernel_fp32<<<blocks, threads>>>(
                (const float*)gate_outputs[i],
                (const float*)up_outputs[i],
                (float*)merged_outputs[i],
                num_tokens,
                intermediate_size
            );
        }
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error for group %d: %s\n", i, cudaGetErrorString(err));
            return err;
        }
        
        // Synchronize to catch runtime errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution error for group %d: %s\n", i, cudaGetErrorString(err));
            return err;
        }
    }
    
    return cudaSuccess;
}

/**
 * Down projection: final_output = merged @ down_weights^T
 * Input: merged is ROW-MAJOR [tokens, intermediate] from SiLU kernel
 * Weights: down_weights is stored as [hidden, intermediate] (transposed)
 * Output: ROW-MAJOR [tokens, hidden]
 */
cudaError_t moe_grouped_gemm_down(
    cublasHandle_t handle,
    int num_groups,
    const int* tokens_per_group,
    const void** merged_ptrs,         // ROW-MAJOR: [tokens, intermediate]
    const void** down_weight_ptrs,    // Weight matrix: [hidden, intermediate]
    void** outputs,                   // ROW-MAJOR: [tokens, hidden]
    int hidden_size,
    int intermediate_size,
    cudaDataType_t compute_type,
    bool use_fp16
) {
    cublasStatus_t status;
    
    // Create cuBLAS handle if not provided
    cublasHandle_t local_handle = handle;
    bool created_handle = false;
    if (local_handle == nullptr) {
        status = cublasCreate(&local_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return cudaErrorInitializationError;
        }
        created_handle = true;
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_fp16 = __float2half(1.0f);
    half beta_fp16 = __float2half(0.0f);
    
    for (int i = 0; i < num_groups; i++) {
        int m = tokens_per_group[i];
        if (m == 0) continue;
        
        int k = intermediate_size;
        int n = hidden_size;
        
        // Row-major: C[m,n] = A[m,k] @ B[k,n]^T = A[m,k] @ B[n,k]
        // In cuBLAS column-major: C^T[n,m] = B[n,k] @ A^T[k,m]
        // Since A is row-major [m,k], A^T in col-major has lda=k
        // Since B is stored as [n,k], it's already transposed for us
        
        if (use_fp16) {
            status = cublasGemmEx(
                local_handle,
                CUBLAS_OP_N,      // down_weights[n,k]: no transpose
                CUBLAS_OP_T,      // merged[m,k] row-major -> transpose for col-major
                n, m, k,
                &alpha_fp16,
                down_weight_ptrs[i], CUDA_R_16F, n,  // lda = n (leading dim of col-major view)
                merged_ptrs[i], CUDA_R_16F, k,       // ldb = k (leading dim of row-major as transposed)
                &beta_fp16,
                outputs[i], CUDA_R_16F, n,           // ldc = n (output is col-major)
                compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Down GEMM failed for expert %d: %d\n", i, status);
                if (created_handle) cublasDestroy(local_handle);
                return cudaErrorUnknown;
            }
        } else {
            // FP32 version: same layout handling
            status = cublasSgemm(
                local_handle,
                CUBLAS_OP_N,      // down_weights[n,k]: no transpose
                CUBLAS_OP_T,      // merged[m,k] row-major -> transpose
                n, m, k,
                &alpha,
                (const float*)down_weight_ptrs[i], n,
                (const float*)merged_ptrs[i], k,
                &beta,
                (float*)outputs[i], n
            );
            if (status != CUBLAS_STATUS_SUCCESS) {
                if (created_handle) cublasDestroy(local_handle);
                return cudaErrorUnknown;
            }
        }
    }
    
    if (created_handle) {
        cublasDestroy(local_handle);
    }
    
    return cudaSuccess;
}

/**
 * Apply routing weights and scatter back to original positions
 * expert_output is COLUMN-MAJOR [hidden, num_tokens] from cuBLAS down projection
 * final_output is ROW-MAJOR [total_tokens, hidden]
 * output[token_indices[i]] += expert_output[i] * routing_weights[i]
 */
__global__ void scatter_weighted_kernel_fp16(
    const half* expert_output,      // Column-major: [hidden, num_tokens]
    half* final_output,             // Row-major: [total_tokens, hidden]
    const int* token_indices,
    const float* routing_weights,
    int num_tokens,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * hidden_size;
    
    if (idx < total_elements) {
        // Output is row-major indexed
        int token_idx = idx / hidden_size;
        int hidden_idx = idx % hidden_size;
        
        // Read from column-major: expert_output[hidden, token] = expert_output[hidden * num_tokens + token]
        int col_major_idx = hidden_idx * num_tokens + token_idx;
        
        // Write to row-major: final_output[token_indices[token_idx], hidden_idx]
        int out_idx = token_indices[token_idx] * hidden_size + hidden_idx;
        
        float weight = routing_weights[token_idx];
        float value = __half2float(expert_output[col_major_idx]);
        atomicAdd((float*)&final_output[out_idx], weight * value);
    }
}

__global__ void scatter_weighted_kernel_fp32(
    const float* expert_output,     // Column-major: [hidden, num_tokens]
    float* final_output,            // Row-major: [total_tokens, hidden]
    const int* token_indices,
    const float* routing_weights,
    int num_tokens,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * hidden_size;
    
    if (idx < total_elements) {
        int token_idx = idx / hidden_size;
        int hidden_idx = idx % hidden_size;
        
        // Read from column-major
        int col_major_idx = hidden_idx * num_tokens + token_idx;
        
        // Write to row-major
        int out_idx = token_indices[token_idx] * hidden_size + hidden_idx;
        
        float weight = routing_weights[token_idx];
        float value = expert_output[col_major_idx];
        atomicAdd(&final_output[out_idx], weight * value);
    }
}

cudaError_t moe_scatter_weighted(
    void** expert_outputs,
    void* final_output,
    const int** token_indices,
    const float** routing_weights,
    const int* tokens_per_group,
    int num_groups,
    int hidden_size,
    bool use_fp16
) {
    int threads = 256;
    
    for (int i = 0; i < num_groups; i++) {
        int num_tokens = tokens_per_group[i];
        if (num_tokens == 0) continue;
        
        int size = num_tokens * hidden_size;
        int blocks = (size + threads - 1) / threads;
        
        if (use_fp16) {
            scatter_weighted_kernel_fp16<<<blocks, threads>>>(
                (const half*)expert_outputs[i],
                (half*)final_output,
                token_indices[i],
                routing_weights[i],
                num_tokens,
                hidden_size
            );
        } else {
            scatter_weighted_kernel_fp32<<<blocks, threads>>>(
                (const float*)expert_outputs[i],
                (float*)final_output,
                token_indices[i],
                routing_weights[i],
                num_tokens,
                hidden_size
            );
        }
    }
    
    return cudaGetLastError();
}

} // extern "C"
