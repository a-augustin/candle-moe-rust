#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s:%d - status %d\n", __FILE__, __LINE__, status); \
            return 1; \
        } \
    } while(0)

#define MAX_EXPERTS 64
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define TILE_M 128
#define TILE_N 128
#define TILE_K 16

__device__ __forceinline__ float silu_activation(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ half silu_activation(half x) {
    float fx = __half2float(x);
    return __float2half(fx / (1.0f + expf(-fx)));
}

template<typename T>
__global__ void fused_silu_mul_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ output,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        T g = gate[i];
        T u = up[i];
        output[i] = silu_activation(g) * u;
    }
}

__global__ void fused_silu_mul_kernel_vec4(
    const float4* __restrict__ gate,
    const float4* __restrict__ up,
    float4* __restrict__ output,
    int vec4_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < vec4_elements; i += blockDim.x * gridDim.x) {
        float4 g = gate[i];
        float4 u = up[i];
        float4 o;
        
        o.x = silu_activation(g.x) * u.x;
        o.y = silu_activation(g.y) * u.y;
        o.z = silu_activation(g.z) * u.z;
        o.w = silu_activation(g.w) * u.w;
        
        output[i] = o;
    }
}

template<typename T>
__global__ void weighted_scatter_reduce_kernel(
    const T* __restrict__ expert_output,
    const int* __restrict__ token_indices,
    const float* __restrict__ routing_weights,
    T* __restrict__ final_output,
    int num_tokens_expert,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens_expert * hidden_size;
    
    if (idx < total_elements) {
        int local_token_idx = idx / hidden_size;
        int feature_idx = idx % hidden_size;
        
        int global_token_idx = token_indices[local_token_idx];
        float weight = routing_weights[local_token_idx];
        int output_idx = global_token_idx * hidden_size + feature_idx;
        
        if constexpr (sizeof(T) == 4) {
            atomicAdd(&final_output[output_idx], static_cast<T>(expert_output[idx] * weight));
        } else {
            float val = __half2float(expert_output[idx]) * weight;
            atomicAdd((float*)&final_output[output_idx], val);
        }
    }
}

__global__ void weighted_scatter_reduce_fp32(
    const float* __restrict__ expert_output,
    const int* __restrict__ token_indices,
    const float* __restrict__ routing_weights,
    float* __restrict__ final_output,
    int num_tokens_expert,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_elements = num_tokens_expert * hidden_size;
    
    for (int i = idx; i < total_elements; i += stride) {
        int local_token_idx = i / hidden_size;
        int feature_idx = i % hidden_size;
        
        int global_token_idx = token_indices[local_token_idx];
        float weight = routing_weights[local_token_idx];
        int output_idx = global_token_idx * hidden_size + feature_idx;
        
        float value = expert_output[i] * weight;
        atomicAdd(&final_output[output_idx], value);
    }
}

struct GemmProblem {
    int m;
    int n;
    int k;
    const void* A;
    const void* B;
    void* C;
    int lda;
    int ldb;
    int ldc;
    int expert_id;
};

extern "C" {

int launch_fused_silu_mul_fp32(
    const float* gate,
    const float* up,
    float* output,
    int num_tokens,
    int intermediate_size,
    cudaStream_t stream
) {
    int total_elements = num_tokens * intermediate_size;
    
    if (intermediate_size % 4 == 0) {
        int vec4_elements = total_elements / 4;
        int threads = 256;
        int blocks = (vec4_elements + threads - 1) / threads;
        
        fused_silu_mul_kernel_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(gate),
            reinterpret_cast<const float4*>(up),
            reinterpret_cast<float4*>(output),
            vec4_elements
        );
    } else {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        fused_silu_mul_kernel<float><<<blocks, threads, 0, stream>>>(
            gate, up, output, total_elements
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int launch_fused_silu_mul_fp16(
    const half* gate,
    const half* up,
    half* output,
    int num_tokens,
    int intermediate_size,
    cudaStream_t stream
) {
    int total_elements = num_tokens * intermediate_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_silu_mul_kernel<half><<<blocks, threads, 0, stream>>>(
        gate, up, output, total_elements
    );
    
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int launch_weighted_scatter_fp32(
    const float* expert_output,
    const int* token_indices,
    const float* routing_weights,
    float* final_output,
    int num_tokens_expert,
    int hidden_size,
    cudaStream_t stream
) {
    int total_elements = num_tokens_expert * hidden_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    weighted_scatter_reduce_fp32<<<blocks, threads, 0, stream>>>(
        expert_output,
        token_indices,
        routing_weights,
        final_output,
        num_tokens_expert,
        hidden_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int grouped_gemm_moe_forward_fp32(
    int num_experts,
    const int* tokens_per_expert,
    const float* const* input_ptrs,
    const float* const* gate_weight_ptrs,
    const float* const* up_weight_ptrs,
    const float* const* down_weight_ptrs,
    const int* const* token_indices,
    const float* const* routing_weights,
    float* final_output,
    int total_tokens,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream
) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CUDA_CHECK(cudaMemsetAsync(final_output, 0, total_tokens * hidden_size * sizeof(float), stream));
    
    for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        int num_tokens = tokens_per_expert[expert_idx];
        if (num_tokens == 0) continue;
        
        float *gate_out, *up_out, *merged_out, *down_out;
        CUDA_CHECK(cudaMallocAsync(&gate_out, num_tokens * intermediate_size * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&up_out, num_tokens * intermediate_size * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&merged_out, num_tokens * intermediate_size * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&down_out, num_tokens * hidden_size * sizeof(float), stream));
        
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size,
            num_tokens,
            hidden_size,
            &alpha,
            gate_weight_ptrs[expert_idx], hidden_size,
            input_ptrs[expert_idx],       hidden_size,
            &beta,
            gate_out, intermediate_size
        ));
        
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size, num_tokens, hidden_size,
            &alpha,
            up_weight_ptrs[expert_idx], hidden_size,
            input_ptrs[expert_idx],     hidden_size,
            &beta,
            up_out, intermediate_size
        ));
        
        launch_fused_silu_mul_fp32(
            gate_out, up_out, merged_out,
            num_tokens, intermediate_size,
            stream
        );
        
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_size,
            num_tokens,
            intermediate_size,
            &alpha,
            down_weight_ptrs[expert_idx], intermediate_size,
            merged_out,                   intermediate_size,
            &beta,
            down_out, hidden_size
        ));
        
        launch_weighted_scatter_fp32(
            down_out,
            token_indices[expert_idx],
            routing_weights[expert_idx],
            final_output,
            num_tokens,
            hidden_size,
            stream
        );
        
        CUDA_CHECK(cudaFreeAsync(gate_out, stream));
        CUDA_CHECK(cudaFreeAsync(up_out, stream));
        CUDA_CHECK(cudaFreeAsync(merged_out, stream));
        CUDA_CHECK(cudaFreeAsync(down_out, stream));
    }
    
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return 0;
}

int grouped_gemm_moe_forward_fp16(
    int num_experts,
    const int* tokens_per_expert,
    const half* const* input_ptrs,
    const half* const* gate_weight_ptrs,
    const half* const* up_weight_ptrs,
    const half* const* down_weight_ptrs,
    const int* const* token_indices,
    const float* const* routing_weights,
    half* final_output,
    int total_tokens,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream
) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    CUDA_CHECK(cudaMemsetAsync(final_output, 0, total_tokens * hidden_size * sizeof(half), stream));
    
    for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        int num_tokens = tokens_per_expert[expert_idx];
        if (num_tokens == 0) continue;
        
        half *gate_out, *up_out, *merged_out, *down_out;
        CUDA_CHECK(cudaMallocAsync(&gate_out, num_tokens * intermediate_size * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&up_out, num_tokens * intermediate_size * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&merged_out, num_tokens * intermediate_size * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&down_out, num_tokens * hidden_size * sizeof(half), stream));
        
        CUBLAS_CHECK(cublasHgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size, num_tokens, hidden_size,
            &alpha,
            gate_weight_ptrs[expert_idx], hidden_size,
            input_ptrs[expert_idx],       hidden_size,
            &beta,
            gate_out, intermediate_size
        ));
        
        CUBLAS_CHECK(cublasHgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size, num_tokens, hidden_size,
            &alpha,
            up_weight_ptrs[expert_idx], hidden_size,
            input_ptrs[expert_idx],     hidden_size,
            &beta,
            up_out, intermediate_size
        ));
        
        launch_fused_silu_mul_fp16(
            gate_out, up_out, merged_out,
            num_tokens, intermediate_size,
            stream
        );
        
        CUBLAS_CHECK(cublasHgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_size, num_tokens, intermediate_size,
            &alpha,
            down_weight_ptrs[expert_idx], intermediate_size,
            merged_out,                   intermediate_size,
            &beta,
            down_out, hidden_size
        ));
        
        launch_weighted_scatter_fp32(
            reinterpret_cast<const float*>(down_out),
            token_indices[expert_idx],
            routing_weights[expert_idx],
            reinterpret_cast<float*>(final_output),
            num_tokens,
            hidden_size,
            stream
        );
        
        CUDA_CHECK(cudaFreeAsync(gate_out, stream));
        CUDA_CHECK(cudaFreeAsync(up_out, stream));
        CUDA_CHECK(cudaFreeAsync(merged_out, stream));
        CUDA_CHECK(cudaFreeAsync(down_out, stream));
    }
    
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}

}

extern "C" __global__ void silu_mul_kernel_fp32(
    const float* gate,
    const float* up,
    float* output,
    int num_tokens,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * intermediate_size;
    
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        float g = gate[i];
        float u = up[i];
        float sigmoid = 1.0f / (1.0f + expf(-g));
        output[i] = (sigmoid * g) * u;
    }
}

extern "C" __global__ void silu_mul_kernel_fp16(
    const half* gate,
    const half* up,
    half* output,
    int num_tokens,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * intermediate_size;
    
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float sigmoid = 1.0f / (1.0f + expf(-g));
        output[i] = __float2half((sigmoid * g) * u);
    }
}
