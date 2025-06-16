import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

softmax_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <float.h>

// Simple CUDA error-checking macro
#define CUDA_CHECK_ERROR()                                                   \
    do {                                                                     \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess) {                                            \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err));                                 \
            return;                                                          \
        }                                                                    \
    } while(0)

// Warp-level max reduction
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level max reduction using shared memory + warp reduction
__inline__ __device__ float blockReduceMax(float val) {
    static __shared__ float shared[32];  // Assumes up to 1024 threads, 32 warps max
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Each warp performs partial reduction
    val = warpReduceMax(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Only threads within first warp read those reduced values
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;

    // Final reduce within a single warp
    if (wid == 0) {
        val = warpReduceMax(val);
    }
    return val;
}

// Warp-level sum reduction
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level sum reduction using shared memory + warp reduction
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Each warp performs partial reduction
    val = warpReduceSum(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Only threads within first warp read those reduced values
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    // Final reduce within a single warp
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Kernel to compute softmax row-by-row
__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int batch_size,
                               int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    int row_start = row * dim;

    // 1) Compute max value per row
    float thread_max = -FLT_MAX;
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        thread_max = fmaxf(thread_max, input[row_start + idx]);
    }
    float max_val = blockReduceMax(thread_max);
    __syncthreads();

    // 2) Compute sum of exponentials
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        float val = __expf(input[row_start + idx] - max_val);
        thread_sum += val;
    }
    float sum_val = blockReduceSum(thread_sum);
    __syncthreads();

    // 3) Write output (normalized exponent)
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        float val = __expf(input[row_start + idx] - max_val) / sum_val;
        output[row_start + idx] = val;
    }
    __syncthreads();

    CUDA_CHECK_ERROR();
}

torch::Tensor softmax_cuda_forward(torch::Tensor input) {
    // Input shape: [batch_size, dim]
    int batch_size = input.size(0);
    int dim = input.size(1);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = batch_size;

    softmax_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                        output.data_ptr<float>(),
                                        batch_size,
                                        dim);
    return output;
}
""";

softmax_cpp = r"""
torch::Tensor softmax_cuda_forward(torch::Tensor input);
""";

softmax_mod = load_inline(
    name="softmax_mod",
    cpp_sources=softmax_cpp,
    cuda_sources=softmax_src,
    functions=["softmax_cuda_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_mod = softmax_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_mod.softmax_cuda_forward(x)
