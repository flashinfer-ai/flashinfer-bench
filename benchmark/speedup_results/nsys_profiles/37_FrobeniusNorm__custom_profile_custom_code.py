import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

# Inline CUDA code for Frobenius norm normalization
fro_norm_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel to compute partial sums of x^2 in shared memory.
__global__ void partial_sum_squares_kernel(const float* __restrict__ input,
                                          float* __restrict__ partial_sums,
                                          const int64_t size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (i < size) {
        float tmp = input[i];
        val = tmp * tmp;
    }
    sdata[tid] = val;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write out reduced result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel to reduce partial sums into a single value.
__global__ void final_reduce_kernel(const float* __restrict__ partial_sums,
                                    float* __restrict__ norm,
                                    const int num_blocks) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    float val = 0.0f;
    if (i < num_blocks) {
        val = partial_sums[i];
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        norm[0] = sdata[0];
    }
}

// Kernel to take the square root of the single-element norm on GPU.
__global__ void sqrt_kernel(float* __restrict__ norm) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        norm[0] = sqrtf(norm[0]);
    }
}

// Kernel to divide every element of input by the computed norm.
__global__ void elementwise_div_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const float* __restrict__ norm,
                                       const int64_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = input[i] / norm[0];
    }
}

// Orchestrator function to compute Frobenius norm on GPU and normalize.
torch::Tensor fro_norm_forward_cuda(torch::Tensor x) {
    auto size = x.numel();
    // Create output tensor
    auto out = torch::empty_like(x);

    // Number of threads and blocks
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // Allocate partial sums tensor
    auto partial_sums = torch::empty({blocks}, x.options().dtype(torch::kFloat));
    // Allocate norm tensor (single element)
    auto norm = torch::zeros({1}, x.options().dtype(torch::kFloat));

    // 1) Compute partial sums of x^2
    partial_sum_squares_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        size
    );

    // 2) Final reduce kernel to sum partial_sums
    const int final_threads = 256;
    const int final_blocks = 1;  // We just need one block to reduce partial_sums
    final_reduce_kernel<<<final_blocks, final_threads, final_threads * sizeof(float)>>>(
        partial_sums.data_ptr<float>(),
        norm.data_ptr<float>(),
        blocks
    );

    // 3) Take sqrt of norm
    sqrt_kernel<<<1,1>>>(norm.data_ptr<float>());

    // 4) Divide input by norm
    elementwise_div_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        norm.data_ptr<float>(),
        size
    );

    return out;
}
""";

fro_norm_cpp_source = r"""
torch::Tensor fro_norm_forward_cuda(torch::Tensor x);
""";

# Compile the inline CUDA extension
fro_norm_extension = load_inline(
    name="fro_norm_extension",
    cpp_sources=fro_norm_cpp_source,
    cuda_sources=fro_norm_source,
    functions=["fro_norm_forward_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fro_norm_extension = fro_norm_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fro_norm_extension.fro_norm_forward_cuda(x)

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return []
