import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Error-checking macro
#define CUDA_CHECK(call)                                                       \
{                                                                              \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        printf("CUDA error at %s %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return torch::Tensor();                                               \
    }                                                                          \
}

// Kernel for argmax along dimension 2 for a 3D tensor of shape (B, M, N).
// Each block processes one (B, M) pair, threads reduce along N.
__global__ void argmax_dim2_kernel(const float* __restrict__ input,
                                   long* __restrict__ output,
                                   int B, int M, int N)
{
    // Block index: b*M + m
    int idx = blockIdx.x;
    if (idx >= B * M) return;

    int b = idx / M;
    int m = idx % M;

    // Start pointer for this row
    const float* row_ptr = input + (b * M + m) * N;

    // Thread index for partial reduction
    int tid = threadIdx.x;
    float max_val = -3.402823e+38f; // Small float
    long max_idx = 0;

    // Grid-stride loop if N > blockDim.x
    for(int n = tid; n < N; n += blockDim.x) {
        float val = row_ptr[n];
        if(val > max_val) {
            max_val = val;
            max_idx = n;
        }
    }

    // Use shared memory to finalize reduction within the block
    __shared__ float sdata_val[256];
    __shared__ long  sdata_idx[256];

    sdata_val[tid] = max_val;
    sdata_idx[tid] = max_idx;
    __syncthreads();

    // Parallel reduction
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            float v1 = sdata_val[tid];
            float v2 = sdata_val[tid + s];
            if(v2 > v1) {
                sdata_val[tid] = v2;
                sdata_idx[tid] = sdata_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // Write result
    if(tid == 0) {
        output[b * M + m] = sdata_idx[0];
    }
}

// Kernel for argmax along dimension 1 for a 3D tensor of shape (B, M, N).
// Each block processes one (B, N) pair, threads reduce along M.
__global__ void argmax_dim1_kernel(const float* __restrict__ input,
                                   long* __restrict__ output,
                                   int B, int M, int N)
{
    int idx = blockIdx.x;
    if (idx >= B * N) return;

    int b = idx / N;
    int n = idx % N;

    const float* col_ptr = input + b * M * N + n;
    int tid = threadIdx.x;

    float max_val = -3.402823e+38f;
    long max_idx = 0;

    for(int m = tid; m < M; m += blockDim.x) {
        float val = col_ptr[m * N];
        if(val > max_val) {
            max_val = val;
            max_idx = m;
        }
    }

    __shared__ float sdata_val[256];
    __shared__ long  sdata_idx[256];

    sdata_val[tid] = max_val;
    sdata_idx[tid] = max_idx;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            float v1 = sdata_val[tid];
            float v2 = sdata_val[tid + s];
            if(v2 > v1) {
                sdata_val[tid] = v2;
                sdata_idx[tid] = sdata_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if(tid == 0) {
        output[b * N + n] = sdata_idx[0];
    }
}

// Kernel for argmax along dimension 0 for a 3D tensor of shape (B, M, N).
// Each block processes one (M, N) pair, threads reduce along B.
__global__ void argmax_dim0_kernel(const float* __restrict__ input,
                                   long* __restrict__ output,
                                   int B, int M, int N)
{
    int idx = blockIdx.x;
    if (idx >= M * N) return;

    int m = idx / N;
    int n = idx % N;

    int tid = threadIdx.x;
    float max_val = -3.402823e+38f;
    long max_idx = 0;

    for(int b = tid; b < B; b += blockDim.x) {
        float val = input[b * M * N + m * N + n];
        if(val > max_val) {
            max_val = val;
            max_idx = b;
        }
    }

    __shared__ float sdata_val[256];
    __shared__ long  sdata_idx[256];

    sdata_val[tid] = max_val;
    sdata_idx[tid] = max_idx;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            float v1 = sdata_val[tid];
            float v2 = sdata_val[tid + s];
            if(v2 > v1) {
                sdata_val[tid] = v2;
                sdata_idx[tid] = sdata_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if(tid == 0) {
        output[m * N + n] = sdata_idx[0];
    }
}

torch::Tensor argmax_cuda(torch::Tensor x, int dim) {
    // Ensure x is float tensor on CUDA
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input must be Float32");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 3, "Only 3D tensors supported in this example");

    int B = sizes[0];
    int M = sizes[1];
    int N = sizes[2];

    // Argmax produces a tensor of int64 indices
    torch::Tensor out;
    if (dim == 0) {
        out = torch::empty({M, N}, torch::dtype(torch::kLong).device(x.device()));
    } else if (dim == 1) {
        out = torch::empty({B, N}, torch::dtype(torch::kLong).device(x.device()));
    } else {
        out = torch::empty({B, M}, torch::dtype(torch::kLong).device(x.device()));
    }

    // Launch
    const int blockSize = 256;
    cudaError_t err;

    if (dim == 2) {
        int gridSize = (B * M + blockSize - 1) / blockSize;
        argmax_dim2_kernel<<<gridSize, blockSize>>>(x.data_ptr<float>(),
                                                    out.data_ptr<long>(),
                                                    B, M, N);
    } else if (dim == 1) {
        int gridSize = (B * N + blockSize - 1) / blockSize;
        argmax_dim1_kernel<<<gridSize, blockSize>>>(x.data_ptr<float>(),
                                                    out.data_ptr<long>(),
                                                    B, M, N);
    } else {
        int gridSize = (M * N + blockSize - 1) / blockSize;
        argmax_dim0_kernel<<<gridSize, blockSize>>>(x.data_ptr<float>(),
                                                    out.data_ptr<long>(),
                                                    B, M, N);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\\n", cudaGetErrorString(err));
        return torch::Tensor();
    }

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel sync error: %s\\n", cudaGetErrorString(err));
        return torch::Tensor();
    }

    return out;
}
'''.strip()

argmax_cpp_source = r'''
torch::Tensor argmax_cuda(torch::Tensor x, int dim);
'''

# Build the extension
argmax_extension = load_inline(
    name="argmax_extension",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension with a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmax_extension.argmax_cuda(x, self.dim)
