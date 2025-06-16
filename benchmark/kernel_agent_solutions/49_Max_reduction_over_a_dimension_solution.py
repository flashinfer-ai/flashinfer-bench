import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduce_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__ float blockReduceMax(float val) {
    static __shared__ float shared[32];  // Up to 32 warps per block
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceMax(val); 
    // Write reduced value for this warp to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    float blockVal = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        blockVal = warpReduceMax(blockVal);
    }
    return blockVal;
}

// Kernel assumes x is 3D: (B, D, M), reducing over D (dim=1)
__global__ void max_reduce_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  int B, int D, int M, int reduceDim) {
    // Currently only supports dim=1
    if (reduceDim != 1) return;

    int idx = blockIdx.x;
    if (idx >= B * M) return;

    int b = idx / M;  // which batch
    int m = idx % M;  // position in the last dimension

    float threadMax = -FLT_MAX;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = x[b * D * M + d * M + m];
        threadMax = fmaxf(threadMax, val);
    }

    // Parallel block-wide reduction
    float blockMax = blockReduceMax(threadMax);
    if (threadIdx.x == 0) {
        out[b * M + m] = blockMax;
    }
}

torch::Tensor max_reduce_cuda(torch::Tensor x, int dim) {
    // Only handle a 3D tensor: (B, D, M)
    TORCH_CHECK(x.dim() == 3, "Expected 3D tensor");
    TORCH_CHECK(dim == 1, "Currently only dim=1 is supported");
    int B = x.size(0);
    int D = x.size(1);
    int M = x.size(2);

    auto out = torch::zeros({B, M}, x.options());

    // Launch config
    int threads = 256;
    int blocks = B * M;

    max_reduce_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                          out.data_ptr<float>(),
                                          B, D, M, dim);
    // Check for kernel errors
    auto err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    return out;
}
""";

max_reduce_cpp_hdr = "torch::Tensor max_reduce_cuda(torch::Tensor x, int dim);"

max_reduce_module = load_inline(
    name="max_reduce_module",
    cpp_sources=max_reduce_cpp_hdr,
    cuda_sources=max_reduce_src,
    functions=["max_reduce_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs max-reduction over dim=1 on a [B, D, M] tensor using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.custom_max_reduce = max_reduce_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_max_reduce.max_reduce_cuda(x, self.dim)
