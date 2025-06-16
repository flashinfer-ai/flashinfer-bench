import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for product reduction along dim=1 of a 3D tensor: [B, D1, D2]
# This kernel launches one block per (b, d2) pair, with blockDim.x threads covering D1.
# Threads coalesce partial products in shared memory using a block reduction.
prod_reduce_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// A helper macro to check for CUDA errors
#define CUDA_CHECK(err) if (err != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
}

// Warp-level reduce multiply
__inline__ __device__ float warpReduceProd(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduce multiply using shared memory, then warp reduce
__inline__ __device__ float blockReduceProd(float val) {
    static __shared__ float shared[32];  // one element per warp
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceProd(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Only the first warp in the block reduces all warps' results
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 1.0f;
    if (wid == 0) val = warpReduceProd(val);
    return val;
}

__global__ void prod_reduce_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int D1, int D2)
{
    // blockIdx.x corresponds to d2, blockIdx.y corresponds to b
    int d2 = blockIdx.x;
    int b  = blockIdx.y;

    // Each block handles the product for x[b, :, d2]
    // Each thread processes one element along dim=1
    int tid = threadIdx.x;  // range [0..(D1-1)]

    // Accumulate partial product
    float val = 1.0f;
    if (tid < D1) {
        // The striding in memory is B*D1*D2, indexing is b*D1*D2 + i*D2 + d2
        // For coalescing improvements, we rely on block dimension alignment
        // but we still have a stride across D2 in row-major format.
        val = x[b * (D1 * D2) + tid * D2 + d2];
    }

    // Reduce product in the block
    float prod = blockReduceProd(val);

    // Write the result from thread 0
    if (threadIdx.x == 0) {
        out[b * D2 + d2] = prod;
    }
}

torch::Tensor prod_reduce_cuda(torch::Tensor x, int64_t reduction_dim) {
    // We only support reducing dim=1 for a 3D tensor [B, D1, D2] here.
    if (x.dim() != 3) {
        throw std::runtime_error("Input must be a 3D tensor [B, D1, D2].");
    }
    if (reduction_dim != 1) {
        throw std::runtime_error("This custom kernel only supports reduction_dim=1.");
    }

    const auto B = x.size(0);
    const auto D1 = x.size(1);
    const auto D2 = x.size(2);

    // Allocate output of shape [B, D2]
    auto out = torch::zeros({B, D2}, x.options());

    // Configure kernel launch
    dim3 blocks(D2, B);        // one block per (b, d2)
    int threads = D1;          // each thread covers one element along dim=1
    if (threads > 1024) {
        throw std::runtime_error("D1 is too large for a single block. Please adjust kernel.");
    }

    // Launch kernel
    prod_reduce_dim1_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, D1, D2
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    CUDA_CHECK(err);

    return out;
}
""".strip()

prod_reduce_cpp_source = r"""
torch::Tensor prod_reduce_cuda(torch::Tensor x, int64_t reduction_dim);
"""

# Load/compile the inline extension
prod_reduce = load_inline(
    name="prod_reduce",
    cpp_sources=prod_reduce_cpp_source,
    cuda_sources=prod_reduce_source,
    functions=["prod_reduce_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs product reduction over dim=1 using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over. (Only dim=1 supported here)
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.prod_reduce = prod_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs product reduction over dim=1 using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with product reduction applied.
        """
        return self.prod_reduce.prod_reduce_cuda(x, self.dim)
