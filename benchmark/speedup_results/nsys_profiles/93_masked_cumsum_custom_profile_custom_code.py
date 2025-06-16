import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/*
  This kernel computes a masked cumulative sum along dimension=1 for a 2D tensor
  of shape (batch_size, length). It assumes:
    - x and mask have the same shape: (batch_size, length)
    - mask is bool
    - The result is written to 'out' which has the same shape (batch_size, length)
*/
__global__ void masked_cumsum_kernel(const float* __restrict__ x,
                                     const bool* __restrict__ mask,
                                     float* __restrict__ out,
                                     int batch_size,
                                     int length) {
    // One block per row
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Starting index in x/mask/out for this row
    int start_idx = row * length;
    // Shared memory for partial sums within a chunk
    extern __shared__ float sdata[];

    float partialSum = 0.0f;

    // Process the row in chunks of blockDim.x
    for (int chunkStart = 0; chunkStart < length; chunkStart += blockDim.x) {
        int tid = threadIdx.x;
        int i = chunkStart + tid;
        // Load data into shared memory
        float val = 0.0f;
        if (i < length) {
            val = x[start_idx + i] * static_cast<float>(mask[start_idx + i]);
        }
        sdata[tid] = val;

        __syncthreads();

        // In-block inclusive scan
        // offset: distance between elements to add
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            __syncthreads();
            if (tid >= offset && i < length) {
                sdata[tid] += sdata[tid - offset];
            }
        }

        __syncthreads();

        // Add the partialSum from previous chunk
        if (i < length) {
            sdata[tid] += partialSum;
        }

        __syncthreads();

        // Write results back to global memory
        if (i < length) {
            out[start_idx + i] = sdata[tid];
        }

        __syncthreads();

        // Update partialSum for the next chunk
        if ((chunkStart + blockDim.x - 1) < length) {
            // Last valid element in this chunk
            int lastIdxInChunk = min(blockDim.x - 1, length - chunkStart - 1);
            if (tid == lastIdxInChunk) {
                // sdata[lastIdxInChunk] now has the inclusive sum up to that point
                partialSum = sdata[lastIdxInChunk];
            }
        } else {
            // This was the last chunk
            if (i == (length - 1)) {
                partialSum = sdata[tid];
            }
        }
        __syncthreads();
    }
}

// Host function to call the kernel
torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    // This implementation supports dim=1 for a 2D tensor
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(mask.dim() == 2, "Input mask must be 2D");
    TORCH_CHECK(x.size(0) == mask.size(0) && x.size(1) == mask.size(1),
                "x and mask must match in shape");
    TORCH_CHECK(dim == 1, "This custom kernel only supports dim=1 for 2D tensors");

    int batch_size = x.size(0);
    int length     = x.size(1);

    auto out = torch::zeros_like(x);

    // Configure the kernel
    const int blockSize = 256;  // can be tuned
    dim3 block(blockSize);
    dim3 grid(batch_size);

    // Shared memory size: blockSize * sizeof(float)
    size_t sharedMemSize = blockSize * sizeof(float);

    masked_cumsum_kernel<<<grid, block, sharedMemSize>>>(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        batch_size,
        length
    );

    return out;
}
""";

masked_cumsum_cpp_source = r"""
torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim);
""";

masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model performing a masked cumulative sum using a custom CUDA kernel.
    Only supports dim=1 for 2D tensors.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)


# The following functions replicate the original user's interface for data generation:
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()
    return [x, mask]

def get_init_inputs():
    return [dim]
