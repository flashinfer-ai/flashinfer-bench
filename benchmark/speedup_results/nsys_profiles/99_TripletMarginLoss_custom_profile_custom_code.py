import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

triplet_margin_loss_code = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// A block-level reduction utility using shared memory.
// Each thread adds its local value into shared memory, and then we reduce within the block.
template <unsigned int blockSize>
__device__ void blockReduceSum(float* sdata, int tid) {
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512)  { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256)  { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128)  { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64)   { sdata[tid] += sdata[tid + 64]; }  __syncthreads(); }

    // Now we reduce within a single warp
    if (tid < 32) {
        volatile float* vsmem = sdata;
        if (blockSize >= 64) vsmem[tid] += vsmem[tid + 32];
        if (blockSize >= 32) vsmem[tid] += vsmem[tid + 16];
        if (blockSize >= 16) vsmem[tid] += vsmem[tid + 8];
        if (blockSize >= 8)  vsmem[tid] += vsmem[tid + 4];
        if (blockSize >= 4)  vsmem[tid] += vsmem[tid + 2];
        if (blockSize >= 2)  vsmem[tid] += vsmem[tid + 1];
    }
}

// Kernel 1: For each batch item (blockIdx.x), compute squared distance for anchor-positive
// and anchor-negative, compute triplet margin loss, store in partial_sums[blockIdx.x].
template <unsigned int blockSize>
__global__ void triplet_margin_loss_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ partial_sums,
    const float margin,
    const int dims)
{
    // Each block handles one "row" in the batch
    int row = blockIdx.x;
    // global offset for that row
    int offset = row * dims;

    // Each thread accumulates partial sums of (anchor - pos)^2 and (anchor - neg)^2
    float pos_val = 0.0f;
    float neg_val = 0.0f;

    for (int idx = threadIdx.x; idx < dims; idx += blockSize) {
        float diff_pos = anchor[offset + idx] - positive[offset + idx];
        float diff_neg = anchor[offset + idx] - negative[offset + idx];
        pos_val += diff_pos * diff_pos;
        neg_val += diff_neg * diff_neg;
    }

    // Reduce pos_val and neg_val inside the block
    __shared__ float s_pos[blockSize];
    __shared__ float s_neg[blockSize];

    int tid = threadIdx.x;
    s_pos[tid] = pos_val;
    s_neg[tid] = neg_val;

    blockReduceSum<blockSize>(s_pos, tid);
    blockReduceSum<blockSize>(s_neg, tid);

    if (tid == 0) {
        float pos_dist = sqrtf(s_pos[0]);
        float neg_dist = sqrtf(s_neg[0]);
        float trip_loss = fmaxf(pos_dist - neg_dist + margin, 0.0f);
        partial_sums[row] = trip_loss;
    }
}

// Kernel 2: Reduce partial_sums array into a single scalar and divide by batch_size for mean.
template <unsigned int blockSize>
__global__ void reduce_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ out,
    const int batch_size)
{
    __shared__ float sdata[blockSize];
    int tid = threadIdx.x;
    float val = 0.0f;

    // Stride through partial_sums
    for (int i = tid; i < batch_size; i += blockSize) {
        val += partial_sums[i];
    }
    sdata[tid] = val;
    blockReduceSum<blockSize>(sdata, tid);

    if (tid == 0) {
        // Store mean, replicate PyTorch's default "mean" reduction
        out[0] = sdata[0] / (float)batch_size;
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    double margin)
{
    TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.is_cuda(), "negative must be a CUDA tensor");
    TORCH_CHECK(anchor.dim() == 2, "anchor must be 2D");
    TORCH_CHECK(positive.dim() == 2, "positive must be 2D");
    TORCH_CHECK(negative.dim() == 2, "negative must be 2D");

    int batch_size = anchor.size(0);
    int dims = anchor.size(1);

    auto opts = anchor.options();
    // Intermediate array to hold per-row losses
    auto partial_sums = torch::empty({batch_size}, opts);
    auto out = torch::empty({}, opts);  // zero-dimensional scalar tensor

    const int blockSize = 256;
    // For kernel 1: one block per batch item
    dim3 grid(batch_size);
    dim3 block(blockSize);

    // Launch the triplet margin loss kernel
    triplet_margin_loss_kernel<blockSize><<<grid, block>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        static_cast<float>(margin),
        dims
    );

    // For kernel 2: reduce partial_sums into a single scalar
    reduce_kernel<blockSize><<<1, blockSize>>>(
        partial_sums.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size
    );

    return out;
}
''';

triplet_margin_loss_cpp = r'''
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    double margin);
''';

# Build the custom extension
triplet_margin_loss_ext = load_inline(
    name="triplet_margin_loss_ext",
    cpp_sources=triplet_margin_loss_cpp,
    cuda_sources=triplet_margin_loss_code,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
)

# New optimized model
class ModelNew(nn.Module):
    """
    An optimized version of the triplet margin loss model using a custom CUDA kernel.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Call the fused custom CUDA triplet margin loss
        loss = triplet_margin_loss_ext.triplet_margin_loss_cuda(anchor, positive, negative, float(self.margin))
        return loss
