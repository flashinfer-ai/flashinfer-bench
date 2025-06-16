import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cosine_loss_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Compute per-batch cosine similarity for each pair of vectors in predictions & targets
//   Inputs:  predictions [batch_size, dim]
//            targets     [batch_size, dim]
//   Output:  cos_out    [batch_size]
__global__ void compute_cos_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ cos_out,
    const int dim
)
{
    // Each block corresponds to one row (one batch element).
    // We assume blockIdx.x < batch_size.
    // We do partial sums over 'dim' using blockDim.x threads.
    int row = blockIdx.x;  // which sample in the batch
    int base_idx = row * dim;
    float dot_val = 0.0f;
    float normA_val = 0.0f;
    float normB_val = 0.0f;

    for (int tid = threadIdx.x; tid < dim; tid += blockDim.x) {
        float p = predictions[base_idx + tid];
        float t = targets[base_idx + tid];
        dot_val += p * t;
        normA_val += p * p;
        normB_val += t * t;
    }

    // Reduction in shared memory
    __shared__ float sdot[256];
    __shared__ float snormA[256];
    __shared__ float snormB[256];

    sdot[threadIdx.x] = dot_val;
    snormA[threadIdx.x] = normA_val;
    snormB[threadIdx.x] = normB_val;
    __syncthreads();

    // Parallel reduction for dot, normA, normB
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdot[threadIdx.x] += sdot[threadIdx.x + stride];
            snormA[threadIdx.x] += snormA[threadIdx.x + stride];
            snormB[threadIdx.x] += snormB[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Compute final cosine similarity for the row
    if (threadIdx.x == 0) {
        float final_dot = sdot[0];
        float final_normA = sqrtf(snormA[0]);
        float final_normB = sqrtf(snormB[0]);
        // Numerically safe division
        float cos_val = final_dot / ((final_normA * final_normB) + 1e-8f);
        cos_out[row] = cos_val;
    }
}

// Kernel 2: Reduce an array of size batch_size to produce a mean
//   Input:  cos_out [batch_size]
//   Output: single scalar in out[0] (which will later be turned into a 0-dim tensor)
__global__ void reduce_mean_kernel(const float* __restrict__ cos_out,
                                   float* __restrict__ out,
                                   const int batch_size)
{
    // For simplicity, do a standard parallel sum of cos_out, then we divide by batch_size.
    __shared__ float sdata[256];
    float val = 0.0f;
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; 
         tid < batch_size; 
         tid += blockDim.x * gridDim.x) {
        val += cos_out[tid];
    }
    sdata[threadIdx.x] = val;
    __syncthreads();

    // Parallel reduction in one block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write result
    if (threadIdx.x == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// Main function to compute the cosine-loss = mean(1 - cos_sim)
torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets)
{
    // Inputs should be [batch_size, dim]
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");

    auto batch_size = predictions.size(0);
    auto dim = predictions.size(1);

    // Create an output array for cos values
    // cos_out shape: [batch_size]
    auto cos_out = torch::zeros({batch_size}, predictions.options());

    // 1) Compute cos for each row
    int threads = 256;
    dim3 grid1(batch_size);
    compute_cos_kernel<<<grid1, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        cos_out.data_ptr<float>(),
        dim
    );

    // 2) Reduce cos_out to a single mean
    auto temp = torch::zeros({1}, predictions.options()); // accumulator
    int blocks = 1;
    reduce_mean_kernel<<<blocks, threads>>>(
        cos_out.data_ptr<float>(),
        temp.data_ptr<float>(),
        batch_size
    );
    // Now temp[0] holds sum of cos values

    // Transfer sum to host or do inline transform in device
    // We'll do in Python: mean_cos = temp[0] / batch_size
    // Then out = 1 - mean_cos
    auto mean_cos = temp / static_cast<float>(batch_size);
    auto result = torch::empty({}, predictions.options());  // shape []
    result.fill_(1.0f - mean_cos.item<float>());

    return result;
}
'''

cosine_loss_header = r'''
torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
'''

# Build the custom CUDA extension
cosine_loss_lib = load_inline(
    name="cosine_loss_lib",
    cpp_sources=cosine_loss_header,
    cuda_sources=cosine_loss_source,
    functions=["cosine_loss_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    A model that computes Cosine Similarity Loss for comparing vectors using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.lossop = cosine_loss_lib

    def forward(self, predictions, targets):
        # Call our custom kernel that returns a 0-dim tensor
        return self.lossop.cosine_loss_cuda(predictions, targets)
