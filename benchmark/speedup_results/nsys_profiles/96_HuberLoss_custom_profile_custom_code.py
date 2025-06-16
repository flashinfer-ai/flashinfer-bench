import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

smooth_l1_loss_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float smooth_l1_loss_func(float x, float beta) {
    float abs_x = fabsf(x);
    if (abs_x < beta) {
        float r = x / beta;
        return 0.5f * r * r * beta;
    } else {
        return abs_x - 0.5f * beta;
    }
}

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ preds,
    const float* __restrict__ tgt,
    float* __restrict__ partial_sums,
    int size,
    float beta
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < size) {
        float diff = preds[idx] - tgt[idx];
        val = smooth_l1_loss_func(diff, beta);
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void finalize_loss_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ out,
    int num_blocks,
    int size
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        val += partial_sums[i];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out = sdata[0] / (float)size;
    }
}

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor preds, 
    torch::Tensor targets, 
    float beta
) {
    int size = preds.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    auto options = preds.options();
    auto partial_sums = torch::zeros({blocks}, options);
    auto out = torch::empty({}, options);

    size_t shared_mem_size = threads * sizeof(float);
    smooth_l1_loss_kernel<<<blocks, threads, shared_mem_size>>>(
        preds.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        size,
        beta
    );

    finalize_loss_kernel<<<1, threads, shared_mem_size>>>(
        partial_sums.data_ptr<float>(),
        out.data_ptr<float>(),
        blocks,
        size
    );

    return out;
}
"""

smooth_l1_loss_cpp_source = r"""
torch::Tensor smooth_l1_loss_cuda(torch::Tensor preds, torch::Tensor targets, float beta);
"""

smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss using a custom CUDA kernel.
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, predictions, targets):
        return smooth_l1_loss.smooth_l1_loss_cuda(predictions, targets, self.beta)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape, device="cuda"), 
            torch.randn(batch_size, *input_shape, device="cuda")]

def get_init_inputs():
    return []
