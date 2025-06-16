import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

gelu_approx_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__constant__ float c[3];  // c[0] = 0.5f, c[1] = sqrtf(2.0f / M_PI), c[2] = 0.044715f

__device__ __forceinline__ float gelu_approx_scalar(float x) {
    float x3 = x * x * x;
    float temp = x + c[2] * x3;
    float val = c[1] * temp;
    float t = tanhf(val);
    return c[0] * x * (1.f + t);
}

__device__ __forceinline__ float4 gelu_approx_float4(const float4 &val) {
    return make_float4(
        gelu_approx_scalar(val.x),
        gelu_approx_scalar(val.y),
        gelu_approx_scalar(val.z),
        gelu_approx_scalar(val.w)
    );
}

__global__ void gelu_approx_kernel_vector(const float* __restrict__ x, float* __restrict__ out, size_t numel4, size_t leftover) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel4) {
        float4 v = reinterpret_cast<const float4*>(x)[idx];
        float4 r = gelu_approx_float4(v);
        reinterpret_cast<float4*>(out)[idx] = r;
    }
    if (idx == 0) {
        for (int i = 0; i < leftover; i++) {
            size_t offset = numel4 * 4 + i;
            out[offset] = gelu_approx_scalar(x[offset]);
        }
    }
}

torch::Tensor gelu_approx_cuda(torch::Tensor input) {
    auto x = input.contiguous();
    auto out = torch::empty_like(x);
    size_t size = x.numel();
    size_t numel4 = size / 4;
    size_t leftover = size % 4;

    int block_size = 256;
    int grid_size = (numel4 + block_size - 1) / block_size;

    gelu_approx_kernel_vector<<<grid_size, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), numel4, leftover);
    return out;
}

void set_constants() {
    float val0 = 0.5f;
    float val1 = sqrtf(2.0f / (float)M_PI);
    float val2 = 0.044715f;
    float vals[3] = {val0, val1, val2};
    cudaMemcpyToSymbol(c, vals, 3 * sizeof(float));
}
'''

gelu_approx_cpp_source = r'''
torch::Tensor gelu_approx_cuda(torch::Tensor input);
void set_constants();
'''

gelu_approx = load_inline(
    name="gelu_approx",
    cpp_sources=gelu_approx_cpp_source,
    cuda_sources=gelu_approx_source,
    functions=["gelu_approx_cuda", "set_constants"],
    verbose=False
)

# Initialize constants in constant memory
gelu_approx.set_constants()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_approx = gelu_approx

    def forward(self, x):
        return self.gelu_approx.gelu_approx_cuda(x)

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim).cuda()]

def get_init_inputs():
    return []
