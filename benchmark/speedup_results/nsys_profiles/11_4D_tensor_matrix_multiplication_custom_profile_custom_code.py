import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source for 4D-tensor x 2D-matrix multiplication
matmul4d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul4d_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int b, int i, int j, int l, int k) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = b * i * j * k;

    if (idx < total) {
        int temp = idx;
        int b_idx = temp / (i * j * k);
        temp %= (i * j * k);
        int i_idx = temp / (j * k);
        temp %= (j * k);
        int j_idx = temp / k;
        int k_idx = temp % k;

        float sum_val = 0.0f;
        for (int l_idx = 0; l_idx < l; l_idx++) {
            long long A_index = ((long long)b_idx * i + i_idx) * j + j_idx;
            A_index = A_index * l + l_idx;
            long long B_index = (long long)l_idx * k + k_idx;
            sum_val += A[A_index] * B[B_index];
        }

        long long C_index = ((long long)b_idx * i + i_idx) * j + j_idx;
        C_index = C_index * k + k_idx;
        C[C_index] = sum_val;
    }
}

torch::Tensor matmul4d_cuda(torch::Tensor A, torch::Tensor B) {
    // A: shape (b, i, j, l)
    // B: shape (l, k)
    const int b = A.size(0);
    const int i = A.size(1);
    const int j = A.size(2);
    const int l = A.size(3);
    const int k = B.size(1);

    auto C = torch::zeros({b, i, j, k}, A.options());
    const int total = b * i * j * k;
    const int blockSize = 256;
    const int gridSize = (total + blockSize - 1) / blockSize;

    matmul4d_kernel<<<gridSize, blockSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        b, i, j, l, k
    );

    return C;
}
"""

matmul4d_cpp_source = "torch::Tensor matmul4d_cuda(torch::Tensor A, torch::Tensor B);"

# Load the inline extension
matmul4d = load_inline(
    name="matmul4d_extension",
    cpp_sources=matmul4d_cpp_source,
    cuda_sources=matmul4d_source,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    functions=["matmul4d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul4d = matmul4d

    def forward(self, A, B):
        return self.matmul4d.matmul4d_cuda(A, B)

# Inputs for the model
b = 16
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    # Same shapes as original model
    A = torch.randn(b, i, j, l)
    B = torch.randn(l, k)
    return [A, B]

def get_init_inputs():
    # No special initialization needed
    return []
