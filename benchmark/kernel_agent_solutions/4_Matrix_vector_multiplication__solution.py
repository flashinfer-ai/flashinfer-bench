import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matvec_mul_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void matvec_mul_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  const int M,
                                  const int K) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tx = threadIdx.x;
    float sum = 0.0f;

    // Each block processes one row of A
    // Each thread accumulates partial sum of that row * B
    for (int col = tx; col < K; col += blockDim.x) {
        sum += A[row * K + col] * B[col];
    }

    // Store partial sums in shared memory
    sdata[tx] = sum;
    __syncthreads();

    // Reduce within the block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tx < offset) {
            sdata[tx] += sdata[tx + offset];
        }
        __syncthreads();
    }

    // The first thread in each block writes the result to C
    if (tx == 0) {
        C[row] = sdata[0];
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    // Expecting A of shape (M, K), B of shape (K) or (K, 1)
    // Output will have shape (M, 1)
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be 1D or 2D with second dim == 1");
    
    int M = A.size(0);
    int K = A.size(1);

    // If B is (K,1), we flatten it to (K)
    auto B_flat = B.dim() == 2 ? B.view({B.size(0)}) : B;

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::empty({M, 1}, options);

    // Launch kernel
    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid(M);

    size_t sharedMemSize = block_size * sizeof(float);
    matvec_mul_kernel<<<grid, block, sharedMemSize>>>(
        A.data_ptr<float>(),
        B_flat.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        K
    );

    return C;
}
"""

matvec_mul_cpp_source = """
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Load custom CUDA extension
matvec_mul = load_inline(
    name="matvec_mul",
    cpp_sources=matvec_mul_cpp_source,
    cuda_sources=matvec_mul_source,
    functions=["matvec_mul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication (C = A * B)
    using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec_mul = matvec_mul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec_mul.matvec_mul_cuda(A.contiguous(), B.contiguous())
