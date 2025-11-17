import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace torch_example_cuda {

__global__ void AddOneKernel(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] + 1;
  }
}

torch::Tensor add_one_cuda(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  
  auto y = torch::empty_like(x);
  
  int64_t n = x.numel();
  float* x_data = x.data_ptr<float>();
  float* y_data = y.data_ptr<float>();
  
  int64_t threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  
  AddOneKernel<<<blocks, threads, 0, stream>>>(x_data, y_data, n);
  
  return y;
}

}  // namespace torch_example_cuda
"""

cpp_source = """
#include <torch/extension.h>

namespace torch_example_cuda {
torch::Tensor add_one_cuda(torch::Tensor x);
}

torch::Tensor add_one(torch::Tensor x) {
  return torch_example_cuda::add_one_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one, "Add one to each element (CUDA)");
}
"""


def main():
    print("Compiling CUDA extension with torch.utils.cpp_extension.load_inline...")
    
    module = load_inline(
        name="add_one_cuda",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        verbose=True,
        with_cuda=True,
    )
    
    print("Compilation successful!")
    
    # Test the compiled module
    print("\nTesting the module...")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
    print(f"Input:  {x}")
    
    y = module.add_one(x)
    print(f"Output: {y}")
    
    expected = x + 1
    print(f"Expected: {expected}")
    print(f"Match: {torch.allclose(y, expected)}")


if __name__ == "__main__":
    main()

