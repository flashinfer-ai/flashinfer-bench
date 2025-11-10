import torch
from tvm_ffi import Module
import tvm_ffi.cpp
from pathlib import Path
import re

cuda_source = """
// File: compile/add_one_cuda.cu
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace tvm_ffi_example_cuda {

__global__ void AddOneKernel(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] + 1;
  }
}

void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int64_t n = x.size(0);
  float* x_data = static_cast<float*>(x.data_ptr());
  float* y_data = static_cast<float*>(y.data_ptr());
  int64_t threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  AddOneKernel<<<blocks, threads, 0, stream>>>(x_data, y_data, n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cuda, tvm_ffi_example_cuda::AddOne);
}  // namespace tvm_ffi_example_cuda
"""

def main():
    mod: Module = tvm_ffi.cpp.load_inline(
        name='add_one_cuda',
        cuda_sources=cuda_source,
    )
    print("Compilation successful")

if __name__ == "__main__":
    main()