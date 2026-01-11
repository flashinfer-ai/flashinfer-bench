# Agent Instructions: CUDA GEMM Implementation with TVM FFI

## Task Overview

Write a complete CUDA implementation solving the GEMM definition `gemm_n4096_k4096` and output it as a JSON file to:

**Output Path**: `Example-FlashInfer-Trace/solutions/agent_vibecode_gemm.json`

The implementation must use TVM FFI bindings and conform to the Solution JSON schema.

## Target Operation

**Operation**: General Matrix Multiply (GEMM)
**Formula**: `C = A @ B.T`
**Shapes**:
- A: `[M, K]` where M is variable, K = 4096
- B: `[N, K]` where N = 4096, K = 4096
- C: `[M, N]`
- **Data type**: `float16` (FP16)

## Solution Structure

Your solution **must** include exactly 3 source files with these names:

1. **`kernel.h`**: Header file with function declarations and shared definitions
2. **`kernel.cu`**: CUDA kernel device code implementation
3. **`main.cpp`**: TVM FFI host code with bindings

## TVM FFI Usage

### Headers files for binding
```cpp
#include <tvm/ffi/container/tensor.h>   // TensorView: tensor arguments
#include <tvm/ffi/function.h>           // TVM_FFI_DLL_EXPORT_TYPED_FUNC
#include <tvm/ffi/error.h>              // TVM_FFI_ICHECK, TVM_FFI_THROW
#include <tvm/ffi/extra/c_env_api.h>    // TVMFFIEnvGetStream
#include <cuda_fp16.h>
#include "kernel.h"
```

### Function Signature
The exported function **must** be named `run` and match the definition's input/output names:

```cpp
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C);
```

**Important**: The function takes A, B, and C as parameters. C is pre-allocated by the caller.

### TVM FFI Binding Instructions
Use TVM_FFI_DLL_EXPORT_TYPED_FUNC to expose the function:

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
```

### Input Validation
Validate inputs using TVM FFI error handling:

```cpp
// Check dimensions
TVM_FFI_ICHECK_EQ(A.ndim(), 2) << "A must be 2D";
TVM_FFI_ICHECK_EQ(B.ndim(), 2) << "B must be 2D";
TVM_FFI_ICHECK_EQ(C.ndim(), 2) << "C must be 2D";

// Check shapes
TVM_FFI_ICHECK_EQ(A.size(1), 4096) << "A.shape[1] must be 4096 (K)";
TVM_FFI_ICHECK_EQ(B.size(0), 4096) << "B.shape[0] must be 4096 (N)";
TVM_FFI_ICHECK_EQ(B.size(1), 4096) << "B.shape[1] must be 4096 (K)";

// Check shape compatibility
TVM_FFI_ICHECK_EQ(A.size(1), B.size(1)) << "K dimension mismatch";
TVM_FFI_ICHECK_EQ(C.size(0), A.size(0)) << "M dimension mismatch";
TVM_FFI_ICHECK_EQ(C.size(1), B.size(0)) << "N dimension mismatch";

// Check data types (float16)
TVM_FFI_ICHECK_EQ(A.dtype().code, kDLFloat) << "A must be float type";
TVM_FFI_ICHECK_EQ(A.dtype().bits, 16) << "A must be float16";
TVM_FFI_ICHECK_EQ(B.dtype().code, kDLFloat) << "B must be float type";
TVM_FFI_ICHECK_EQ(B.dtype().bits, 16) << "B must be float16";
TVM_FFI_ICHECK_EQ(C.dtype().code, kDLFloat) << "C must be float type";
TVM_FFI_ICHECK_EQ(C.dtype().bits, 16) << "C must be float16";

// Check device
TVM_FFI_ICHECK_EQ(A.device().device_type, kDLCUDA) << "A must be on CUDA";
TVM_FFI_ICHECK_EQ(B.device().device_type, kDLCUDA) << "B must be on CUDA";
TVM_FFI_ICHECK_EQ(C.device().device_type, kDLCUDA) << "C must be on CUDA";
```

### CUDA Stream Management
Get the CUDA stream from TVM FFI environment:

```cpp
DLDevice dev = A.device();
cudaStream_t stream = static_cast<cudaStream_t>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));

// Launch kernel on the stream
kernel_launch<<<grid, block, 0, stream>>>(args...);
```

### Memory Access
Access tensors through TensorView API:

```cpp
const __half* A_data = static_cast<const __half*>(A.data_ptr());
const __half* B_data = static_cast<const __half*>(B.data_ptr());
__half* C_data = static_cast<__half*>(C.data_ptr());

int64_t M = A.size(0);
int64_t K = A.size(1);
int64_t N = B.size(0);
```

## File Organization

### File Structure

**File 1: `kernel.h`**
- CUDA kernel function declarations
- Host launcher function declarations
- Shared constants and type definitions
- Include guards

**File 2: `kernel.cu`**
- `__global__` kernel implementations
- `__device__` helper functions
- Host-side kernel launcher function
- CUDA-specific optimizations (shared memory, tensor cores, etc.)

**File 3: `main.cpp`**
- TVM FFI bindings
- `run` function that matches definition signature: `void run(TensorView A, TensorView B, TensorView C)`
- Input validation using `TVM_FFI_ICHECK_*` macros
- Stream management via `TVMFFIEnvGetStream()`
- `TVM_FFI_DLL_EXPORT_TYPED_FUNC` for function export

## JSON Schema Format

The output JSON must conform to the Solution schema and be written to:

**`Example-FlashInfer-Trace/solutions/agent_vibecode_gemm.json`**

### FlashInfer-Bench Solution JSON Structure

```json
{
  "name": "agent_example_gemm",
  "definition": "gemm_n4096_k4096",
  "description": "High-performance CUDA GEMM implementation for C = A @ B.T using TVM FFI bindings",
  "author": "vibecode-agent",
  "spec": {
    "language": "cuda",
    "target_hardware": [
      "NVIDIA_H100",
      "NVIDIA_A100"
    ],
    "dependencies": [],
    "entry_point": "main.cpp::run"
  },
  "sources": [
    {
      "path": "kernel.h",
      "content": "... complete header file content as string ..."
    },
    {
      "path": "kernel.cu",
      "content": "... complete CUDA kernel code as string ..."
    },
    {
      "path": "main.cpp",
      "content": "... complete TVM FFI binding code as string ..."
    }
  ]
}
```

### Critical Schema Fields

| Field | Value | Notes |
|-------|-------|-------|
| `name` | `"agent_vibecode_gemm"` | Unique identifier for this solution |
| `definition` | `"gemm_n4096_k4096"` | **Must** match the definition name exactly |
| `language` | `"cuda"` | Lowercase, primary language |
| `target_hardware` | Array of strings | e.g., `["NVIDIA_H100", "NVIDIA_A100"]` |
| `entry_point` | `"main.cpp::run"` | Format: `{filename}::{function_name}` |
| `sources` | Array of 3 file objects | Each with `path` and `content` fields |

### Entry Point Convention

- Format: `"main.cpp::run"`
- The function `run` must be exposed via `TVM_FFI_DLL_EXPORT_TYPED_FUNC`

## Implementation Example

### kernel.h
```cpp
#ifndef GEMM_N4096_K4096_KERNEL_H
#define GEMM_N4096_K4096_KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int GEMM_N_CONST = 4096;
constexpr int GEMM_K_CONST = 4096;

// Kernel launcher function
void gemm_n4096_k4096_launch(
    const __half* A,
    const __half* B,
    __half* C,
    int M,
    cudaStream_t stream
);

#endif // GEMM_N4096_K4096_KERNEL_H
```

### kernel.cu
```cpp
#include "kernel.h"
#include <mma.h>  // For tensor cores

using namespace nvcuda;

// Kernel configuration
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;

__global__ void gemm_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M
) {
    // Implement optimized GEMM with:
    // - Shared memory tiling
    // - WMMA/Tensor Core operations
    // - Coalesced memory access
    // - Proper synchronization

    // C = A @ B.T
    // A is [M, 4096], B is [4096, 4096], C is [M, 4096]
}

void gemm_n4096_k4096_launch(
    const __half* A,
    const __half* B,
    __half* C,
    int M,
    cudaStream_t stream
) {
    if (M <= 0) return;

    dim3 block(256);
    dim3 grid((GEMM_N_CONST + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M);
}
```

### main.cpp
```cpp
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cuda_fp16.h>
#include "kernel.h"

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    TVM_FFI_ICHECK_EQ(A.ndim(), 2) << "A must be 2D";
    TVM_FFI_ICHECK_EQ(B.ndim(), 2) << "B must be 2D";
    TVM_FFI_ICHECK_EQ(C.ndim(), 2) << "C must be 2D";

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    TVM_FFI_ICHECK_EQ(K, 4096) << "A.shape[1] must be 4096 (K)";
    TVM_FFI_ICHECK_EQ(N, 4096) << "B.shape[0] must be 4096 (N)";
    TVM_FFI_ICHECK_EQ(B.size(1), 4096) << "B.shape[1] must be 4096 (K)";
    TVM_FFI_ICHECK_EQ(C.size(0), M) << "C.shape[0] must match A.shape[0] (M)";
    TVM_FFI_ICHECK_EQ(C.size(1), N) << "C.shape[1] must be 4096 (N)";

    TVM_FFI_ICHECK_EQ(A.dtype().code, kDLFloat) << "A must be float type";
    TVM_FFI_ICHECK_EQ(A.dtype().bits, 16) << "A must be float16";
    TVM_FFI_ICHECK_EQ(B.dtype().code, kDLFloat) << "B must be float type";
    TVM_FFI_ICHECK_EQ(B.dtype().bits, 16) << "B must be float16";
    TVM_FFI_ICHECK_EQ(C.dtype().code, kDLFloat) << "C must be float type";
    TVM_FFI_ICHECK_EQ(C.dtype().bits, 16) << "C must be float16";

    TVM_FFI_ICHECK_EQ(A.device().device_type, kDLCUDA) << "A must be on CUDA";
    TVM_FFI_ICHECK_EQ(B.device().device_type, kDLCUDA) << "B must be on CUDA";
    TVM_FFI_ICHECK_EQ(C.device().device_type, kDLCUDA) << "C must be on CUDA";

    const __half* A_data = static_cast<const __half*>(A.data_ptr());
    const __half* B_data = static_cast<const __half*>(B.data_ptr());
    __half* C_data = static_cast<__half*>(C.data_ptr());

    DLDevice dev = A.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    gemm_n4096_k4096_launch(A_data, B_data, C_data, static_cast<int>(M), stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
```

## Output Format

Write the complete JSON solution to:
**`Example-FlashInfer-Trace/solutions/agent_vibecode_gemm.json`**

The JSON must be valid and contain:
- All required schema fields
- Complete source code for all 3 files in the `content` fields
- Properly escaped strings (use JSON encoding)

## Validation Checklist

Before finalizing, verify:
- [ ] File names are exactly: `kernel.h`, `kernel.cu`, `main.cpp`
- [ ] Entry point is `"main.cpp::run"`
- [ ] Function signature: `void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C)`
- [ ] TVM_FFI_DLL_EXPORT_TYPED_FUNC exposes the `run` function
- [ ] All three files included in `sources` array
- [ ] Input validation with `TVM_FFI_ICHECK_*` macros
- [ ] Kernel implements `C = A @ B.T` (transpose of B)
- [ ] Data type is `__half` (float16)
- [ ] CUDA stream from `TVMFFIEnvGetStream()`
- [ ] Checks that all tensors are on CUDA device
- [ ] JSON is valid and properly formatted
- [ ] All TVM FFI headers included correctly
