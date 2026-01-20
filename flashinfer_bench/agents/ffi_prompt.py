"""Prompt templates for TVM FFI API documentation used by agents."""

FFI_PROMPT_SIMPLE = """
Use TVM FFI format for your generated kernel host function and bindings

# TVM FFI API Documentation

## 1. TensorView (tvm/ffi/container/tensor.h)

Non-owning view of tensor data. Use `tvm::ffi::TensorView` for function parameters.

### Essential Methods
```cpp
void* data_ptr() const               // Raw data pointer
DLDevice device() const              // Device info (.device_type, .device_id)
DLDataType dtype() const             // Data type (.code, .bits, .lanes)
int32_t ndim() const                 // Number of dimensions
int64_t size(int64_t idx) const      // Size of dimension idx (negative idx: count from end)
int64_t numel() const                // Total number of elements
bool IsContiguous() const            // Check if memory is contiguous
```

### ShapeView Access
```cpp
ShapeView shape() const              // Get shape (array-like)
ShapeView strides() const            // Get strides (array-like)

// ShapeView can be indexed like an array:
// tensor.shape()[0], tensor.shape()[1], etc.
```

### Device Type Constants (DLDevice.device_type)
```cpp
kDLCPU = 1          // CPU
kDLCUDA = 2         // CUDA GPU
kDLCUDAHost = 3     // CUDA pinned memory
kDLROCM = 10        // ROCm/HIP
```

### Data Type Constants (DLDataType)
```cpp
// DLDataType has: uint8_t code, uint8_t bits, uint16_t lanes
// Common .code values:
kDLInt = 0          // Signed integer
kDLUInt = 1         // Unsigned integer
kDLFloat = 2        // IEEE floating point
kDLBfloat = 4       // Brain floating point

// Example: float32 has code=2 (kDLFloat), bits=32, lanes=1
// Example: half/fp16 has code=2, bits=16, lanes=1
// Example: float8_e4m3 has code=8, bits=8, lanes=1 (packed in memory)
```

## 2. Function API (tvm/ffi/function.h)

### Export Macro
```cpp
// Use this to export your C++ function for FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(export_name, cpp_function)

// Example:
void MyKernel(tvm::ffi::TensorView a, tvm::ffi::TensorView b) { ... }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, MyKernel);
```

**Supported function signatures:**
- `void func(TensorView t1, TensorView t2, ...)`
- `void func(TensorView t, int64_t size, float alpha, ...)`
- `int64_t func(TensorView t)`
- Any combination of: `TensorView`, `int32_t`, `int64_t`, `float`, `double`, `bool`, `std::string`

## 3. Error Handling (tvm/ffi/error.h)

### Throwing Errors
```cpp
// Throw with custom error kind
TVM_FFI_THROW(ValueError) << "Invalid input: " << x;
TVM_FFI_THROW(RuntimeError) << "CUDA error: " << cudaGetErrorString(err);
TVM_FFI_THROW(TypeError) << "Expected float32, got int32";
```

### Assertions (for internal logic errors)
```cpp
TVM_FFI_ICHECK(condition) << "message"           // General check
TVM_FFI_ICHECK_EQ(x, y) << "x must equal y"      // x == y
TVM_FFI_ICHECK_NE(x, y) << "x must not equal y"  // x != y
TVM_FFI_ICHECK_LT(x, y) << "x must be less than y"     // x < y
TVM_FFI_ICHECK_LE(x, y) << "x must be at most y"       // x <= y
TVM_FFI_ICHECK_GT(x, y) << "x must be greater than y"  // x > y
TVM_FFI_ICHECK_GE(x, y) << "x must be at least y"      // x >= y
```

### User Input Validation (use TVM_FFI_THROW instead)
```cpp
// For user-facing errors, use TVM_FFI_THROW with appropriate error kind:
if (x.ndim() != 2) {
  TVM_FFI_THROW(ValueError) << "Expected 2D tensor, got " << x.ndim() << "D";
}
if (x.dtype().code != kDLFloat || x.dtype().bits != 32) {
  TVM_FFI_THROW(TypeError) << "Expected float32 dtype";
}
```

## 4. CUDA Stream Management (tvm/ffi/extra/c_env_api.h)

```cpp
// Get the current CUDA stream for a device
TVMFFIStreamHandle TVMFFIEnvGetStream(int32_t device_type, int32_t device_id)

// Usage:
DLDevice dev = tensor.device();
cudaStream_t stream = static_cast<cudaStream_t>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));

// Launch kernel on the stream:
my_kernel<<<blocks, threads, 0, stream>>>(...);
```

# Example: CUDA Kernel Binding
```cpp
// File: add_one_cuda.cu
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

namespace my_kernels {

__global__ void AddOneKernel(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] + 1.0f;
  }
}

void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Input validation
  TVM_FFI_ICHECK_EQ(x.ndim(), 1) << "x must be 1D";
  TVM_FFI_ICHECK_EQ(y.ndim(), 1) << "y must be 1D";
  TVM_FFI_ICHECK_EQ(x.size(0), y.size(0)) << "Shape mismatch";

  // Get data pointers
  float* x_data = static_cast<float*>(x.data_ptr());
  float* y_data = static_cast<float*>(y.data_ptr());
  int64_t n = x.size(0);

  // Get CUDA stream from environment
  DLDevice dev = x.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  // Launch kernel
  int64_t threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  AddOneKernel<<<blocks, threads, 0, stream>>>(x_data, y_data, n);
}

// Export the function with name "add_one_cuda"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cuda, AddOne);

}  // namespace my_kernels
```
"""
"""Simplified TVM FFI API documentation with essential methods and a basic example."""

FFI_PROMPT = """
Use TVM FFI format for your generated kernel host function and bindings

# TVM FFI API Documentation

## 1. Tensor API (tvm/ffi/container/tensor.h)

### Tensor Class
A managed n-dimensional array with reference counting.

**Methods:**
- `void* data_ptr() const` - Returns raw data pointer (already accounts for byte_offset, don't add byte_offset manually)
- `DLDevice device() const` - Returns device info (`.device_type`, `.device_id`)
- `int32_t ndim() const` - Returns number of dimensions
- `DLDataType dtype() const` - Returns data type (`.code`, `.bits`, `.lanes`)
- `ShapeView shape() const` - Returns shape array (indexable: `shape()[0]`, `shape()[1]`, ...)
- `ShapeView strides() const` - Returns strides array
- `int64_t size(int64_t idx) const` - Returns size of dimension at idx (negative idx counts from end)
- `int64_t stride(int64_t idx) const` - Returns stride of dimension at idx
- `int64_t numel() const` - Returns total number of elements
- `uint64_t byte_offset() const` - Returns byte offset
- `bool IsContiguous() const` - Checks if tensor memory is contiguous
- `bool IsAligned(size_t alignment) const` - Checks if data is aligned to given bytes

**Static Factory Methods:**
- `static Tensor FromDLPack(DLManagedTensor* tensor, size_t require_alignment = 0, bool require_contiguous = false)`
- `static Tensor FromDLPackVersioned(DLManagedTensorVersioned* tensor, size_t require_alignment = 0, bool require_contiguous = false)`
- `template<typename TNDAlloc, typename... ExtraArgs> static Tensor FromNDAlloc(TNDAlloc alloc, ffi::ShapeView shape, DLDataType dtype, DLDevice device, ExtraArgs&&... extra_args)`
- `static Tensor FromEnvAlloc(int (*env_alloc)(DLTensor*, TVMFFIObjectHandle*), ffi::ShapeView shape, DLDataType dtype, DLDevice device)`

**Conversion Methods:**
- `DLManagedTensor* ToDLPack() const` - Convert to DLPack managed tensor
- `DLManagedTensorVersioned* ToDLPackVersioned() const` - Convert to versioned DLPack
- `const DLTensor* GetDLTensorPtr() const` - Get underlying DLTensor pointer

### TensorView Class
Non-owning lightweight view of a Tensor. Kernel entrypoints should use `tvm::ffi::TensorView` (or `const TensorView&`) for tensor inputs/outputs.

**Constructors:**
- `TensorView(const Tensor& tensor)` - From Tensor
- `TensorView(const DLTensor* tensor)` - From DLTensor pointer

**Methods (same interface as Tensor):**
- `void* data_ptr() const`
- `DLDevice device() const`
- `int32_t ndim() const`
- `DLDataType dtype() const`
- `ShapeView shape() const`
- `ShapeView strides() const`
- `int64_t size(int64_t idx) const`
- `int64_t stride(int64_t idx) const`
- `int64_t numel() const`
- `uint64_t byte_offset() const`
- `bool IsContiguous() const`

### Utility Functions
- `bool IsContiguous(const DLTensor& arr)` - Check if DLTensor is contiguous
- `bool IsDirectAddressDevice(const DLDevice& device)` - Check if device uses direct addressing
- `size_t GetDataSize(size_t numel, DLDataType dtype)` - Calculate bytes for packed data
- `size_t GetDataSize(const DLTensor& arr)` - Calculate bytes in DLTensor
- `size_t GetDataSize(const Tensor& tensor)` - Calculate bytes in Tensor
- `size_t GetDataSize(const TensorView& tensor)` - Calculate bytes in TensorView

### Device Type Constants (DLDevice.device_type)
```cpp
kDLCPU = 1          // CPU
kDLCUDA = 2         // CUDA GPU
kDLCUDAHost = 3     // CUDA pinned memory
kDLROCM = 10        // ROCm/HIP
```

### Data Type Constants (DLDataType)
```cpp
// DLDataType has: uint8_t code, uint8_t bits, uint16_t lanes
kDLInt = 0          // Signed integer
kDLUInt = 1         // Unsigned integer
kDLFloat = 2        // IEEE floating point
kDLBfloat = 4       // Brain floating point

// Example: float32 has code=2 (kDLFloat), bits=32, lanes=1
// Example: half/fp16 has code=2, bits=16, lanes=1
// Example: float8_e4m3 has code=8, bits=8, lanes=1 (packed in memory)
```

## 2. Function API (tvm/ffi/function.h)

### Function Class
Type-erased callable object.

**Constructors:**
- `Function(std::nullptr_t)` - Null constructor
- `template<typename TCallable> Function(TCallable packed_call)` - From callable (legacy)

**Static Factory Methods:**
- `template<typename TCallable> static Function FromPacked(TCallable packed_call)` - From packed signature: `void(const AnyView*, int32_t, Any*)` or `void(PackedArgs, Any*)`
- `template<typename TCallable> static Function FromTyped(TCallable callable)` - From typed C++ function
- `template<typename TCallable> static Function FromTyped(TCallable callable, std::string name)` - With name for error messages
- `static Function FromExternC(void* self, TVMFFISafeCallType safe_call, void (*deleter)(void* self))` - From C callback

**Global Registry:**
- `static std::optional<Function> GetGlobal(std::string_view name)`
- `static std::optional<Function> GetGlobal(const std::string& name)`
- `static std::optional<Function> GetGlobal(const String& name)`
- `static std::optional<Function> GetGlobal(const char* name)`
- `static Function GetGlobalRequired(std::string_view name)` - Throws if not found
- `static Function GetGlobalRequired(const std::string& name)`
- `static Function GetGlobalRequired(const String& name)`
- `static Function GetGlobalRequired(const char* name)`
- `static void SetGlobal(std::string_view name, Function func, bool override = false)`
- `static std::vector<String> ListGlobalNames()`
- `static void RemoveGlobal(const String& name)`

**Invocation:**
- `template<typename... Args> Any operator()(Args&&... args) const` - Call with unpacked args
- `void CallPacked(const AnyView* args, int32_t num_args, Any* result) const`
- `void CallPacked(PackedArgs args, Any* result) const`
- `template<typename... Args> static Any InvokeExternC(void* handle, TVMFFISafeCallType safe_call, Args&&... args)`

### TypedFunction<R(Args...)> Class
Type-safe wrapper around Function.

**Constructors:**
- `TypedFunction()` - Default
- `TypedFunction(std::nullptr_t)`
- `TypedFunction(Function packed)` - From Function
- `template<typename FLambda> TypedFunction(FLambda typed_lambda)` - From lambda
- `template<typename FLambda> TypedFunction(FLambda typed_lambda, std::string name)` - With name

**Methods:**
- `R operator()(Args... args) const` - Type-safe invocation
- `operator Function() const` - Convert to Function
- `const Function& packed() const&` - Get internal Function
- `Function&& packed() &&` - Move internal Function
- `static std::string TypeSchema()` - Get JSON type schema

### PackedArgs Class
Represents packed arguments.

**Constructor:**
- `PackedArgs(const AnyView* data, int32_t size)`

**Methods:**
- `int size() const` - Number of arguments
- `const AnyView* data() const` - Raw argument array
- `PackedArgs Slice(int begin, int end = -1) const` - Get subset
- `AnyView operator[](int i) const` - Access argument
- `template<typename... Args> static void Fill(AnyView* data, Args&&... args)` - Pack arguments

### Export Macro
```cpp
// Export typed C++ function for FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(export_name, cpp_function)

// Example:
void MyKernel(tvm::ffi::TensorView a, tvm::ffi::TensorView b) { ... }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, MyKernel);
```

**Supported function signatures:**
- `void func(TensorView t1, TensorView t2, ...)`
- `void func(TensorView t, int64_t size, float alpha, ...)`
- `int64_t func(TensorView t)`
- Any combination of: `TensorView`, `int32_t`, `int64_t`, `float`, `double`, `bool`, `std::string`

## 3. Error Handling (tvm/ffi/error.h)

### Error Class
Exception object with stack trace.

**Constructor:**
- `Error(std::string kind, std::string message, std::string backtrace)`
- `Error(std::string kind, std::string message, const TVMFFIByteArray* backtrace)`

**Methods:**
- `std::string kind() const` - Error category
- `std::string message() const` - Error description
- `std::string backtrace() const` - Stack trace
- `std::string TracebackMostRecentCallLast() const` - Python-style traceback
- `const char* what() const noexcept override` - Standard exception interface
- `void UpdateBacktrace(const TVMFFIByteArray* backtrace_str, int32_t update_mode)` - Modify traceback

### EnvErrorAlreadySet Exception
Thrown when error exists in frontend environment (e.g., Python interrupt).

**Usage:**
```cpp
void LongRunningFunction() {
  if (TVMFFIEnvCheckSignals() != 0) {
    throw ::tvm::ffi::EnvErrorAlreadySet();
  }
  // do work here
}
```

### Error Macros
```cpp
// Throw with backtrace
TVM_FFI_THROW(ErrorKind) << message

// Log to stderr and throw (for startup functions)
TVM_FFI_LOG_AND_THROW(ErrorKind) << message

// Check C function return code
TVM_FFI_CHECK_SAFE_CALL(func)

// Wrap C++ code for C API
TVM_FFI_SAFE_CALL_BEGIN();
// c++ code region here
TVM_FFI_SAFE_CALL_END();
```

### Assertion Macros
```cpp
TVM_FFI_ICHECK(condition) << "message"           // General check
TVM_FFI_CHECK(condition, ErrorKind) << "message" // Custom error type
TVM_FFI_ICHECK_EQ(x, y) << "x must equal y"      // x == y
TVM_FFI_ICHECK_NE(x, y) << "x must not equal y"  // x != y
TVM_FFI_ICHECK_LT(x, y) << "x must be less than y"     // x < y
TVM_FFI_ICHECK_LE(x, y) << "x must be at most y"       // x <= y
TVM_FFI_ICHECK_GT(x, y) << "x must be greater than y"  // x > y
TVM_FFI_ICHECK_GE(x, y) << "x must be at least y"      // x >= y
TVM_FFI_ICHECK_NOTNULL(ptr) << "ptr must not be null"  // ptr != nullptr
```

**Common Error Kinds:**
- `ValueError` - Invalid argument values
- `TypeError` - Type mismatch
- `RuntimeError` - Runtime failures (CUDA errors, etc.)
- `InternalError` - Internal logic errors

### Utility Functions
- `int32_t TypeKeyToIndex(std::string_view type_key)` - Get type index from key

## 4. Environment APIs (tvm/ffi/extra/c_env_api.h)

### Stream Management
```cpp
// Set current stream for a device
int TVMFFIEnvSetStream(int32_t device_type, int32_t device_id,
                       TVMFFIStreamHandle stream,
                       TVMFFIStreamHandle* opt_out_original_stream)

// Get current stream for a device
TVMFFIStreamHandle TVMFFIEnvGetStream(int32_t device_type, int32_t device_id)

// Usage example:
DLDevice dev = tensor.device();
cudaStream_t stream = static_cast<cudaStream_t>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));
```

### Tensor Allocation
```cpp
// Set DLPack allocator (TLS or global)
int TVMFFIEnvSetDLPackManagedTensorAllocator(
    DLPackManagedTensorAllocator allocator,
    int write_to_global_context,
    DLPackManagedTensorAllocator* opt_out_original_allocator)

// Get current allocator
DLPackManagedTensorAllocator TVMFFIEnvGetDLPackManagedTensorAllocator()

// Allocate tensor using environment allocator
int TVMFFIEnvTensorAlloc(DLTensor* prototype, TVMFFIObjectHandle* out)

// Usage with FromEnvAlloc:
ffi::Tensor tensor = ffi::Tensor::FromEnvAlloc(
    TVMFFIEnvTensorAlloc, shape, dtype, device);
```

### Module & Symbol Management
```cpp
// Lookup function from module imports
int TVMFFIEnvModLookupFromImports(TVMFFIObjectHandle library_ctx,
                                  const char* func_name,
                                  TVMFFIObjectHandle* out)

// Register context symbol (available when library is loaded)
int TVMFFIEnvModRegisterContextSymbol(const char* name, void* symbol)

// Register system library symbol
int TVMFFIEnvModRegisterSystemLibSymbol(const char* name, void* symbol)

// Register C API symbol
int TVMFFIEnvRegisterCAPI(const char* name, void* symbol)
```

### Utilities
```cpp
// Check for environment signals (e.g., Python Ctrl+C)
int TVMFFIEnvCheckSignals()
```

### Type Definitions
```cpp
typedef void* TVMFFIStreamHandle  // Stream handle type
```

# Common Use Cases

## Dtype Validation
```cpp
void MyKernel(tvm::ffi::TensorView x) {
  DLDataType dt = x.dtype();
  TVM_FFI_ICHECK_EQ(dt.code, kDLFloat) << "Expected float dtype";
  TVM_FFI_ICHECK_EQ(dt.bits, 32) << "Expected 32-bit dtype";

  // Or for user-facing errors:
  if (dt.code != kDLFloat || dt.bits != 32) {
    TVM_FFI_THROW(TypeError) << "Expected float32, got "
                             << "code=" << dt.code << " bits=" << dt.bits;
  }
}
```

## Shape Validation
```cpp
void MatMul(tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView c) {
  TVM_FFI_ICHECK_EQ(a.ndim(), 2) << "a must be 2D";
  TVM_FFI_ICHECK_EQ(b.ndim(), 2) << "b must be 2D";
  TVM_FFI_ICHECK_EQ(a.size(1), b.size(0)) << "Shape mismatch for matmul";
  TVM_FFI_ICHECK_EQ(c.size(0), a.size(0)) << "Output shape mismatch";
  TVM_FFI_ICHECK_EQ(c.size(1), b.size(1)) << "Output shape mismatch";

  TVM_FFI_ICHECK_EQ(a.device().device_type, b.device().device_type);
  TVM_FFI_ICHECK_EQ(a.device().device_id, b.device().device_id);
  TVM_FFI_ICHECK_EQ(a.device().device_type, c.device().device_type);
  TVM_FFI_ICHECK_EQ(a.device().device_id, c.device().device_id);
}
```

## Multi-dimensional Indexing
```cpp
void Process2D(tvm::ffi::TensorView x) {
  int64_t height = x.size(0);
  int64_t width = x.size(1);
  float* data = static_cast<float*>(x.data_ptr());

  // If contiguous (row-major), access as: data[i * width + j]
  TVM_FFI_ICHECK(x.IsContiguous()) << "Expected contiguous tensor";

  // With strides:
  int64_t stride0 = x.stride(0);
  int64_t stride1 = x.stride(1);
  // Access: data[i * stride0 + j * stride1]
}
```

## Device-specific Kernels
```cpp
void MyKernel(tvm::ffi::TensorView x) {
  DLDevice dev = x.device();
  if (dev.device_type == kDLCUDA) {
    // Launch CUDA kernel
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));
    // kernel<<<blocks, threads, 0, stream>>>(...);
  } else if (dev.device_type == kDLCPU) {
    // CPU implementation
  } else {
    TVM_FFI_THROW(RuntimeError) << "Unsupported device type: " << dev.device_type;
  }
}
```

## Allocating New Tensors
```cpp
void CreateOutput(tvm::ffi::TensorView input) {
  // Create output tensor with same device and dtype as input
  ffi::ShapeView shape = input.shape();
  DLDataType dtype = input.dtype();
  DLDevice device = input.device();

  // Use environment allocator
  ffi::Tensor output = ffi::Tensor::FromEnvAlloc(
      TVMFFIEnvTensorAlloc, shape, dtype, device);
}
```

# Example: CUDA Kernel Binding

```cpp
// File: add_one_cuda.cu
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

namespace my_kernels {

__global__ void AddOneKernel(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] + 1.0f;
  }
}

void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Input validation
  TVM_FFI_ICHECK_EQ(x.ndim(), 1) << "x must be 1D";
  TVM_FFI_ICHECK_EQ(y.ndim(), 1) << "y must be 1D";
  TVM_FFI_ICHECK_EQ(x.size(0), y.size(0)) << "Shape mismatch";

  // Get data pointers
  float* x_data = static_cast<float*>(x.data_ptr());
  float* y_data = static_cast<float*>(y.data_ptr());
  int64_t n = x.size(0);

  // Get CUDA stream from environment
  DLDevice dev = x.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  // Launch kernel
  int64_t threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  AddOneKernel<<<blocks, threads, 0, stream>>>(x_data, y_data, n);
}

// Export the function with name "add_one_cuda"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cuda, AddOne);

}  // namespace my_kernels
```
"""
"""Comprehensive TVM FFI API documentation with full method signatures and multiple examples."""
