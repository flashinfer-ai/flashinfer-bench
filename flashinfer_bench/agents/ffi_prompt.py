FFI_PROMPT_SIMPLE = """
Use TVM FFI format for your generated kernel host function and bindings

Use the following headers:
#include <tvm/ffi/container/tensor.h>   // TensorView: tensor arguments
#include <tvm/ffi/function.h>           // TVM_FFI_DLL_EXPORT_TYPED_FUNC
#include <tvm/ffi/error.h>              // TVM_FFI_ICHECK, TVM_FFI_THROW

Include when using CUDA / env-managed streams or allocators:
#include <tvm/ffi/extra/c_env_api.h>    // TVMFFIEnvGetStream, TVMFFIEnvTensorAlloc

# TVM FFI API Reference

## TensorView (tvm/ffi/container/tensor.h)

**Purpose:** Non-owning view of tensor data. Use `tvm::ffi::TensorView` for function parameters.

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
```

## Function Export (tvm/ffi/function.h)

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

## Error Handling (tvm/ffi/error.h)

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

## CUDA Stream Management (tvm/ffi/extra/c_env_api.h)

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

Example Usage of FFI binding:
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

FFI_FULL_PROMPT = """
# TVM FFI Complete API Reference

This guide provides comprehensive documentation for writing CUDA kernels with TVM FFI bindings.

## Required Headers

```cpp
#include <tvm/ffi/container/tensor.h>   // Tensor, TensorView
#include <tvm/ffi/container/array.h>    // Array<T>
#include <tvm/ffi/container/map.h>      // Map<K, V>
#include <tvm/ffi/container/tuple.h>    // Tuple<...>
#include <tvm/ffi/container/shape.h>    // Shape, ShapeView
#include <tvm/ffi/container/variant.h>  // Variant<...>
#include <tvm/ffi/string.h>             // String
#include <tvm/ffi/dtype.h>              // DLDataType utilities
#include <tvm/ffi/function.h>           // Function export macros
#include <tvm/ffi/error.h>              // Error handling
#include <tvm/ffi/extra/c_env_api.h>    // CUDA stream management
#include <tvm/ffi/extra/module.h>       // Module loading
```

---

# Part 1: Data Container Types

## 1.1 Tensor & TensorView (tvm/ffi/container/tensor.h)

### TensorView - Lightweight Non-Owning View

**Purpose:** Non-owning view of tensor data. Always use `tvm::ffi::TensorView` for function parameters.

**Constructors:**
```cpp
TensorView(const Tensor& tensor)          // From Tensor
TensorView(const DLTensor* tensor)        // From DLTensor pointer
```

**Essential Methods:**
```cpp
// Data access
void* data_ptr() const                    // Raw data pointer
int64_t byte_offset() const               // Byte offset from base pointer

// Device information
DLDevice device() const                   // Returns device (.device_type, .device_id)

// Type information
DLDataType dtype() const                  // Returns data type (.code, .bits, .lanes)

// Dimension information
int32_t ndim() const                      // Number of dimensions
int64_t size(int64_t idx) const          // Size of dimension idx (supports negative indexing)
int64_t stride(int64_t idx) const        // Stride of dimension idx (supports negative indexing)
int64_t numel() const                    // Total number of elements

// Shape and strides access
ShapeView shape() const                   // Get shape (array-like, indexable: shape()[0], shape()[1], ...)
ShapeView strides() const                 // Get strides (array-like)

// Memory properties
bool IsContiguous() const                 // Check if memory layout is contiguous
bool IsAligned(size_t alignment) const   // Check if data pointer is aligned
```

**Usage Example:**
```cpp
void MyKernel(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
  // Get dimensions
  int64_t batch = input.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);

  // Get data pointer
  float* in_data = static_cast<float*>(input.data_ptr());
  float* out_data = static_cast<float*>(output.data_ptr());

  // Check device
  DLDevice dev = input.device();
  if (dev.device_type == kDLCUDA) {
    // CUDA-specific code
  }
}
```

### Tensor - Managed Owning Container

**Purpose:** Reference-counted tensor with memory ownership. Use when you need to allocate or store tensors.

**All TensorView methods** (Tensor has same interface as TensorView)

**Static Factory Methods:**
```cpp
// From DLPack
static Tensor FromDLPack(DLManagedTensor* tensor,
                         size_t require_alignment = 0,
                         bool require_contiguous = false)

static Tensor FromDLPackVersioned(DLManagedTensorVersioned* tensor,
                                   size_t require_alignment = 0,
                                   bool require_contiguous = false)

// Allocate from environment (recommended for kernel outputs)
static Tensor FromEnvAlloc(int (*env_alloc)(DLTensor*, TVMFFIObjectHandle*),
                          ffi::ShapeView shape,
                          DLDataType dtype,
                          DLDevice device)

// Example: Allocate tensor using environment allocator
ffi::Tensor output = ffi::Tensor::FromEnvAlloc(
    TVMFFIEnvTensorAlloc, shape, dtype, device);
```

**Conversion Methods:**
```cpp
DLManagedTensor* ToDLPack() const                 // Export to DLPack
DLManagedTensorVersioned* ToDLPackVersioned() const
const DLTensor* GetDLTensorPtr() const            // Get underlying DLTensor
```

### Utility Functions
```cpp
// Check if tensor is contiguous
bool IsContiguous(const DLTensor& arr)

// Check alignment
bool IsAligned(const DLTensor& arr, size_t alignment)

// Check if device supports direct addressing (CPU, CUDA, etc.)
bool IsDirectAddressDevice(const DLDevice& device)

// Calculate data size in bytes
size_t GetDataSize(size_t numel, DLDataType dtype)
size_t GetDataSize(const DLTensor& arr)
size_t GetDataSize(const Tensor& tensor)
size_t GetDataSize(const TensorView& tensor)
```

### Device Type Constants
```cpp
kDLCPU = 1          // CPU
kDLCUDA = 2         // CUDA GPU
kDLCUDAHost = 3     // CUDA pinned memory
kDLROCM = 10        // ROCm/HIP
kDLMetal = 8        // Metal (Apple)
kDLVulkan = 7       // Vulkan
```

### Data Type Constants (DLDataType)
```cpp
// DLDataType structure: { uint8_t code, uint8_t bits, uint16_t lanes }

// Type codes (code field)
kDLInt = 0          // Signed integer
kDLUInt = 1         // Unsigned integer
kDLFloat = 2        // IEEE floating point
kDLBfloat = 4       // Brain floating point
kDLBool = 5         // Boolean
kDLFloat8_e4m3fn = 6    // FP8 E4M3
kDLFloat8_e5m2 = 7      // FP8 E5M2
// ... and more FP8 variants

// Common data types (examples):
// float32:  code=kDLFloat, bits=32, lanes=1
// float16:  code=kDLFloat, bits=16, lanes=1
// int32:    code=kDLInt,   bits=32, lanes=1
// uint8:    code=kDLUInt,  bits=8,  lanes=1
// bfloat16: code=kDLBfloat, bits=16, lanes=1
```

---

## 1.2 Array<T> (tvm/ffi/container/array.h)

**Purpose:** Dynamic array container with copy-on-write semantics. Similar to `std::vector` but FFI-compatible.

**Type Parameter:** `T` must be compatible with `tvm::ffi::Any` (ObjectRef types, primitives via TypeTraits)

**Constructors:**
```cpp
Array<T>()                                      // Empty array
Array<T>(size_t n, const T& val)               // n copies of val
Array<T>(std::initializer_list<T> init)        // From initializer list
Array<T>(const std::vector<T>& vec)            // From std::vector
Array<T>(IterType first, IterType last)        // From iterator range
```

**Element Access:**
```cpp
T operator[](int64_t i) const                   // Read i-th element
T front() const                                 // First element
T back() const                                  // Last element
T at(int64_t i) const                          // Bounds-checked access
```

**Capacity:**
```cpp
size_t size() const                             // Number of elements
size_t capacity() const                         // Allocated capacity
bool empty() const                              // Check if empty
void reserve(int64_t n)                         // Reserve capacity
void resize(int64_t n)                          // Resize array
```

**Modifiers (Copy-on-Write):**
```cpp
void push_back(const T& item)                   // Add element to end
void emplace_back(Args&&... args)               // Construct element in-place
void pop_back()                                 // Remove last element
void insert(iterator pos, const T& val)         // Insert at position
void erase(iterator pos)                        // Remove at position
void erase(iterator first, iterator last)       // Remove range
void clear()                                    // Remove all elements
void Set(int64_t i, T value)                   // Set i-th element
```

**Iterators:**
```cpp
iterator begin() const                          // Begin iterator
iterator end() const                            // End iterator
reverse_iterator rbegin() const                 // Reverse begin
reverse_iterator rend() const                   // Reverse end
```

**Functional Operations:**
```cpp
// Map function over array
template<typename F>
Array<U> Map(F fmap) const                      // Returns Array<U>

// Mutate array in-place (if unique owner)
template<typename F>
void MutateByApply(F fmutate)                   // F: T -> T
```

**Static Methods:**
```cpp
// Aggregate multiple arrays/elements
template<typename... Args>
static Array<T> Agregate(Args... args)          // Combine T and Array<T> args
```

**Usage Example:**
```cpp
// Create array
Array<int64_t> dims = {1, 3, 224, 224};

// Access elements
int64_t batch = dims[0];

// Modify (copy-on-write)
dims.push_back(512);
dims.Set(0, 2);

// Map operation
Array<int64_t> doubled = dims.Map([](int64_t x) { return x * 2; });

// Iterate
for (int64_t d : dims) {
  // process d
}
```

---

## 1.3 Map<K, V> (tvm/ffi/container/map.h)

**Purpose:** Hash map container with copy-on-write semantics. Similar to `std::unordered_map`.

**Type Parameters:**
- `K` (key type) must be hashable and compatible with Any
- `V` (value type) must be compatible with Any

**Constructors:**
```cpp
Map<K, V>()                                     // Empty map
Map<K, V>(std::initializer_list<std::pair<K, V>> init)
Map<K, V>(IterType first, IterType last)       // From iterator range
Map<K, V>(const std::unordered_map<K, V>& map) // From std::unordered_map
```

**Element Access:**
```cpp
V at(const K& key) const                        // Throws if key not found
V operator[](const K& key) const                // Throws if key not found
Optional<V> Get(const K& key) const            // Returns Optional (nullopt if not found)
V Get(const K& key, const V& default_value) const // Returns default if not found
```

**Capacity:**
```cpp
size_t size() const                             // Number of elements
size_t count(const K& key) const               // 0 or 1 (check if key exists)
bool empty() const                              // Check if empty
```

**Modifiers (Copy-on-Write):**
```cpp
void Set(const K& key, const V& value)          // Insert or update
void erase(const K& key)                        // Remove key
void clear()                                    // Remove all elements
```

**Iterators:**
```cpp
iterator begin() const                          // Begin iterator
iterator end() const                            // End iterator

// Iterator dereference returns std::pair<K, V>
for (auto kv : map) {
  K key = kv.first;
  V value = kv.second;
}
```

**Static Methods:**
```cpp
// Construct map from keys and values
static Map<K, V> FromItems(const Array<std::pair<K, V>>& items)
```

**Usage Example:**
```cpp
// Create map
Map<String, int64_t> config = {
  {"batch_size", 32},
  {"num_layers", 12}
};

// Access
int64_t batch = config["batch_size"];
Optional<int64_t> opt = config.Get("hidden_dim");
if (opt.defined()) {
  int64_t hidden = opt.value();
}

// Modify
config.Set("hidden_dim", 768);
config.erase("num_layers");

// Iterate
for (auto kv : config) {
  String key = kv.first;
  int64_t value = kv.second;
}
```

---

## 1.4 Tuple<T...> (tvm/ffi/container/tuple.h)

**Purpose:** Fixed-size heterogeneous container. Similar to `std::tuple` but FFI-compatible.

**Constructors:**
```cpp
Tuple<T1, T2, ...>()                            // Default construct
Tuple<T1, T2, ...>(T1 v1, T2 v2, ...)          // From values
```

**Element Access:**
```cpp
// Compile-time index access
std::get<0>(tuple)                              // Get first element
std::get<1>(tuple)                              // Get second element

// Runtime index access (returns Any)
Any operator[](int64_t idx) const               // Get element at runtime index
```

**Capacity:**
```cpp
static constexpr size_t size()                  // Number of elements (compile-time)
```

**Structured Binding:**
```cpp
auto [a, b, c] = tuple;                         // C++17 structured binding
```

**Usage Example:**
```cpp
// Create tuple
Tuple<int64_t, float, String> result = {42, 3.14f, "success"};

// Access elements
int64_t code = std::get<0>(result);
float value = std::get<1>(result);
String msg = std::get<2>(result);

// Or with structured binding
auto [code, value, msg] = result;
```

---

## 1.5 Shape & ShapeView (tvm/ffi/container/shape.h)

### ShapeView - Lightweight Non-Owning View

**Purpose:** Non-owning view of shape dimensions. Use for passing shapes without allocation.

**Constructors:**
```cpp
ShapeView()                                     // Empty
ShapeView(const int64_t* data, size_t size)    // From pointer and size
ShapeView(std::initializer_list<int64_t> init) // From initializer list
```

**Element Access:**
```cpp
int64_t operator[](size_t idx) const           // Access dimension
int64_t at(size_t idx) const                   // Bounds-checked access
int64_t front() const                          // First dimension
int64_t back() const                           // Last dimension
```

**Properties:**
```cpp
const int64_t* data() const                    // Data pointer
size_t size() const                            // Number of dimensions
bool empty() const                             // Check if empty
int64_t Product() const                        // Product of all dimensions
```

**Iterators:**
```cpp
const int64_t* begin() const
const int64_t* end() const
```

### Shape - Managed Owning Container

**Purpose:** Reference-counted shape with memory ownership.

**Constructors:**
```cpp
Shape()                                        // Empty
Shape(std::initializer_list<int64_t> init)
Shape(std::vector<int64_t> vec)
Shape(Array<int64_t> arr)
Shape(ShapeView view)
Shape(IterType first, IterType last)
```

**All ShapeView methods** (same interface)

**Static Factory:**
```cpp
// Create strides from shape (row-major)
static Shape StridesFromShape(ShapeView shape)
```

**Conversion:**
```cpp
operator ShapeView() const                     // Implicit conversion to ShapeView
```

**Usage Example:**
```cpp
// Create shape
Shape shape = {2, 3, 224, 224};  // NCHW

// Access
int64_t batch = shape[0];
int64_t channels = shape[1];

// Calculate total elements
int64_t numel = shape.Product();  // 2 * 3 * 224 * 224

// Create strides
Shape strides = Shape::StridesFromShape(shape);
// strides = {3*224*224, 224*224, 224, 1}
```

---

## 1.6 Variant<V...> (tvm/ffi/container/variant.h)

**Purpose:** Type-safe union that can hold one of several types. Similar to `std::variant`.

**Constructors:**
```cpp
Variant<T1, T2, ...>(T1 val)                   // Construct with T1
Variant<T1, T2, ...>(T2 val)                   // Construct with T2
```

**Type Checking and Casting:**
```cpp
// Try cast (returns std::optional)
template<typename T>
std::optional<T> as() const                     // Returns nullopt if wrong type

// Cast (throws on failure)
template<typename T>
T get() const&                                  // Copy value
T get() &&                                      // Move value

// Get type information
std::string GetTypeKey() const                  // Get type key string
```

**Usage Example:**
```cpp
using Value = Variant<int64_t, float, String>;

Value v1 = 42;
Value v2 = 3.14f;
Value v3 = String("hello");

// Check and extract
if (auto opt_int = v1.as<int64_t>()) {
  int64_t val = *opt_int;
}

// Or direct cast (throws if wrong type)
int64_t val = v1.get<int64_t>();
```

---

## 1.7 String (tvm/ffi/string.h)

**Purpose:** FFI-compatible string with small-string optimization.

**Constructors:**
```cpp
String()                                       // Empty string
String(const char* str)
String(const std::string& str)
String(std::string&& str)                      // Move from std::string
String(const char* data, size_t size)
```

**Properties:**
```cpp
const char* data() const                       // Data pointer
const char* c_str() const                      // Null-terminated C string
size_t size() const                            // Length
size_t length() const                          // Length (same as size)
bool empty() const                             // Check if empty
```

**Element Access:**
```cpp
char at(size_t pos) const                      // Bounds-checked access
```

**Comparison:**
```cpp
int compare(const String& other) const
int compare(const std::string& other) const
int compare(const char* other) const

// Operators: ==, !=, <, >, <=, >=
```

**Concatenation:**
```cpp
String operator+(const String& lhs, const String& rhs)
String operator+(const String& lhs, const char* rhs)
// ... and other combinations
```

**Conversion:**
```cpp
operator std::string() const                   // Convert to std::string
```

**Utility:**
```cpp
String EscapeString(const String& value)       // Escape for JSON/C++
```

---

# Part 2: Data Type Utilities (tvm/ffi/dtype.h)

## DLDataType Structure

```cpp
struct DLDataType {
  uint8_t code;      // Type code (kDLInt, kDLFloat, etc.)
  uint8_t bits;      // Number of bits
  uint16_t lanes;    // Number of lanes (for vector types)
};
```

## Type Code Constants

```cpp
kDLInt = 0          // Signed integer
kDLUInt = 1         // Unsigned integer
kDLFloat = 2        // IEEE floating point
kDLOpaqueHandle = 3 // Opaque handle
kDLBfloat = 4       // Brain floating point
kDLBool = 5         // Boolean
kDLFloat8_e4m3fn = 6
kDLFloat8_e5m2 = 7
// ... many more FP8 variants
```

## Conversion Functions

```cpp
// String to DLDataType
DLDataType StringToDLDataType(const String& str)
// Examples: "float32", "int64", "float16", "uint8"

// DLDataType to String
String DLDataTypeToString(DLDataType dtype)

// Get type code as string (internal)
const char* DLDataTypeCodeAsCStr(DLDataTypeCode type_code)
```

## Operators

```cpp
// Comparison
bool operator==(const DLDataType& lhs, const DLDataType& rhs)
bool operator!=(const DLDataType& lhs, const DLDataType& rhs)

// Output stream
std::ostream& operator<<(std::ostream& os, DLDataType dtype)
```

**Usage Example:**
```cpp
// Check dtype
DLDataType dt = tensor.dtype();
if (dt.code == kDLFloat && dt.bits == 32) {
  // float32 tensor
}

// Convert from string
DLDataType float16_dtype = StringToDLDataType("float16");

// Convert to string
String dtype_str = DLDataTypeToString(dt);  // "float32"
```

---

# Part 3: Error Handling (tvm/ffi/error.h)

## Error Class

```cpp
class Error : public std::exception {
public:
  Error(std::string kind, std::string message, std::string backtrace);

  std::string kind() const;                    // Error category
  std::string message() const;                 // Error message
  std::string backtrace() const;               // Stack trace
  std::string TracebackMostRecentCallLast() const; // Python-style traceback
  const char* what() const noexcept override;  // Standard exception interface
};
```

## Throwing Errors

```cpp
// Throw with automatic backtrace
TVM_FFI_THROW(ErrorKind) << "message" << variable << "more text"

// Common error kinds:
TVM_FFI_THROW(ValueError) << "Invalid value: " << x;
TVM_FFI_THROW(TypeError) << "Expected float32, got " << dtype;
TVM_FFI_THROW(RuntimeError) << "CUDA error: " << error_string;
TVM_FFI_THROW(InternalError) << "Unexpected condition";
TVM_FFI_THROW(IndexError) << "Index " << i << " out of bounds";

// Log to stderr and throw (for startup functions)
TVM_FFI_LOG_AND_THROW(ErrorKind) << "message";
```

## Assertions (Internal Logic Checks)

```cpp
// General check
TVM_FFI_ICHECK(condition) << "message"

// Comparison checks (more informative error messages)
TVM_FFI_ICHECK_EQ(x, y) << "x must equal y"              // x == y
TVM_FFI_ICHECK_NE(x, y) << "x must not equal y"          // x != y
TVM_FFI_ICHECK_LT(x, y) << "x must be less than y"       // x < y
TVM_FFI_ICHECK_LE(x, y) << "x must be at most y"         // x <= y
TVM_FFI_ICHECK_GT(x, y) << "x must be greater than y"    // x > y
TVM_FFI_ICHECK_GE(x, y) << "x must be at least y"        // x >= y
TVM_FFI_ICHECK_NOTNULL(ptr) << "ptr must not be null"    // ptr != nullptr

// Custom error type check
TVM_FFI_CHECK(condition, ErrorKind) << "message"
```

## Environment Error Handling

```cpp
// Check for interrupts (e.g., Python Ctrl+C)
int TVMFFIEnvCheckSignals()

// Exception for pre-existing errors in environment
throw tvm::ffi::EnvErrorAlreadySet();

// Usage in long-running functions
void LongRunningFunction() {
  if (TVMFFIEnvCheckSignals() != 0) {
    throw ::tvm::ffi::EnvErrorAlreadySet();
  }
  // ... do work
}
```

## Safe Call Wrappers (for C API)

```cpp
// Wrap C++ code for C API
TVM_FFI_SAFE_CALL_BEGIN();
// C++ code that may throw
TVM_FFI_SAFE_CALL_END();

// Check C function return codes
TVM_FFI_CHECK_SAFE_CALL(function_call);
```

**Usage Examples:**
```cpp
// User input validation
void MyKernel(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  if (x.ndim() != 2) {
    TVM_FFI_THROW(ValueError) << "Expected 2D tensor, got " << x.ndim() << "D";
  }

  DLDataType dt = x.dtype();
  if (dt.code != kDLFloat || dt.bits != 32) {
    TVM_FFI_THROW(TypeError) << "Expected float32, got " << DLDataTypeToString(dt);
  }

  // Internal consistency checks
  TVM_FFI_ICHECK_EQ(x.size(1), y.size(0)) << "Dimension mismatch for matmul";
}
```

---

# Part 4: Function Export (tvm/ffi/function.h)

## Export Macro

```cpp
// Export typed C++ function for FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(export_name, cpp_function)

// Example:
void MyKernel(tvm::ffi::TensorView a, tvm::ffi::TensorView b) {
  // kernel implementation
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, MyKernel);

// Or with lambda
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, [](int x) { return x + 1; });
```

## Supported Function Signatures

The macro supports automatic type marshalling for:

**Primitive Types:**
- `int32_t`, `int64_t`
- `uint32_t`, `uint64_t`
- `float`, `double`
- `bool`

**FFI Types:**
- `tvm::ffi::TensorView`, `tvm::ffi::Tensor`
- `tvm::ffi::String`, `std::string`, `const char*`
- `tvm::ffi::Array<T>`, `tvm::ffi::Map<K,V>`, `tvm::ffi::Tuple<...>`
- `tvm::ffi::Shape`, `tvm::ffi::ShapeView`
- `tvm::ffi::Any`, `tvm::ffi::Optional<T>`
- `DLDataType`, `DLDevice`

**Return Types:**
- `void` (no return)
- Any of the above types

**Example Signatures:**
```cpp
void func1(TensorView t1, TensorView t2)
int64_t func2(TensorView t)
Tensor func3(TensorView t, int64_t size, float alpha)
Array<int64_t> func4(Shape shape, String name)
Tuple<int64_t, float> func5(TensorView t)
```

## Function Class (for dynamic usage)

```cpp
class Function {
public:
  // Get global function by name
  static std::optional<Function> GetGlobal(std::string_view name);
  static Function GetGlobalRequired(std::string_view name);  // Throws if not found

  // Set global function
  static void SetGlobal(std::string_view name, Function func, bool override = false);

  // List all global function names
  static std::vector<String> ListGlobalNames();

  // Remove global function
  static void RemoveGlobal(const String& name);

  // Call function with arguments
  template<typename... Args>
  Any operator()(Args&&... args) const;

  // Create from typed function
  template<typename TCallable>
  static Function FromTyped(TCallable callable);

  template<typename TCallable>
  static Function FromTyped(TCallable callable, std::string name);
};
```

**Usage Example:**
```cpp
// Get and call existing function
auto opt_func = Function::GetGlobal("my.function.name");
if (opt_func.has_value()) {
  Any result = (*opt_func)(arg1, arg2);
}

// Create and register function
Function func = Function::FromTyped([](int64_t x) { return x * 2; });
Function::SetGlobal("double", func);
```

## TypedFunction<R(Args...)>

```cpp
// Type-safe function wrapper
TypedFunction<int(int, int)> add_func = [](int a, int b) { return a + b; };

// Call with type checking
int result = add_func(1, 2);

// Convert to/from Function
Function erased = add_func;
TypedFunction<int(int, int)> typed = TypedFunction<int(int, int)>(erased);
```

---

# Part 5: Module Loading (tvm/ffi/extra/module.h)

**Purpose:** Load and use functions from other compiled modules (for importing helper kernels).

## Module Class

```cpp
class Module {
public:
  // Load module from file
  static Module LoadFromFile(const String& file_name);

  // Get function from module
  Optional<Function> GetFunction(const String& name, bool query_imports = true);

  // Check if function exists
  bool ImplementsFunction(const String& name, bool query_imports = true);

  // Get function metadata
  Optional<String> GetFunctionDoc(const String& name, bool query_imports = true);
  Optional<String> GetFunctionMetadata(const String& name, bool query_imports = true);

  // Import another module
  void ImportModule(const Module& other);

  // Export module
  void WriteToFile(const String& file_name, const String& format) const;
  Bytes SaveToBytes() const;
  String InspectSource(const String& format) const;
  Array<String> GetWriteFormats() const;
};
```

## Module Properties

```cpp
enum ModulePropertyMask {
  kBinarySerializable = 0b001,      // Can be serialized to bytes
  kRunnable = 0b010,                // Has runnable functions
  kCompilationExportable = 0b100    // Can export to .o/.cc/.cu
};
```

## Symbol Names

```cpp
namespace symbol {
  constexpr const char* tvm_ffi_symbol_prefix = "__tvm_ffi_";
  constexpr const char* tvm_ffi_main = "__tvm_ffi_main";
  constexpr const char* tvm_ffi_library_ctx = "__tvm_ffi__library_ctx";
  constexpr const char* tvm_ffi_library_bin = "__tvm_ffi__library_bin";
  constexpr const char* tvm_ffi_metadata_prefix = "__tvm_ffi__metadata_";
}
```

**Usage Example:**
```cpp
// Load helper module
Module helpers = Module::LoadFromFile("helpers.so");

// Get function from module
Optional<Function> opt_func = helpers->GetFunction("helper_kernel");
if (opt_func.defined()) {
  Function helper = opt_func.value();
  // Call helper function
  helper(tensor1, tensor2);
}
```

---

# Part 6: Environment APIs (tvm/ffi/extra/c_env_api.h)

## CUDA Stream Management

```cpp
// Get current CUDA stream for device
TVMFFIStreamHandle TVMFFIEnvGetStream(int32_t device_type, int32_t device_id)

// Set current CUDA stream for device
int TVMFFIEnvSetStream(int32_t device_type,
                       int32_t device_id,
                       TVMFFIStreamHandle stream,
                       TVMFFIStreamHandle* opt_out_original_stream)

// Usage in kernel
DLDevice dev = tensor.device();
cudaStream_t stream = static_cast<cudaStream_t>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));

// Launch kernel on stream
my_kernel<<<blocks, threads, 0, stream>>>(...);
```

## Tensor Allocation

```cpp
// Allocate tensor using environment allocator (respects TLS/global allocator)
int TVMFFIEnvTensorAlloc(DLTensor* prototype, TVMFFIObjectHandle* out)

// Get/Set allocator
DLPackManagedTensorAllocator TVMFFIEnvGetDLPackManagedTensorAllocator()

int TVMFFIEnvSetDLPackManagedTensorAllocator(
    DLPackManagedTensorAllocator allocator,
    int write_to_global_context,
    DLPackManagedTensorAllocator* opt_out_original_allocator)

// Usage: Allocate output tensor
ffi::Tensor output = ffi::Tensor::FromEnvAlloc(
    TVMFFIEnvTensorAlloc,
    shape,   // ffi::ShapeView
    dtype,   // DLDataType
    device   // DLDevice
);
```

## Module Symbol Management

```cpp
// Lookup function from module imports
int TVMFFIEnvModLookupFromImports(TVMFFIObjectHandle library_ctx,
                                  const char* func_name,
                                  TVMFFIObjectHandle* out)

// Register context symbols (available when library is loaded)
int TVMFFIEnvModRegisterContextSymbol(const char* name, void* symbol)

// Register system library symbols
int TVMFFIEnvModRegisterSystemLibSymbol(const char* name, void* symbol)

// Register C API symbols
int TVMFFIEnvRegisterCAPI(const char* name, void* symbol)
```

## Signal Checking

```cpp
// Check for environment signals (e.g., Python Ctrl+C interrupts)
int TVMFFIEnvCheckSignals()

// Usage in long-running loops
for (int i = 0; i < iterations; ++i) {
  if (TVMFFIEnvCheckSignals() != 0) {
    throw tvm::ffi::EnvErrorAlreadySet();
  }
  // ... do work
}
```

---

# Common Patterns and Examples

## Pattern 1: Basic CUDA Kernel

```cpp
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

__global__ void AddKernel(float* a, float* b, float* c, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

void Add(tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView c) {
  // Validate inputs
  TVM_FFI_ICHECK_EQ(a.ndim(), 1) << "a must be 1D";
  TVM_FFI_ICHECK_EQ(b.ndim(), 1) << "b must be 1D";
  TVM_FFI_ICHECK_EQ(c.ndim(), 1) << "c must be 1D";
  TVM_FFI_ICHECK_EQ(a.size(0), b.size(0)) << "Shape mismatch";
  TVM_FFI_ICHECK_EQ(a.size(0), c.size(0)) << "Shape mismatch";

  // Check dtype
  if (a.dtype().code != kDLFloat || a.dtype().bits != 32) {
    TVM_FFI_THROW(TypeError) << "Expected float32 tensor";
  }

  // Get data
  float* a_data = static_cast<float*>(a.data_ptr());
  float* b_data = static_cast<float*>(b.data_ptr());
  float* c_data = static_cast<float*>(c.data_ptr());
  int64_t n = a.size(0);

  // Get CUDA stream
  DLDevice dev = a.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  // Launch kernel
  int64_t threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  AddKernel<<<blocks, threads, 0, stream>>>(a_data, b_data, c_data, n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, Add);
```

## Pattern 2: Multi-Dimensional Tensor Processing

```cpp
void Process2D(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
  // Validate
  TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "Expected 2D tensor";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "Expected 2D tensor";

  // Get dimensions
  int64_t height = input.size(0);
  int64_t width = input.size(1);
  TVM_FFI_ICHECK_EQ(output.size(0), height);
  TVM_FFI_ICHECK_EQ(output.size(1), width);

  // Check contiguous
  TVM_FFI_ICHECK(input.IsContiguous()) << "Input must be contiguous";

  // Access with strides (more general)
  int64_t stride_h = input.stride(0);
  int64_t stride_w = input.stride(1);

  // For contiguous tensors: element[i][j] at data[i * stride_h + j * stride_w]
}
```

## Pattern 3: Allocating Output Tensors

```cpp
tvm::ffi::Tensor AllocateOutput(tvm::ffi::TensorView input,
                                tvm::ffi::ShapeView output_shape) {
  // Create output with same device and dtype as input
  DLDevice device = input.device();
  DLDataType dtype = input.dtype();

  // Allocate using environment allocator
  tvm::ffi::Tensor output = tvm::ffi::Tensor::FromEnvAlloc(
      TVMFFIEnvTensorAlloc,
      output_shape,
      dtype,
      device
  );

  return output;
}

// Usage
void MyKernel(tvm::ffi::TensorView input) {
  // Create output shape
  Shape out_shape = {input.size(0), input.size(1) * 2};

  // Allocate
  Tensor output = AllocateOutput(input, out_shape);

  // Now use output.data_ptr() in kernel
}
```

## Pattern 4: Device-Specific Dispatch

```cpp
void UniversalKernel(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
  DLDevice dev = input.device();

  switch (dev.device_type) {
    case kDLCUDA: {
      // CUDA implementation
      cudaStream_t stream = static_cast<cudaStream_t>(
          TVMFFIEnvGetStream(dev.device_type, dev.device_id));
      // Launch CUDA kernel
      break;
    }
    case kDLCPU: {
      // CPU implementation
      float* in_data = static_cast<float*>(input.data_ptr());
      float* out_data = static_cast<float*>(output.data_ptr());
      // ... CPU code
      break;
    }
    case kDLROCM: {
      // ROCm/HIP implementation
      break;
    }
    default:
      TVM_FFI_THROW(RuntimeError) << "Unsupported device type: " << dev.device_type;
  }
}
```

## Pattern 5: Using Configuration Parameters

```cpp
void ConfigurableKernel(tvm::ffi::TensorView input,
                       tvm::ffi::TensorView output,
                       tvm::ffi::Map<tvm::ffi::String, int64_t> config) {
  // Extract config with defaults
  int64_t block_size = config.Get("block_size", 256);
  int64_t num_threads = config.Get("num_threads", 1024);

  // Or check if key exists
  if (config.count("custom_param") > 0) {
    int64_t custom = config.at("custom_param");
    // use custom
  }

  // Launch with config
  // ...
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(configurable_kernel, ConfigurableKernel);
```

## Pattern 6: Returning Multiple Values

```cpp
// Return tuple
tvm::ffi::Tuple<tvm::ffi::Tensor, int64_t, float>
ComputeWithStats(tvm::ffi::TensorView input) {
  // Allocate output
  tvm::ffi::Tensor output = AllocateOutput(input, input.shape());

  // Compute statistics
  int64_t count = input.numel();
  float mean = 0.0f;  // ... compute mean

  // Return multiple values
  return {output, count, mean};
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(compute_with_stats, ComputeWithStats);
```

## Pattern 7: Using Arrays for Variable Arguments

```cpp
void BatchProcess(tvm::ffi::Array<tvm::ffi::TensorView> inputs,
                 tvm::ffi::Array<tvm::ffi::TensorView> outputs) {
  TVM_FFI_ICHECK_EQ(inputs.size(), outputs.size()) << "Size mismatch";

  for (size_t i = 0; i < inputs.size(); ++i) {
    TensorView in = inputs[i];
    TensorView out = outputs[i];
    // Process each pair
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_process, BatchProcess);
```

---

# Quick Reference Summary

## Most Common Imports

```cpp
#include <tvm/ffi/container/tensor.h>   // TensorView, Tensor
#include <tvm/ffi/function.h>           // TVM_FFI_DLL_EXPORT_TYPED_FUNC
#include <tvm/ffi/error.h>              // TVM_FFI_THROW, TVM_FFI_ICHECK
#include <tvm/ffi/extra/c_env_api.h>    // TVMFFIEnvGetStream
```

## Most Common Function Signature

```cpp
void MyKernel(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
  // 1. Validate inputs
  // 2. Get CUDA stream
  // 3. Launch kernel
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, MyKernel);
```

## Most Common Validation Pattern

```cpp
// Shape
TVM_FFI_ICHECK_EQ(tensor.ndim(), 2);
TVM_FFI_ICHECK_EQ(tensor.size(0), expected_dim0);

// Dtype
DLDataType dt = tensor.dtype();
TVM_FFI_ICHECK_EQ(dt.code, kDLFloat);
TVM_FFI_ICHECK_EQ(dt.bits, 32);

// Device
TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA);

// Memory layout
TVM_FFI_ICHECK(tensor.IsContiguous());
```

## Most Common CUDA Launch Pattern

```cpp
DLDevice dev = tensor.device();
cudaStream_t stream = static_cast<cudaStream_t>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));

float* data = static_cast<float*>(tensor.data_ptr());
int64_t n = tensor.numel();

int threads = 256;
int blocks = (n + threads - 1) / threads;
MyKernel<<<blocks, threads, 0, stream>>>(data, n);
```

---

This completes the comprehensive TVM FFI API reference. Use this as your guide for writing FFI-compatible CUDA kernels and host functions.
"""

FFI_PROMPT = """
You should use TVM FFI bindings for your code.

# Required Headers

```cpp
#include <tvm/ffi/container/tensor.h>   // Tensor, TensorView
#include <tvm/ffi/function.h>           // Function export macros
#include <tvm/ffi/error.h>              // Error handling
#include <tvm/ffi/extra/c_env_api.h>    // Environment APIs (streams, allocators)
```

# Complete Example: CUDA Kernel Binding

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

# TVM FFI API Documentation

## 1. Tensor Container API (tvm/ffi/container/tensor.h)

### Tensor Class
A managed n-dimensional array with reference counting.

**Methods:**
- `void* data_ptr() const` - Returns raw data pointer
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
- `bool IsAligned(const DLTensor& arr, size_t alignment)` - Check alignment
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

## 3. Error Handling API (tvm/ffi/error.h)

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

## 4. Environment API (tvm/ffi/extra/c_env_api.h)

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

# Common Patterns

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
"""
