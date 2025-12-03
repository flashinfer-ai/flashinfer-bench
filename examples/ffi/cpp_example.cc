/**
 * C++ example: Load and run the distributed .so kernel using TVM-FFI C++ API.
 * Build with:
 *   g++ -std=c++17 cpp_example.cc -o cpp_example \
 *       -I/path/to/tvm-ffi/include \
 *       -L/path/to/tvm-ffi/lib \
 *       -ltvm_ffi -lcuda -lcublas \
 *       -Wl,-rpath=/path/to/tvm-ffi/lib
 */

 #include <tvm/ffi/container/tensor.h>
 #include <tvm/ffi/extra/module.h>
 #include <tvm/ffi/extra/c_env_api.h>
 #include <dlpack/dlpack.h>
 #include <iostream>
 #include <fstream>
 #include <string>
 #include <vector>
 #include <cmath>
 #include <cuda_runtime.h>
 #include <cuda_fp16.h>

 namespace ffi = tvm::ffi;

 std::string read_entry_symbol(const std::string& metadata_path) {
     std::ifstream file(metadata_path);
     std::string line;
     while (std::getline(file, line)) {
         if (line.find("Entry Symbol:") == 0) {
             size_t colon_pos = line.find(":");
             if (colon_pos != std::string::npos) {
                 std::string symbol = line.substr(colon_pos + 1);
                 size_t start = symbol.find_first_not_of(" \t");
                 size_t end = symbol.find_last_not_of(" \t\r\n");
                 if (start != std::string::npos && end != std::string::npos) {
                     return symbol.substr(start, end - start + 1);
                 }
             }
         }
     }
     return "";
 }

 void init_random_tensor(ffi::TensorView tensor) {
     size_t num_elements = 1;
     for (int i = 0; i < tensor.ndim(); ++i) {
         num_elements *= tensor.shape()[i];
     }

     std::vector<__half> host_data(num_elements);

     for (size_t i = 0; i < num_elements; ++i) {
         host_data[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
     }

     cudaMemcpy(tensor.data_ptr(), host_data.data(),
                num_elements * sizeof(__half), cudaMemcpyHostToDevice);
 }

 int cuda_tensor_alloc(DLTensor* prototype, DLManagedTensorVersioned** out, void* error_ctx,
                      void (*SetError)(void* error_ctx, const char* kind, const char* message)) {
     size_t num_bytes = 1;
     for (int i = 0; i < prototype->ndim; ++i) {
         num_bytes *= prototype->shape[i];
     }
     num_bytes *= (prototype->dtype.bits * prototype->dtype.lanes + 7) / 8;

     void* ptr;
     cudaError_t err = cudaMalloc(&ptr, num_bytes);
     if (err != cudaSuccess) {
         if (SetError) {
             SetError(error_ctx, "RuntimeError", cudaGetErrorString(err));
         }
         return -1;
     }

     int64_t* shape = new int64_t[prototype->ndim];
     int64_t* strides = nullptr;
     for (int i = 0; i < prototype->ndim; ++i) {
         shape[i] = prototype->shape[i];
     }
     if (prototype->strides) {
         strides = new int64_t[prototype->ndim];
         for (int i = 0; i < prototype->ndim; ++i) {
             strides[i] = prototype->strides[i];
         }
     }

     // Allocate DLManageTensorVersioned structure
     DLManagedTensorVersioned* managed = new DLManagedTensorVersioned();
     managed->version = {1, 0};
     managed->manager_ctx = nullptr;
     managed->flags = 0;

     // Setup deleter
     managed->deleter = [](DLManagedTensorVersioned* self) {
         if (self->dl_tensor.data) {
             cudaFree(self->dl_tensor.data);
         }
         delete[] self->dl_tensor.shape;
         if (self->dl_tensor.strides) {
             delete[] self->dl_tensor.strides;
         }
         delete self;
     };

     // Setup DLTensor
     managed->dl_tensor = *prototype;
     managed->dl_tensor.data = ptr;
     managed->dl_tensor.shape = shape;
     managed->dl_tensor.strides = strides;

     *out = managed;
     return 0;
 }

 ffi::Tensor allocate_cuda_tensor(std::vector<int64_t> shape, DLDataType dtype) {
     DLDevice device{kDLCUDA, 0};
     ffi::ShapeView shape_view(shape.data(), shape.size());
     return ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, shape_view, dtype, device);
 }

 int main() {
     cudaSetDevice(0);

     DLDevice device{kDLCUDA, 0};
     TVMFFIEnvSetDLPackManagedTensorAllocator(cuda_tensor_alloc, 1, nullptr);

     cudaStream_t stream;
     cudaStreamCreate(&stream);
     TVMFFIEnvSetStream(device.device_type, device.device_id, stream, nullptr);

     const std::string dist_dir = "distributed";
     std::string entry_symbol = read_entry_symbol(dist_dir + "/kernel_metadata.txt");

     ffi::Module mod = ffi::Module::LoadFromFile(dist_dir + "/kernel.so");
     ffi::Function kernel_fn = mod->GetFunction(entry_symbol).value();

     std::cout << "Loaded kernel: " << entry_symbol << std::endl;

     // Prepare inputs: C = A @ B.T
     const int64_t M = 1024, N = 4096, K = 4096;
     DLDataType dtype{kDLFloat, 16, 1};

     ffi::Tensor A = allocate_cuda_tensor({M, K}, dtype);
     ffi::Tensor B = allocate_cuda_tensor({N, K}, dtype);
     ffi::Tensor C = allocate_cuda_tensor({M, N}, dtype);

     init_random_tensor(A);
     init_random_tensor(B);
     cudaMemset(C.data_ptr(), 0, M * N * sizeof(__half));

     kernel_fn(A, B, C);
     cudaDeviceSynchronize();

     std::vector<__half> host_output(M * N);
     cudaMemcpy(host_output.data(), C.data_ptr(),
                M * N * sizeof(__half), cudaMemcpyDeviceToHost);

     std::cout << "Output shape: (" << M << ", " << N << ")" << std::endl;

     std::cout << "First 10 elements: [";
     size_t num_to_print = std::min(size_t(10), host_output.size());
     for (size_t i = 0; i < num_to_print; ++i) {
         std::cout << __half2float(host_output[i]);
         if (i < num_to_print - 1) std::cout << ", ";
     }
     std::cout << "]" << std::endl;

     cudaStreamDestroy(stream);

     return 0;
 }
