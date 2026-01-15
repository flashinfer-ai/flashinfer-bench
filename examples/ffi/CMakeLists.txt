cmake_minimum_required(VERSION 3.18)
project(TVM_FFI_CPP_Example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tvm_ffi; print(tvm_ffi.__path__[0])"
    OUTPUT_VARIABLE TVM_FFI_PKG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE TVM_FFI_FOUND
)

if(TVM_FFI_FOUND EQUAL 0)
    message(STATUS "TVM-FFI found via Python: ${TVM_FFI_PKG}")
    set(TVM_FFI_INCLUDE_DIR "${TVM_FFI_PKG}/include" CACHE PATH "TVM-FFI include directory")
    set(TVM_FFI_LIB_DIR "${TVM_FFI_PKG}/lib" CACHE PATH "TVM-FFI library directory")

    get_filename_component(TVM_FFI_PARENT "${TVM_FFI_PKG}/.." ABSOLUTE)
    set(DLPACK_INCLUDE_DIR "${TVM_FFI_PARENT}/3rdparty/dlpack/include" CACHE PATH "DLPack include directory")
else()
    message(WARNING "tvm_ffi not found via Python, falling back to /usr/local")
    set(TVM_FFI_INCLUDE_DIR "/usr/local/include" CACHE PATH "TVM-FFI include directory")
    set(TVM_FFI_LIB_DIR "/usr/local/lib" CACHE PATH "TVM-FFI library directory")
    set(DLPACK_INCLUDE_DIR "/usr/local/include" CACHE PATH "DLPack include directory")
endif()

if(NOT EXISTS "${TVM_FFI_INCLUDE_DIR}/tvm/ffi/function.h")
    message(FATAL_ERROR "TVM-FFI headers not found at ${TVM_FFI_INCLUDE_DIR}")
endif()

message(STATUS "TVM-FFI include: ${TVM_FFI_INCLUDE_DIR}")
message(STATUS "TVM-FFI lib: ${TVM_FFI_LIB_DIR}")
message(STATUS "DLPack include: ${DLPACK_INCLUDE_DIR}")

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")

find_library(TVM_FFI_LIB
    NAMES tvm_ffi libtvm_ffi
    PATHS ${TVM_FFI_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)
message(STATUS "TVM-FFI library: ${TVM_FFI_LIB}")

add_executable(cpp_example cpp_example.cc)

target_include_directories(cpp_example PRIVATE
    ${TVM_FFI_INCLUDE_DIR}
    ${DLPACK_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cpp_example PRIVATE
    ${TVM_FFI_LIB}
    CUDA::cuda_driver
    CUDA::cudart
)

set_target_properties(cpp_example PROPERTIES
    BUILD_RPATH "${TVM_FFI_LIB_DIR};${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${TVM_FFI_LIB_DIR};${CUDAToolkit_LIBRARY_DIR}"
    BUILD_RPATH_USE_ORIGIN TRUE
)

add_custom_target(run
    COMMAND cpp_example
    DEPENDS cpp_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running cpp_example..."
)

add_custom_target(check
    COMMAND ${CMAKE_COMMAND} -E echo "Detected Configuration:"
    COMMAND ${CMAKE_COMMAND} -E echo "  Python:           ${Python3_EXECUTABLE}"
    COMMAND ${CMAKE_COMMAND} -E echo "  TVM-FFI Package:  ${TVM_FFI_PKG}"
    COMMAND ${CMAKE_COMMAND} -E echo "  TVM-FFI Include:  ${TVM_FFI_INCLUDE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "  TVM-FFI Library:  ${TVM_FFI_LIB_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "  DLPack Include:   ${DLPACK_INCLUDE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "  CUDA Version:     ${CUDAToolkit_VERSION}"
    COMMAND ${CMAKE_COMMAND} -E echo "  CUDA Include:     ${CUDAToolkit_INCLUDE_DIRS}"
    COMMAND ${CMAKE_COMMAND} -E echo "  CUDA Library:     ${CUDAToolkit_LIBRARY_DIR}"
    COMMENT "Showing detected configuration"
)
