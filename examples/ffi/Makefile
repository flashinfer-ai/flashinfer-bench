# Makefile for building the C++ TVM-FFI example
#
# Usage:
#   make              # Build the example
#   make run          # Build and run the example
#   make clean        # Clean build artifacts

# Compiler settings
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++17 -O3 -Wall

# TVM-FFI paths
# You can override these by setting environment variables:
#   make TVM_FFI_HOME=/path/to/tvm-ffi
# Or specify custom paths directly:
#   make TVM_FFI_INCLUDE=/path/to/include TVM_FFI_LIB=/path/to/lib
PYTHON ?= python3

TVM_FFI_PKG := $(shell $(PYTHON) -c "import tvm_ffi; print(tvm_ffi.__path__[0])" 2>/dev/null)

ifeq ($(TVM_FFI_PKG),)
    $(warning tvm_ffi not found via Python, falling back to /usr/local)
    TVM_FFI_PKG := /usr/local
    TVM_FFI_INCLUDE ?= $(TVM_FFI_PKG)/include
    TVM_FFI_LIB ?= $(TVM_FFI_PKG)/lib
    DLPACK_INCLUDE ?= $(TVM_FFI_PKG)/include
else
    TVM_FFI_INCLUDE ?= $(TVM_FFI_PKG)/include
    TVM_FFI_LIB ?= $(TVM_FFI_PKG)/lib
    DLPACK_INCLUDE ?= $(TVM_FFI_PKG)/../3rdparty/dlpack/include
endif

# CUDA paths
CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDE := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64

# Include and library flags
INCLUDES := -I$(TVM_FFI_INCLUDE) -I$(DLPACK_INCLUDE) -I$(CUDA_INCLUDE)
LDFLAGS := -L$(TVM_FFI_LIB) -L$(CUDA_LIB)
LIBS := -ltvm_ffi -lcuda -lcudart
RPATH := -Wl,-rpath=$(TVM_FFI_LIB) -Wl,-rpath=$(CUDA_LIB)

TARGET := cpp_example
SOURCE := cpp_example.cc

.PHONY: all run clean help check

all: $(TARGET)

check:
	@echo "Detected Configuration:"
	@echo "  Python:           $(PYTHON)"
	@echo "  TVM-FFI Package:  $(TVM_FFI_PKG)"
	@echo "  TVM-FFI Include:  $(TVM_FFI_INCLUDE)"
	@echo "  TVM-FFI Library:  $(TVM_FFI_LIB)"
	@echo "  DLPack Include:   $(DLPACK_INCLUDE)"
	@echo "  CUDA Home:        $(CUDA_HOME)"
	@echo "  CUDA Include:     $(CUDA_INCLUDE)"
	@echo "  CUDA Library:     $(CUDA_LIB)"

$(TARGET): $(SOURCE)
	@echo "Building $(TARGET)..."
	@echo "  TVM-FFI include: $(TVM_FFI_INCLUDE)"
	@echo "  DLPack include: $(DLPACK_INCLUDE)"
	@echo "  TVM-FFI lib: $(TVM_FFI_LIB)"
	@echo "  CUDA include: $(CUDA_INCLUDE)"
	@echo "  CUDA lib: $(CUDA_LIB)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) -o $(TARGET) $(LDFLAGS) $(LIBS) $(RPATH)
	@echo "✓ Build successful: ./$(TARGET)"

run: $(TARGET)
	@echo "Running $(TARGET)..."
	@echo ""
	./$(TARGET)

clean:
	rm -f $(TARGET)
	@echo "Cleaned build artifacts"

help:
	@echo "TVM-FFI C++ Example Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make        - Build the C++ example"
	@echo "  make run    - Build and run the example"
	@echo "  make check  - Show detected paths and configuration"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help message"
	@echo ""
	@echo "Environment Variables:"
	@echo "  PYTHON            - Python executable to use for detection (default: python3)"
	@echo "  TVM_FFI_INCLUDE   - Path to TVM-FFI headers (default: auto-detect)"
	@echo "  TVM_FFI_LIB       - Path to TVM-FFI libraries (default: auto-detect)"
	@echo "  DLPACK_INCLUDE    - Path to DLPack headers (default: auto-detect)"
	@echo "  CUDA_HOME         - Path to CUDA installation (default: /usr/local/cuda)"
	@echo ""
	@echo "Examples:"
	@echo "  make                                    # Auto-detect all paths"
	@echo "  make PYTHON=python3.12                  # Use specific Python version"
	@echo "  make CUDA_HOME=/usr/local/cuda-12.0     # Use specific CUDA version"
	@echo "  make TVM_FFI_INCLUDE=/custom/include TVM_FFI_LIB=/custom/lib  # Custom paths"

