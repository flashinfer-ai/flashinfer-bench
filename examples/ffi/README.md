# FlashInfer Bench (TVM-FFI) Kernel Distribution Example

This directory contains examples demonstrating how to build, distribute, and load agent generated CUDA kernels using TVM-FFI across different environments.


## Overview

The workflow consists of 4 main stages:

```
┌──────────────────────────────────────────────────────────────────────┐
│ Stage 1: Problem Definition                                          │
│ - Definition in FlashInfer-Trace dataset                             │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Stage 2: LLM Kernel Generation                                       │
│ - LLM reads the Definition + agent_vibecode.md prompt                │
│ - Generates CUDA kernel with TVM-FFI bindings                        │
│ - Outputs Solution JSON with embedded source code                    │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Stage 3: Build & Distribution                                        │
│ - TVMFFIBuilder compiles the Solution                                │
│ - Generates framework-agnostic .so binary                            │
│ - Extracts to distributed/ folder with metadata                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Stage 4: Cross-Framework Usage                                       │
│ - JAX/PyTorch/C++ load the same .so file                             │
│ - Execute kernel without recompilation                               │
│ - Benchmarking or Apply the kernel                                   │
└──────────────────────────────────────────────────────────────────────┘
```


## Installation Prerequisites

### Python Dependencies
```bash
pip install flashinfer-bench tvm-ffi torch
```

### For JAX Example
```bash
pip install jax[cuda13-local] jax-tvm-ffi
```

### For C++ Example
- CUDA Toolkit
- TVM-FFI C++ headers and libraries
- C++17 compatible compiler

## Usage

### 1. Generate Kernel with Agent

You have two options to generate a CUDA kernel solution:

**Option A: IDE Coding Agent**

Use your preferred IDE coding agent to generate a GEMM kernel solution:

```bash
# Open the instructions in your IDE
cat agent_vibecode.md
```

Follow the instructions in `agent_vibecode.md` to have the agent generate the solution interactively.

**Option B: Kernel Generator**

Use the kernel generator agent to generate solutions:

```bash
# Ensure .env is configured in examples/kernel_generator/
# Required: LLM_API_KEY and BASE_URL
python kernel_generator_example.py
```

Configure generation parameters in the script:
- `model_name`: LLM model to use (default: `gpt-5-2025-08-07`)
- `target_gpu`: Target GPU architecture (default: `B200`)
- `gen_rounds`: Number of refinement rounds (default: `10`)

### 2. Build and Distribute
```bash
cd /flashinfer-bench/examples/ffi
python distribute_kernel.py
```

This builds the kernel and extracts `kernel.so` to `distributed/` folder.

### 3. Run Examples

**JAX:**
```bash
python jax_example.py
```

**PyTorch:**
```bash
python pytorch_example.py
```

**C++:**
```bash
make run
```

Each example loads the distributed kernel, executes it, and prints output shape and elements.

## How It Works

The kernel is built using `TVMFFIBuilder`, producing a self-contained `.so` file. This binary can be loaded across different runtimes:

- **JAX**: Uses `tvm_ffi.load_module()` and `jax.ffi.ffi_call()`
- **PyTorch**: Uses `torch.utils.cpp_extension.load()` with custom CUDA extensions
- **C++**: Uses `ffi::Module::LoadFromFile()`

The same `.so` file works across all frameworks without recompilation.

## Notes

- Kernels use destination-passing style (pre-allocated outputs)
- All examples use CUDA tensors on GPU device 0
- Entry point format: `file.cu::function_name`
