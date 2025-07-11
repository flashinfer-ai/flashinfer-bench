# Implementation

This directory contains kernel generator implementations that expose a standardized `generate_kernel` function interface for the benchmark system.

---

## Overview

Whether implemented by AI agents or human experts, all kernel generators in this directory must expose a `generate_kernel` function that takes a PyTorch model description and returns optimized kernel code as a string. The benchmark system compiles this generated code into a runnable module for correctness testing and performance evaluation.

## Generator Interface

### Required Function Signature

```python
def generate_kernel(problem_description: str, turns: int = 1) -> str:
    """Generate optimized kernel code from PyTorch model description
    
    Args:
        problem_description: PyTorch model code with Model class, get_inputs(), get_init_inputs()
        turns: Number of generation attempts (implementation-specific)
        
    Returns:
        Generated kernel code as string (e.g., Triton, CUDA, or optimized PyTorch)
    """
```

---

## Current Files

**`kernelllm_generator.py`** - KernelLLM agent implementation

**`kevin_generator.py`** - Kevin-32B agent implementation

**`kernelllm_server.py`** - Server interface for KernelLLM (via SGLang)

**`kevin_server.py`** - Server interface for Kevin model (via SGLang)


---

## Currnet Input Format

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # model definition
    
    def forward(self, x):
        # forward pass implementation
        return result

def get_inputs():
    # Returns list of input tensors for the model
    return [torch.randn(batch_size, input_dim)]

def get_init_inputs():
    # Returns initialization arguments for Model() constructor
    return []
```

## Current Output Format

Generators must return optimized kernel code that defines a `ModelNew` class with the same interface as the original `Model`:

```python
import torch
import torch.nn as nn
# Import statements for optimization framework (triton, etc.)

# Custom kernel implementations...

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # optimized model definition
    
    def forward(self, x):
        # optimized forward pass using custom kernels
        return result
```

---

## Integration

The benchmark system:

1. Loads the generator module
2. Calls `generate_kernel(problem_description)` 
3. Compiles the returned code string using the workload loader
4. Tests for correctness against the original model
5. Measures performance speedup of the optimized implementation

This standardized interface enables seamless integration of different optimization approaches while maintaining consistent evaluation methodology. 