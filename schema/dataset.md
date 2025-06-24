# Dataset Schema

## Overview

This document describes a JSON schema that defines the structure of benchmark datasets, including:
1. The target model information
2. The inference framework being used
3. The kernel invocation statistics

The dataset schema is used to describe the execution profile of kernels for a specific model running on a particular framework.

## JSON Schema Description

### Top-Level Object Structure

| Field       | Type   | Required | Description                                           |
|-------------|--------|----------|-------------------------------------------------------|
| `model`     | string | Yes      | HuggingFace model name or identifier                  |
| `framework` | string | Yes      | Inference framework name                              |
| `kernels`   | array  | Yes      | Array of kernel execution statistics                  |

### `model` Field: Model Identifier

The `model` field contains the HuggingFace model name or identifier (e.g., `"meta-llama/Llama-3.1-8B"`, `"microsoft/DialoGPT-medium"`).

### `framework` Field: Inference Framework

The `framework` field specifies which inference framework is being used. The following values are allowed:

- `"SGLang"`
- `"vLLM"`
- `"MLC-LLM"`
- `"TRT-LLM"`

### `kernels` Field: Kernel Execution Statistics

The `kernels` field is an array of objects, where each object describes the execution statistics for a specific kernel.

#### Kernel Object Structure

| Field    | Type    | Required | Description                                               |
|----------|---------|----------|-----------------------------------------------------------|
| `kernel` | object  | Yes      | The [kernel signature](kernel_signature.md) of the kernel |
| `count`  | integer | Yes      | Number of times this kernel was invoked                   |

## Examples

### Example 1: SGLang with Llama Model

```json
{
  "model": "meta-llama/Llama-3.1-8B",
  "framework": "SGLang",
  "kernels": [
    {
      "kernel": {
        "name": "silu_and_mul",
        "axes": {
          "M": { "type": "const", "value": 1024 },
          "N": { "type": "const", "value": 1024 }
        },
        "inputs": { "A": { "shape": ["M", "N"], "dtype": "float16" }, "B": { "shape": ["M", "N"], "dtype": "float16" } },
        "outputs": { "C": { "shape": ["M", "N"], "dtype": "float16" } },
        "code": "def forward(A, B):\n    return torch.mul(torch.silu(A), B)"
      },
      "count": 4
    },
    {
      "kernel": {
        "name": "rmsnorm",
        "axes": {
          "M": { "type": "const", "value": 1024 },
          "N": { "type": "const", "value": 1024 }
        },
        "inputs": { "A": { "shape": ["M", "N"], "dtype": "float16" }, "B": { "shape": ["M", "N"], "dtype": "float16" } },
        "outputs": { "C": { "shape": ["M", "N"], "dtype": "float16" } },
        "code": "def forward(A, B):\n    return torch.mul(torch.silu(A), B)"
      },
      "count": 2
    },
  ]
}
```
