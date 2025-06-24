# Benchmark Result

## Overview

This document describes a JSON schema that aims to describe the benchmark results of a kernel, including:
1. The device information where the benchmark was executed
2. The kernel specification being benchmarked
3. Multiple generation rounds with performance metrics
4. Overall speedup statistics

Note the benchmark result contains both compilation success/failure information and performance measurements compared to baseline implementations.

## JSON Schema Description

### Top-Level Object Structure

| Field              | Type   | Required | Description                                |
|--------------------|--------|----------|--------------------------------------------|
| `device`           | object | Yes      | Information about the execution device     |
| `kernel_signature` | object | Yes      | The kernel signature being benchmarked     |
| `baseline`         | object | Yes      | The baseline results                       |
| `generations`      | array  | Yes      | Array of generation results                |
| `success_rate`     | number | Yes      | Correctness rate across all generations    |
| `avg_of_n`         | number | Yes      | Average speedup across all successful runs |
| `max_of_n`         | number | Yes      | Maximum speedup across all successful runs |

### `device` Field: Device Information

The `device` object contains information about the hardware and software environment where the benchmark was executed.

| Field              | Type   | Required | Description                           |
|--------------------|--------|----------|---------------------------------------|
| `name`             | string | Yes      | Device name (e.g., "NVIDIA A100")    |

### `kernel_signature` Field: Kernel Signature

The `kernel_signature` object contains the complete kernel signature as defined in the [kernel signature schema](kernel_signature.md). This includes the kernel's name, axes, inputs, outputs, and code.

### `baseline` Field: Baseline Results

The `baseline` object contains the benchmark results. It has the following fields:

| Field      | Type    | Required | Description                            |
|------------|---------|----------|----------------------------------------|
| `compiled` | boolean | Yes      | Whether the code compiled successfully |
| `time`     | number  | No       | Execution time in milliseconds         |

### `generations` Field: Generation Results

The `generations` field is an array of objects, where each object represents the result of one generation round.

| Field            | Type    | Required | Description                                  |
|------------------|---------|----------|----------------------------------------------|
| `round`          | integer | Yes      | Generation round number (0-indexed)          |
| `generated_code` | string  | Yes      | The generated kernel code                    |
| `compiled`       | boolean | Yes      | Whether the code compiled successfully       |
| `max_diff`       | number  | No       | Maximum difference from reference output     |
| `time`           | number  | No       | Execution time in milliseconds               |
| `speedup`  | number  | No       | Speedup compared to baseline |

#### `max_diff` Field

The `max_diff` field represents the maximum absolute difference between the generated kernel output and the reference implementation output. This is only present when the kernel compiled successfully.

#### `speedup` Field

The `speedup` field is the ratio of the generated kernel's execution time to the baseline's execution time. A value larger than 1.0 indicates the generated kernel is faster than the baseline.

### `avg_of_n` and `max_of_n` Fields

| Field      | Type   | Required | Description                                    |
|------------|--------|----------|------------------------------------------------|
| `avg_of_n` | number | Yes      | Average speedup across all successful runs |
| `max_of_n` | number | Yes      | Maximum speedup across all successful runs |

## Examples
// ... existing code ...

## Examples

### Example 1: Successful GEMM Benchmark

```json
{
  "device": {
    "name": "NVIDIA A100-SXM4-80GB"
  },
  "kernel_spec": {
    "name": "gemm",
    "axes": {
      "M": { "type": "var" },
      "N": { "type": "const", "value": 4096 },
      "K": { "type": "const", "value": 4096 }
    },
    "inputs": {
      "A": {
        "shape": ["M", "K"],
        "dtype": "float16"
      },
      "B": {
        "shape": ["N", "K"],
        "dtype": "float16"
      }
    },
    "outputs": {
      "C": {
        "shape": ["M", "N"],
        "dtype": "float16"
      }
    },
    "code": "def forward(A, B):\n    return torch.matmul(A, B.T)"
  },
  "baseline": {
    "compiled": true,
    "time": 0.250
  },
  "generations": [
    {
      "round": 0,
      "generated_code": "...",
      "compiled": true,
      "max_diff": 1e-6,
      "time": 0.125,
      "speedup": 2.0
    },
    {
      "round": 1,
      "generated_code": "...",
      "compiled": true,
      "max_diff": 5e-7,
      "time": 0.100,
      "speedup": 2.5
    }
  ],
  "success_rate": 1.0,
  "avg_of_n": 2.25,
  "max_of_n": 2.5
}
```

### Example 2: Failed Compilation

```json
{
  "device": {
    "name": "NVIDIA RTX 4090"
  },
  "kernel_spec": {
    "name": "complex_kernel",
    "axes": {
      "N": { "type": "var" }
    },
    "inputs": {
      "x": {
        "shape": ["N"],
        "dtype": "float32"
      }
    },
    "outputs": {
      "y": {
        "shape": ["N"],
        "dtype": "float32"
      }
    },
    "code": "def forward(x):\n    return torch.relu(x)"
  },
  "baseline": {
    "compiled": true,
    "time": 0.060
  },
  "generations": [
    {
      "round": 0,
      "generated_code": "...",
      "compiled": false
    },
    {
      "round": 1,
      "generated_code": "...",
      "compiled": true,
      "max_diff": 0.0,
      "time": 0.050,
      "speedup": 1.2
    }
  ],
  "success_rate": 0.5,
  "avg_of_n": 1.2,
  "max_of_n": 1.2
}
```

### Example 3: Multiple Successful Generations

```json
{
  "device": {
    "name": "NVIDIA H100"
  },
  "kernel_spec": {
    "name": "rmsnorm",
    "axes": {
      "batch_size": { "type": "var" },
      "hidden_size": { "type": "const", "value": 4096 }
    },
    "inputs": {
      "input": {
        "shape": ["batch_size", "hidden_size"],
        "dtype": "float16"
      },
      "weight": {
        "shape": ["hidden_size"],
        "dtype": "float16"
      },
      "eps": {
        "shape": [],
        "dtype": "float32"
      }
    },
    "outputs": {
      "output": {
        "shape": ["batch_size", "hidden_size"],
        "dtype": "float16"
      }
    },
    "code": "..."
  },
  "baseline": {
    "compiled": true,
    "time": 0.100
  },
  "generations": [
    {
      "round": 0,
      "generated_code": "...",
      "compiled": true,
      "max_diff": 2e-6,
      "time": 0.080,
      "speedup": 1.25
    },
    {
      "round": 1,
      "generated_code": "...",
      "compiled": true,
      "max_diff": 1e-6,
      "time": 0.060,
      "speedup": 1.67
    },
    {
      "round": 2,
      "generated_code": "...",
      "compiled": true,
      "max_diff": 3e-7,
      "time": 0.045,
      "speedup": 2.22
    }
  ],
  "success_rate": 1.0,
  "avg_of_n": 1.71,
  "max_of_n": 2.22
}
```
