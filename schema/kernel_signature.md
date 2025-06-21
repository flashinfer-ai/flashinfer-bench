# Kernel Signature

## Overview

This document describes a JSON schema that aims to describe the definition of a kernel, including:
1. The kernel's shape and dtype
2. The kernel's special dimensions (e.g. ragged)
3. The kernel's computation (in PyTorch)

Note the kernel signature does not contain the kernel's exact input data. That will be provided in another file.

## JSON Schema Description

### Top-Level Object Structure

| Field    | Type     | Required | Description                                              |
|----------|----------|----------|---------------------------------------------------------|
| `op`     | string   | Yes      | Operation type. Currently supported: `"gemm"`, `"grouped_gemm"` |
| `axes`   | object   | Yes      | Key-value pairs of axis definitions                     |
| `input`  | object   | Yes      | Named input tensors (e.g. `"A"`, `"B"`)                 |
| `output` | object   | Yes      | Named output tensors (e.g. `"C"`)                       |
| `code`   | string      | Yes       | The PyTorch code of the kernel. |

### `op` Field: Enum of Operation Type

| Value | Meaning |
|-------|---------|
| `gemm` | GEMM or Transposed GEMM |
| `grouped_gemm` | Grouped GEMM where the input tensor can be ragged |

### `axes` Field: Axes

The `axes` object contains any number of keys, where each key is an axis name (e.g., `"M"`, `"N"`, `"K"`, `"G"`), and the value is an object describing the axis type and its constraints.

#### `type`: `const`

Represents a constant axis.

| Field   | Type    | Required | Description                    |
|---------|---------|----------|--------------------------------|
| `type`  | string  | Yes      | Must be `"const"`              |
| `value` | integer | Yes      | Constant value of the axis     |


Example:

```json
"M": {
  "type": "const",
  "value": 4096
}
```

#### `type`: `var`

Represents a variable axis whose value will be determined by the input data. The `parent` field can be used to indicate hierarchical axis relationships, such as a grouped dimension structure.

| Field    | Type    | Required | Description                               | Default |
|----------|---------|----------|-------------------------------------------|---------|
| `type`   | string  | Yes      | Must be `"var"`                           | —       |
| `parent` | string  | No       | (Optional) name of parent axis for nesting| `null`  |

Example:

```json
"M": {
  "type": "var",
  "parent": "G"
}
```

### `input`, `output` Field: Input and Output Tensors

Input and output describes the input and the output tensors of the kernel. They can contain any number of keys, where each key is the name of a tensor (e.g., `"A"`, `"B"`, `"C"`). The value is a tensor description:

| Field       | Type    | Required | Description                                          | Default |
|-------------|---------|----------|------------------------------------------------------|---------|
| `shape`     | array   | Yes      | List of axis names (strings)                         | —       |
| `dim_order` | array   | No      | Permutation indices of the shape axes                | `[0, 1, ...]`       |
| `dtype`     | string  | No      | Data type of the tensor                              | `"fp16"`       |

#### `dtype` Field: Enum of Data Type

The following values are allowed for `dtype`:

- `fp32`
- `fp16`
- `bf16`
- `fp8_e4m3`
- `fp8_e5m2`
- `fp4_e2m1`
- `int4`
- `int8`

Example:

```json
"A": {
  "shape": ["M", "K"],
  "dim_order": [1, 0],
  "dtype": "fp16"
}
```


### `code` Field: PyTorch Code

The `code` field is a string that contains the PyTorch code of the kernel. It should contain a global `forward` function as the entry point.


## Examples

### Example 1: Standard GEMM

```json
{
  "op": "gemm",
  "axes": {
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 }
  },
  "input": {
    "A": {
      "shape": ["M", "K"],
      "dim_order": [0, 1],
      "dtype": "fp16"
    },
    "B": {
      "shape": ["N", "K"],
      "dim_order": [0, 1],
      "dtype": "fp16"
    }
  },
  "output": {
    "C": {
      "shape": ["M", "N"],
      "dim_order": [0, 1],
      "dtype": "fp16"
    }
  },
  "code": "..."
}
```

### Example 2: Quantized GEMM

```json
{
  "op": "gemm",
  "axes": {
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 },
    "N_group": { "type": "const", "value": "4096 / 32" }
  },
  "input": {
    "A": {
      "shape": ["M", "K"],
      "dtype": "float16"
    },
    "B_quantized": {
      "shape": ["N", "K"],
      "dim_order": [0, 1],
      "dtype": "int4"
    },
    "B_scale": {
      "shape": ["N_group", "K"],
      "dim_order": [0, 1],
      "dtype": "fp32"
    },
  },
  "output": {
    "C": {
      "shape": ["M", "N"],
      "dim_order": [0, 1],
      "dtype": "float16"
    }
  },
  "code": "..."
}
```

### Example 3: Grouped GEMM

```json
{
  "op": "grouped_gemm",
  "axes": {
    "G": { "type": "var" },
    "M": { "type": "var", "parent": "G" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 }
  },
  "input": {
    "A": {
      "shape": ["G", "M", "K"],
      "dim_order": [0, 1, 2],
      "dtype": "fp16"
    },
    "B": {
      "shape": ["G", "K", "N"],
      "dim_order": [0, 1, 2],
      "dtype": "fp16"
    }
  },
  "output": {
    "C": {
      "shape": ["G", "M", "N"],
      "dim_order": [0, 1, 2],
      "dtype": "fp16"
    }
  },
  "code": "..."
}
```
