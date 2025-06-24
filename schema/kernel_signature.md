# Kernel Signature

## Overview

This document describes a JSON schema that aims to describe the definition of a kernel, including:
1. The kernel's shape and dtype
2. The kernel's special dimensions (e.g. ragged)
3. The kernel's computation (in PyTorch)

Note the kernel signature does not contain the kernel's exact input data. That will be provided in another file.

## JSON Schema Description

### Top-Level Object Structure

| Field     | Type   | Required | Description                             |
|-----------|--------|----------|-----------------------------------------|
| `name`    | string | Yes      | A unique name indicating the kernel.    |
| `axes`    | object | Yes      | Key-value pairs of axis definitions     |
| `inputs`  | object | Yes      | Named input tensors (e.g. `"A"`, `"B"`) |
| `outputs` | object | Yes      | Named output tensors (e.g. `"C"`)       |
| `code`    | string | Yes      | The PyTorch code of the kernel.         |

Q: "name" or "id"?

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

### `inputs`, `outputs` Field: Input and Output Tensors

Input and output describes the input and the output tensors of the kernel. They can contain any number of keys, where each key is the name of a tensor (e.g., `"A"`, `"B"`, `"C"`). The value is a tensor description:

| Field       | Type   | Required | Description                           | Default       |
|-------------|--------|----------|---------------------------------------|---------------|
| `shape`     | array  | Yes      | List of axis names (strings)          | —             |
| `dim_order` | array  | No       | Permutation indices of the shape axes | `[0, 1, ...]` |
| `dtype`     | string | No       | Data type of the tensor               | `"float16"`   |

#### `dtype` Field: Enum of Data Type

The following values are allowed for `dtype`:

- `float32`
- `float16`
- `bfloat16`
- `float8_e4m3`
- `float8_e5m2`
- `float4_e2m1`
- `int8`
- `bool`

Q: float8 or fp8, fp16 or float16, etc.

#### Scalar Values

We can define a tensor with a empty shape (`[]`) to represent a scalar value. The scalar input can
not only accept tensor data (torch tensor with shape `[]`), but also scalar data (python int, float, bool).
The scalar output will return a python scalar value.

Example:

```json
"inputs": {
  "A": {
    "shape": ["M", "K"],
    "dim_order": [1, 0],
    "dtype": "float16"
  },
  "scalar": {
    "shape": [],
    "dtype": "float16"
  }
},
"outputs": {
  "C": {
    "shape": ["M", "N"],
    "dim_order": [0, 1],
    "dtype": "float16"
  }
}
```


### `code` Field: PyTorch Code

The `code` field is a string that contains the PyTorch code of the kernel. It should contain a global `forward` function as the entry point.


## Examples

### Example 1: Standard GEMM

```json
{
  "name": "gemm",
  "axes": {
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "A": {
      "shape": ["M", "K"],
      "dim_order": [0, 1],
      "dtype": "float16"
    },
    "B": {
      "shape": ["N", "K"],
      "dim_order": [0, 1],
      "dtype": "float16"
    }
  },
  "outputs": {
    "C": {
      "shape": ["M", "N"],
      "dim_order": [0, 1],
      "dtype": "float16"
    }
  },
  "code": "..."
}
```

### Example 2: Quantized GEMM

```json
{
  "name": "quantized_gemm",
  "axes": {
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 },
    "N_group": { "type": "const", "value": 128 },
    "K_group": { "type": "const", "value": 128 },
  },
  "inputs": {
    "A": {
      "shape": ["M", "K"],
      "dtype": "float16"
    },
    "B": {
      "shape": ["N", "K"],
      "dim_order": [0, 1],
      "dtype": "float16"
    },
    "A_scale": {
      "shape": ["M", "K_group"],
      "dim_order": [0, 1],
      "dtype": "float8_e4m3"
    },
    "B_scale": {
      "shape": ["N_group", "K_group"],
      "dim_order": [0, 1],
      "dtype": "float8_e4m3"
    },
  },
  "outputs": {
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
  "name": "grouped_gemm",
  "axes": {
    "G": { "type": "var" },
    "M": { "type": "var", "parent": "G" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "A": {
      "shape": ["G", "M", "K"],
      "dim_order": [0, 1, 2],
      "dtype": "float16"
    },
    "B": {
      "shape": ["G", "K", "N"],
      "dim_order": [0, 1, 2],
      "dtype": "float16"
    }
  },
  "outputs": {
    "C": {
      "shape": ["G", "M", "N"],
      "dim_order": [0, 1, 2],
      "dtype": "float16"
    }
  },
  "code": "..."
}
```

### Example 4: Quantized Grouped GEMM

```json
{
  "name": "quantized_grouped_gemm",
  "axes": {
    "G": { "type": "var" },
    "M": { "type": "var", "parent": "G" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "A": {
      "shape": ["G", "M", "K"],
      "dim_order": [0, 1, 2],
      "dtype": "float16"
    },
    "B": {
      "shape": ["G", "K", "N"],
      "dim_order": [0, 1, 2],
      "dtype": "float16"
    }
  },
  "outputs": {
    "C": {
      "shape": ["G", "M", "N"],
      "dim_order": [0, 1, 2],
      "dtype": "float16"
    }
  },
  "code": "..."
}
```

### Example 5: RMSNorm

```json
{
  "name": "rmsnorm",
  "axes": {
    "batch_size": { "type": "var" },
    "hidden_size": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "input": {
      "shape": ["batch_size", "hidden_size"],
      "dim_order": [0, 1],
      "dtype": "float16"
    },
    "weight": {
      "shape": ["hidden_size"],
      "dim_order": [0],
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
      "dim_order": [0, 1],
      "dtype": "float16"
    }
  },
  "code": "..."
}
```

### Example 6: Attention

```json
{
  "name": "attention(gqa-4)",
  "axes": {
    "B": { "type": "var" },
    "Q": { "type": "var", "parent": "B" },
    "KV": { "type": "var", "parent": "B" },
    "H_qo": { "type": "var" },
    "H_kv": { "type": "var" },
    "H_r": { "type": "const", "value": 4 },
    "D_qk": { "type": "const", "value": 128 },
    "D_vo": { "type": "const", "value": 128 }
  },
  "assumptions": {
    "H_qo == H_kv * H_r"
  },
  "inputs": {
    "q": {
      "shape": ["B", "Q", "H_qo", "D_qk"],
      "dim_order": [0, 1, 2, 3],
      "dtype": "float16"
    },
    "k": {
      "shape": ["B", "KV", "H_kv", "D_qk"],
      "dim_order": [0, 1, 2, 3],
      "dtype": "float16"
    },
    "v": {
      "shape": ["B", "KV", "H_kv", "D_vo"],
      "dim_order": [0, 1, 2, 3],
      "dtype": "float16"
    }
  },
  "outputs": {
    "out": {
      "shape": ["B", "Q", "H_qo", "D_vo"],
      "dim_order": [0, 1, 2, 3],
      "dtype": "float16"
    },
    "lse": {
      "shape": ["B", "Q", "H_qo"],
      "dim_order": [0, 1, 2],
      "dtype": "float32"
    }
  },
  "code": "..."
}
```
