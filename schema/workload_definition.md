# Workload Definition

## Overview

This document describes the JSON schema for a **Workload Definition**.

The `Workload Definition` provides a formal, machine-readable specification for a computational workload found in a model's forward pass. It is designed to be the single source of truth that guides both human and agent-based kernel development. Specifically, this schema defines:

1. **Tensor Formats**: The shape, data type (`dtype`).
2. **Dimension Semantics**: The distinction between `constant` dimensions (fixed at compile time) and `variable` dimensions (determined at runtime).
3. **Computational Logic**: A clear, step-by-step **reference implementation** in plain PyTorch, which serves as the official mathematical specification of the workload.

Note that a Workload Definition does not contain specific input *data* for its variable axes. That data is provided by a separate **Workload Evaluation** file, which is used for benchmarking implementations.

## JSON Schema Description

### Top-Level Object Structure

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | string | Yes | A unique, human-readable name for the workload, should include concrete problem information (e.g., `llama3.1_8b_batch_paged_prefill`). |
| `type` | string | Yes | The general workload type name (e.g., `prefill`). Used for grouping and filtering. |
| `description` | string | No | A brief, human-readable description of the workload and its purpose. |
| `axes` | object | Yes | Key-value pairs defining the symbolic dimensions used in tensor shapes. |
| `inputs` | object | Yes | Named input tensors (e.g.,`"A"`,`"B"`). |
| `outputs` | object | Yes | Named output tensors (e.g.,`"C"`). |
| `reference` | string | Yes | The reference implementation in PyTorch, serving as the mathematical specification. |
| `constraints` | array | No | An optional list of assertions describing relationships between axes. |

### `axes` : Dimension Definitions

The `axes` object contains any number of keys, where each key is a symbolic dimension name (e.g., `"M"`, `"N"`, `"K"`), and the value is an object describing its type.

### `type`: `const`

Represents a constant dimension.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | string | Yes | Must be `"const"` |
| `value` | integer | Yes | Constant value of the axis |

Example:

```json
"hidden_size": {
  "type": "const",
  "value": 4096
}

```

### `type`: `var`

Represents a variable axis whose value will be determined by the input data. The `parent` field can be used to indicate hierarchical axis relationships, such as a grouped dimension structure.

| Field | Type | Required | Description | Default |
| --- | --- | --- | --- | --- |
| `type` | string | Yes | Must be `"var"` | — |
| `parent` | string | No | (Optional) name of parent axis for nesting | `null` |

Example:

```json
"sequence_length": {
  "type": "var",
  "parent": "batch_size"
}

```

### `inputs`, `outputs` : Tensor Definitions

These fields describe the input and output tensors of the workload. They contain any number of key-value pairs, where each key is the name of a tensor (e.g., `"A"`, `"B"`, `"C"`). The value is a tensor description:

| Field | Type | Required | Description | Default |
| --- | --- | --- | --- | --- |
| `shape` | array | Yes | List of axis names (strings) | — |
| `dtype` | string | No | Data type of the tensor | `"float16"` |

### `dtype` : Data Types

The following values are allowed for `dtype`:

- `float32`
- `float16`
- `bfloat16`
- `float8_e4m3`
- `float8_e5m2`
- `float4_e2m1`
- `int8`
- `bool`

### Scalar Values

A tensor with an empty shape `[]` represents a scalar value. The scalar input can
not only accept tensor data (torch tensor with shape `[]`), but also scalar data (python int, float, bool).
The scalar output will return a python scalar value.

Example:

```json
"inputs": {
  "logits": {
    "shape": ["batch_size", "vocab_size"],
    "dtype": "float16"
  },
  "temperature": {
    "shape": [],
    "dtype": "float16"
  }
},
"outputs": {
  "probs": {
    "shape": ["batch_size", "vocab_size"],
    "dtype": "float16"
  }
}

```

### `reference` : Reference Implementation

The `reference` field is a string that contains the reference implementation of the workload in plain PyTorch.

- It must contain a global function named `run` as the entry point.
- This code defines the **official mathematical specification** of the workload.
- It should avoid high-level packagings (e.g., **`torch.nn.functional`**) in favor of explicit, step-by-step computations to ensure maximum clarity for all consumers (human or agent).

## Examples

### Example 1: Standard GEMM

```json
{
  "name": "gemm",
  "type": "gemm",
  "description": "A standard GEMM operation (C = A @ B.T).",
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
  "reference": "import torch\n\ndef run(A, B):\n    C = torch.matmul(A, B.T)\n    return {\"C\": C}"
}

```

### Example 2: Quantized GEMM

```json
{
  "name": "quantized_gemm",
  "type": "gemm",
  "description": "A GEMM operation with per-tensor quantized inputs and per-group scaling factors.",
  "axes": {
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 },
    "N_group": { "type": "const", "value": 128 },
    "K_group": { "type": "const", "value": 128 }
  },
  "inputs": {
    "A": {
      "shape": ["M", "K"],
      "dtype": "float8_e4m3"
    },
    "B": {
      "shape": ["N", "K"],
      "dtype": "float8_e4m3"
    },
    "A_scale": {
      "shape": ["M", "K_group"],
      "dtype": "float32"
    },
    "B_scale": {
      "shape": ["N_group", "K_group"],
      "dtype": "float32"
    }
  },
  "outputs": {
    "C": {
      "shape": ["M", "N"],
      "dtype": "bfloat16"
    }
  },
  "reference": "..."
}
```

### Example 3: Grouped GEMM

```json
{
  "name": "grouped_gemm",
  "type": "gemm",
  "description": "A batch of independent GEMM operations, grouped along a 'G' dimension.",
  "axes": {
    "G": { "type": "var" },
    "M": { "type": "var", "parent": "G" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "A": {
      "shape": ["G", "M", "K"],
      "dtype": "float16"
    },
    "B": {
      "shape": ["G", "K", "N"],
      "dtype": "float16"
    }
  },
  "outputs": {
    "C": {
      "shape": ["G", "M", "N"],
      "dtype": "float16"
    }
  },
  "reference": "..."
}
```

### Example 4: Quantized Grouped GEMM

```json
{
  "name": "quantized_grouped_gemm",
  "type": "gemm",
  "description": "A batched GEMM operation where the inputs are quantized, with per-group scaling factors.",
  "axes": {
    "G": { "type": "var" },
    "M": { "type": "var", "parent": "G" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 },
    "K_group": { "type": "const", "value": 128 }
  },
  "inputs": {
    "A": {
      "shape": ["G", "M", "K"],
      "dtype": "float8_e4m3"
    },
    "B": {
      "shape": ["G", "K", "N"],
      "dtype": "float8_e4m3"
    },
    "A_scale": {
      "shape": ["G", "M", "K_group"],
      "dtype": "float32"
    },
    "B_scale": {
      "shape": ["G", "K_group", "N"],
      "dtype": "float32"
    }
  },
  "outputs": {
    "C": {
      "shape": ["G", "M", "N"],
      "dtype": "bfloat16"
    }
  },
  "reference": "..."
}
```

### Example 5: RMSNorm

```json
{
  "name": "rmsnorm",
  "type": "rmsnorm",
  "description": "Root Mean Square Normalization, a common layer normalization variant.",
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
  "reference": "import torch\n\ndef run(input, weight, eps):\n    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)\n    rstd = torch.rsqrt(variance + eps)\n    hidden_states = input * rstd\n    output = (hidden_states * weight).to(weight.dtype)\n    return {\"output\": output}"
}
```

### Example 6: Attention (GQA-4)

```json
{
  "name": "gqa_4_attention",
  "type": "attention",
  "description": "Grouped-Query Attention with a query-to-key-value head ratio of 4.",
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
  "constraints": [
    "H_qo == H_kv * H_r"
  ],
  "inputs": {
    "q": {
      "shape": ["B", "Q", "H_qo", "D_qk"],
      "dtype": "float16"
    },
    "k": {
      "shape": ["B", "KV", "H_kv", "D_qk"],
      "dtype": "float16"
    },
    "v": {
      "shape": ["B", "KV", "H_kv", "D_vo"],
      "dtype": "float16"
    }
  },
  "outputs": {
    "out": {
      "shape": ["B", "Q", "H_qo", "D_vo"],
      "dtype": "float16"
    },
    "lse": {
      "shape": ["B", "Q", "H_qo"],
      "dtype": "float32"
    }
  },
  "reference": "..."
}
```