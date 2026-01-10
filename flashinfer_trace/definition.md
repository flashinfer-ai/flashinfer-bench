# Definition

## Overview

This document describes the JSON schema for a kernel **Definition**.

The `Definition` provides a formal, machine-readable specification for a computational workload found in a model's forward pass. It is designed to be the single source of truth that guides both human and agent-based kernel development. Specifically, this schema defines:

1. **Tensor Formats**: The shape, data type (`dtype`).
2. **Dimension Semantics**: The distinction between `constant` dimensions (fixed at compile time) and `variable` dimensions (determined at runtime).
3. **Computational Logic**: A clear, step-by-step **reference implementation** in plain PyTorch, which serves as the official mathematical specification of the kernel.

Note that a `Definition` does not contain specific input *data* for its variable axes. That data is provided by the `workload` field of each `Trace`, which is used for benchmarking `Solution` s.

## JSON Schema Description

### Top-Level Object Structure

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | string | Yes | A unique, human-readable name for the kernel, should include concrete problem information. Naming convention: `{op_type}_{props}_{constants}` (e.g. `gqa_paged_decode_h32_kv8_d128_ps1`). |
| `op_type` | string | Yes | The general compute category. |
| `tags` | array | No | The string tags associated with this definition. Used for grouping and filtering. |
| `description` | string | No | A brief, human-readable description of the definition and its purpose. |
| `axes` | object | Yes | Key-value pairs defining the symbolic dimensions used in tensor shapes. |
| `inputs` | object | Yes | Named input tensors (e.g.,`"A"`,`"B"`). |
| `outputs` | object | Yes | Named output tensors (e.g.,`"C"`). |
| `reference` | string | Yes | The reference implementation in PyTorch, serving as the mathematical specification. |
| `constraints` | array | No | An optional list of assertions describing relationships between axes. |

### `op_type`: Compute Category

`op_type` is a `string` field used for grouping and filtering kernels. It represents the general compute characteristic.

Current supported `op_type`s are:

- Attention: `gqa_ragged`, `gqa_paged`, `mla_ragged`, `mla_paged`
- GEMM: `gemm`
- Misc: `rmsnorm`, `fused_add_rmsnorm`

### `tags` : Additional Attributes

`tags` is an array of strings that attaches searchable attributes to a definition. Tags use **namespaced keys** to keep meanings clear and filterable.

Each tag is either:

- a namespaced key–value string: `"<namespace>:<value>"`, or
- a flag without a value (e.g., `"fused"`).

Controlled namespaces:

- `stage: *` — Which computation stage this definition fits to.

    Examples: `stage: prefill`, `stage: decode`.

- `model:*` — Models known to use this definition (ideally **system-derived** from references/traces).

    Examples: `model:llama-3.1-8b`, `model:deepseek-v3`.

- `quantization:*` — Indicates quantization characteristics. For the simple case, encode the effective dtype.

    Examples: `quantization:float8_e4m3fn`, `quantization:int8`.

- `status:*` — Community/validation status.

    Examples: `status:verified`, `status:draft`, `status:deprecated`.

- `fused` — Flag tag indicating the definition represents a fused kernel.

### `axes` : Dimension Definitions

The `axes` object contains any number of keys, where each key is a symbolic dimension name (e.g., `"M"`, `"N"`, `"K"`), and the value is an object describing its type.

### `type`: `const`

Represents a constant dimension.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | string | Yes | Must be `"const"` |
| `value` | integer | Yes | Constant value of the axis |
| `description` | string | No | Brief description. |

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
| `description` | string | No | Brief description |  |

Example:

```json
"sequence_length": {
  "type": "var",
  "parent": "batch_size"
}

```

### `inputs`, `outputs` : Tensor Definitions

These fields describe the input and output tensors of the kernel. They contain any number of key-value pairs, where each key is the name of a tensor (e.g., `"A"`, `"B"`, `"C"`). The value is a tensor description:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `shape` | array or `null` | Yes | List of axis names (strings). Represents a scalar if `null`. |
| `dtype` | string | Yes | Data type of the tensor |
| `description` | string | No | Brief description. |

### `dtype` : Data Types

The following values are allowed for `dtype`:

- `float32`
- `float16`
- `bfloat16`
- `float8_e4m3fn`
- `float8_e5m2`
- `float4_e2m1`
- `int64`
- `int32`
- `int16`
- `int8`
- `bool`

### Scalar Values and 0-D Tensors
Specifically, a tensor with a shape `[]` (empty array) represents a 0-D tensor.

To represent a scalar value, we use shape `null`. The scalar input must receive a python scalar data (int, float, bool). The scalar output will return a python scalar value.

Example:

```json
"inputs": {
  "logits": {
    "shape": ["batch_size", "vocab_size"],
    "dtype": "float16"
  },
  "temperature": {
    "shape": null,
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

The `reference` field is a string that contains the reference implementation of the kernel in plain PyTorch.

- It must contain a global function named `run` as the entry point.
- This code defines the **official mathematical specification** of the kernel.
- It should avoid high-level packagings (e.g., **`torch.nn.functional`**) in favor of explicit, step-by-step computations to ensure maximum clarity for all consumers (human or agent).

## Examples

### Example 1: Standard GEMM

```json
{
  "name": "gemm_n_4096_k_4096",
  "description": "General matrix multiply (GEMM) C = A @ B.T.",
  "op_type": "gemm",
  "tags": [
    "status:verified",
    "model:llama-3.1-8b"
  ],
  "axes": {
    "M": { "type": "var" },
    "N": { "type": "const", "value": 4096 },
    "K": { "type": "const", "value": 4096 }
  },
  "inputs": {
    "A": { "shape": ["M", "K"], "dtype": "float16" },
    "B": { "shape": ["N", "K"], "dtype": "float16" }
  },
  "outputs": {
    "C": { "shape": ["M", "N"], "dtype": "float16" }
  },
  "reference": "import torch\n\ndef run(A, B):\n    C = torch.matmul(A, B.T)\n    return C"
}

```

### Example 2: Quantized GEMM

```json
{
  "name": "quantized_gemm_n4096_k4096_ng128_kg128",
  "description": "A GEMM operation with per-tensor quantized inputs and per-group scaling factors.",
  "op_type": "gemm",
  "tags": [
      "status:draft",
      "model:some_model",
    "quantization:float8_e4m3fn"
    ]
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
      "dtype": "float8_e4m3fn"
    },
    "B": {
      "shape": ["N", "K"],
      "dtype": "float8_e4m3fn"
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
  "name": "grouped_gemm_n4096_k4096",
  "description": "A batch of independent GEMM operations, grouped along a 'G' dimension.",
  "type": "grouped_gemm",
  "tags": [
    "status:draft",
    "model:some_model"
  ]
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
  "reference": "...",
}
```

### Example 4: Quantized Grouped GEMM

```json
{
  "name": "quantized_grouped_gemm_n4096_k4096_kg128",
  "description": "A batched GEMM operation where the inputs are quantized, with per-group scaling factors.",
  "type": "grouped_gemm",
  "tags": [
    "status:draft",
    "quantization:float8_e4m3fn",
    "model:some_model"
  ]
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
      "dtype": "float8_e4m3fn"
    },
    "B": {
      "shape": ["G", "K", "N"],
      "dtype": "float8_e4m3fn"
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
  "name": "rmsnorm_d4096",
  "description": "Root Mean Square Normalization, a common layer normalization variant.",
  "type": "norm",
  "tags": [
    "status:draft",
    "model:some_model"
  ],
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
      "shape": null,
      "dtype": "float32"
    }
  },
  "outputs": {
    "output": {
      "shape": ["batch_size", "hidden_size"],
      "dtype": "float16"
    }
  },
  "reference": "import torch\n\ndef run(input, weight, eps):\n    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)\n    rstd = torch.rsqrt(variance + eps)\n    hidden_states = input * rstd\n    output = (hidden_states * weight).to(weight.dtype)\n    return output",
}
```

### Example 6: Attention (GQA-4)

```json
{
  "name": "gqa_hr4_dqk128_dvo128",
  "description": "Grouped-Query Attention with a query-to-key-value head ratio of 4.",
  "type": "gqa",
  "tags": [
    "status:draft",
    "model:some_model"
  ]
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
  "reference": "...",
}
```
