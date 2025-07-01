# FlashInfer Schema Set

## Kernel Description Schema

Kernel Description aims to describe the definition of a kernel, including:
1. The kernel's shape and dtype
2. The kernel's special dimensions (e.g. ragged)
3. The kernel's computation (in PyTorch)

Note the kernel signature does not contain the kernel's exact input data. That will be provided in another file.

### Top-Level Object Structure

| Field            | Type   | Required | Description                             |
|------------------|--------|----------|-----------------------------------------|
| `id`             | string | Yes      | A unique name indicating the kernel.    |
| `axes`           | object | Yes      | Key-value pairs of axis definitions     |
| `inputs`         | object | Yes      | Named input tensors (e.g. `"A"`, `"B"`) |
| `outputs`        | object | Yes      | Named output tensors (e.g. `"C"`)       |
| `reference_code` | string | Yes      | The PyTorch code of the kernel.         |


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

#### Scalar Values

We can define a tensor with a empty shape (`[]`) to represent a scalar value. The scalar input can
not only accept tensor data (torch tensor with shape `[]`), but also scalar data (python int, float, bool).
The scalar output will return a python scalar value.

Example:

```json
"inputs": {
  "A": {
    "shape": ["M", "K"],
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
    "dtype": "float16"
  }
}
```


### `reference_code` Field: PyTorch Code

The `reference_code` field is a string that contains the PyTorch code of the kernel. It follows the [code schema](code.md).


## Examples

### Example 1: Standard GEMM

```json
{
  "id": "gemm",
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
  "reference_code": "..."
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
    },
  },
  "outputs": {
    "C": {
      "shape": ["M", "N"],
      "dtype": "bfloat16"
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
}
```

### Example 6: Attention (GQA-4)

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
  "code": "..."
}
```

### Example 7: PageAttention

## Code Schema

The code should be a Python file, containing a global function `run`, any other Python functions,
and any kernel code.

### Kernel Code

The kernel code can be in
* CUDA
* Triton
* TVM
* torch.compile
* etc.

The code should handle the linking between the Python code and the kernel code.

### Parameters of `run`

The parameters of `run` should follow the [kernel signature](kernel_signature.md), in the order:
1. All input tensors (may also contain scalar values as configurations)
2. All output tensors

`run` returns nothing. The output tensors are allcated in advance and passed in as arguments
to avoid the overhead of allocating them in the function.

The scalar input can take in torch.Tensor with shape `[]`, or a simple Python scalar.

The kernel signature should try to align with [FlashInfer's API](https://docs.flashinfer.ai/).

### Example

```python
import torch

@torch.compile
def rmsnorm(x, eps=1e-5):
    return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

def run(x: torch.Tensor, eps: float, result: torch.Tensor) -> None:
    result.copy_(rmsnorm(x, eps))
```

## Kernel Calling Record Schema

This document describes a JSON schema that defines the structure of benchmark datasets, including:
1. The target model information
2. The inference framework being used
3. The kernel invocation statistics

The dataset schema is used to describe the execution profile of kernels for a specific model running on a particular framework.

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

| Field   | Type    | Required | Description                             |
|---------|---------|----------|-----------------------------------------|
| `id`    | string  | Yes      | The id of the kernel                    |
| `count` | integer | Yes      | Number of times this kernel was invoked |

## Examples

### Example 1: SGLang with Llama Model

```json
{
  "model": "meta-llama/Llama-3.1-8B",
  "framework": "SGLang",
  "kernels": [
    {
      "id": "silu_and_mul",
      "count": 4
    },
    {
      "id": "rmsnorm",
      "count": 2
    }
  ]
}
```

## Benchmark Result Schema

This is the JSON schema that describes the benchmark results of a kernel, including:
1. The device information where the benchmark was executed
2. The kernel specification being benchmarked
3. Benchmark result metrics

### Example 1: Success

```json
{
  "device": {
    "name": "NVIDIA A100-SXM4-80GB"
  },
  "kernel_id": "gemm",
  "source": "flashinfer",
  "compiled": true,
  "time": 0.250,
  "speedup": 2.0,
  "max_diff": 1e-6,
}
```

### Example 2: Compilation Failure

```json
{
  "device": {
    "name": "NVIDIA A100-SXM4-80GB"
  },
  "kernel_id": "gemm",
  "source": "flashinfer",
  "compiled": false,
  "compile_error": "...",
}
```


## Agent Trace Schema

This is the JSON schema for the Agent Inspector, which aims to capture and visualize traces of agent interactions, including:
1. The model
1. The conversation history
2. Tool calls (MCP and others)

The schema represents traces as a list of objects, where each object represents either an LLM call or a tool call in chronological order.

## Examples

### Example 1: Simple LLM Request

```json
{
  "model": "gpt-4o",
  "annotation": "Basic geography question",
  "conversation": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris."
    }
  ],
}
```

### Example 2: LLM Request with Tool Call

```json
{
  "model": "gpt-4o",
  "annotation": "Basic geography question",
  "conversation": [
    {
      "role": "user",
      "content": "What's the weather like in New York?"
    },
    {
      "role": "assistant",
      "content": "I'll check the weather in New York for you.",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"New York, NY\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "Current weather in New York: 72°F, partly cloudy"
    },
    {
      "role": "assistant",
      "content": "The weather in New York is 72°F, partly cloudy."
    },
  ],
}
```
<!--

## JSON Schema Description

### Top-Level Structure

The agent inspector data is represented as a JSON array of trace objects:

```json
[
  {
    "type": "llm_request",
    // ... llm_request fields
  },
  {
    "type": "tool_call",
    // ... tool_call fields
  },
  {
    "type": "mcp",
    // ... mcp fields
  }
]
```

Each trace object must have a `type` field that determines its structure.

### `type`: `llm_request`

Represents an LLM request-response interaction.

| Field          | Type   | Required | Description                                    |
|----------------|--------|----------|------------------------------------------------|
| `type`         | string | Yes      | Must be `"llm_request"`                        |
| `conversation` | array  | Yes      | Complete conversation history sent to the LLM  |
| `response`     | object | Yes      | LLM response following OpenAI format           |
| `model`        | string | Yes      | Model identifier (e.g., "gpt-4", "claude-3")   |
| `annotation`   | string | No       | User-specified annotation for this LLM request |

#### `conversation` Field

The conversation field follows the [OpenAI Chat API message list format](https://platform.openai.com/docs/api-reference/chat/message-list). Each message in the array has:

| Field     | Type   | Required | Description                                   |
|-----------|--------|----------|-----------------------------------------------|
| `role`    | string | Yes      | One of: "system", "user", "assistant", "tool" |
| `content` | string | Yes      | The message content                           |

##### `role` Field

The role field is one of the following values:

- `"system"`
- `"user"`
- `"assistant"`
- `"tool"`

Example:

```json
[
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
    }
]
```

#### `response` Field

The response field follows the [OpenAI Chat Completion response format](https://platform.openai.com/docs/api-reference/chat/object), containing the LLM's output and any tool calls.

SGLang/Vllm's conversation format may slightly differ from OpenAI's. We should convert their's conversations to OpenAI's format.

Now we only consider the following fields:

Example:

```json
{
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The image shows ...",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"New York, NY\"}"
                        }
                    }
                ],
            },
            "finish_reason": "stop"
        }
    ],
}
```

### `type`: `tool_call`

Represents a tool call execution.

| Field        | Type   | Required | Description                                                  |
|--------------|--------|----------|--------------------------------------------------------------|
| `type`       | string | Yes      | Must be `"tool_call"`                                        |
| `tool_name`  | string | Yes      | Name of the called tool                                      |
| `arguments`  | object | Yes      | Tool arguments in json object (name-value pairs)             |
| `result`     | string | Yes      | Tool execution result in string format (potentially in list) |
| `cli_output` | string | No       | Captured stdout/stderr from tool execution                   |

### `type`: `mcp`

Represents an MCP (Model Context Protocol) interaction.

| Field  | Type   | Required | Description     |
|--------|--------|----------|-----------------|
| `type` | string | Yes      | Must be `"mcp"` |

*Note: MCP schema is to be determined (TBD).*

### Example 3: Multi-Agent Conversation

```json
[
  {
    "type": "llm_request",
    "conversation": [
      {
        "role": "user",
        "content": "Analyze this code for security issues."
      },
      {
        "role": "assistant",
        "content": "I'll analyze the code for potential security vulnerabilities."
      }
    ],
    "response": {
      "model": "claude-3-sonnet",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "I'll analyze the code for potential security vulnerabilities."
          },
          "finish_reason": "stop"
        }
      ]
    },
    "model": "claude-3-sonnet",
    "annotation": "Security analysis agent"
  },
  {
    "type": "llm_request",
    "conversation": [
      {
        "role": "system",
        "content": "You are a code optimization specialist."
      },
      {
        "role": "user",
        "content": "Review this code for performance improvements."
      },
      {
        "role": "assistant",
        "content": "I'll review the code for optimization opportunities."
      }
    ],
    "response": {
      "model": "gpt-4o",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "I'll review the code for optimization opportunities."
          },
          "finish_reason": "stop"
        }
      ]
    },
    "model": "gpt-4o",
    "annotation": "Performance optimization agent"
  }
]
```
-->
