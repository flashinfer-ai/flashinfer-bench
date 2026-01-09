# Trace

This document describes the JSON schema for a **Trace**.

A `Trace` is an atomic, immutable record of a **single benchmark run**. It links a specific `Solution` to a specific `Definition`, details the exact `workload` configuration used for the run (i.e., shapes and input data), and records the complete `evaluation` result. The collection of all Trace files forms the database of benchmark results.

## JSON Schema Description

### **Top-Level Object Structure**

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `definition` | string | Yes | The `name` of the `Definition` used in this run. |
| `solution` | string | Yes | The `name` of the `Solution` tested in this run. |
| `workload` | object | Yes | An object describing the specific input configuration for this run.  |
| `evaluation` | object | Yes | An object containing the detailed results of this run. |

### `workload` : Input Shapes and Data

This object provides the concrete data required to instantiate a `Definition`. This data includes the variable dimensions of inputs and outputs and, for cases where latency is correlated with the input distribution, the specific input values themselves.

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `uuid` | string | Yes | A randomly generated UUID for this workload entry. |
| `axes` | object | Yes | An object mapping `var` axis names from the `Definition` to their concrete integer values. |
| `inputs` | object | Yes | An object describing the location and format of the required input tensor data files. |

### `inputs` : Input Descriptor Objects

This object maps **input names** (e.g., `"A"`, `"weight"`, `"mask"`) to **input descriptors** that explain **where the data comes from** and (when necessary) **how it should be generated or loaded**.

Each descriptor **must** contain at least the `type` field. Additional fields become **required or optional** depending on the chosen `type`.

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `type` | string | **Yes** | Data source type. Could be `random`, `scalar`, or `safetensors`. |

Additional fields for type `scalar`:
| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `value` | int, float, bool | **Yes** | The concrete value of the input. |

Additional fields for type `safetensors`:

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `path` | string | **Yes** | Relative path or URI of the `.safetensors` file. |
| `tensor_key` | string | **Yes** | The key inside the safetensors container that holds this tensor. |

### `evaluation` : Benchmark Statistics Summary

This object represents a single, complete benchmark result.

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `status` | string | Yes | The final status of the evaluation run. Has to be one of the following:
`"PASSED"`, `"INCORRECT_SHAPE"`, `"INCORRECT_NUMERICAL"`, `"INCORRECT_DTYPE"`, `"RUNTIME_ERROR"`, `"COMPILE_ERROR"`. |
| `log` | string | Yes | The embedded record of the stdout and stderr of the evaluation run. |
| `correctness` | object | Yes | The summarized correctness results across all entries in the dataset. |
| `performance` | object | Yes | The summarized performance metrics across all entries in the dataset. |
| `environment` | object | Yes | A snapshot of the hardware and software execution environment. |
| `timestamp` | string | Yes | The ISO 8601 timestamp of when this summary was generated. |

### `correctness` : Correctness Summary

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `max_relative_error` | float | Yes | The maximum relative difference found. |
| `max_absolute_error` | float | Yes | The maximum absolute difference found. |

### `performance` : Performance Summary

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `latency_ms` | float | Yes | The mean latency in milliseconds per execution for this implementation. |
| `reference_latency_ms` | float | Yes | The mean latency of the `Definition`'s reference code on the same data/hardware. |
| `speedup_factor` | float | Yes | The calculated speedup (`reference_latency_ms / latency_ms`). |
> Note that it's normal for the speedup factor to be very large since the references are torch only, unoptimized implementations.

### **`environment`: Environment Definition Object**

The `environment` object specifies the exact execution environment for this benchmark run.

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `hardware` | string | Yes | The name of the hardware, e.g., `"NVIDIA_H100"`. |
| `libs` | object | Yes | A snapshot of the relevant software libraries and their versions. Keys are library names, and values are version strings. |

### The `correctness` and `performance` Nullable Table
The `correctness` and `performance` fields are set to be nullable depending on the `status`.
| status | correctness | performance |
| --- | --- | --- |
| PASSED | Required | Required |
| INCORRECT_NUMERICAL | Required | **None** |
| INCORRECT_SHAPE/DTYPE | **None** | **None** |
| RUNTIME_ERROR | **None** | **None** |
| COMPILE_ERROR | **None** | **None** |

### Example: RMSNorm Trace

```python
{
  "definition": "rmsnorm",
  "solution": "rmsnorm_triton_v1",
  "workload": {
    "uuid": "6120f144-b973-4bd9-b884-77ecb132914e",
    "axes": {
      "batch_size": 32
    },
    "inputs": {
      "input": {
        "type": "safetensors",
        "path": "/data/rmsnorm_evals/b32_input.safetensors",
        "tensor_key": "input"
      },
      "weight": {
        "type": "safetensors",
        "path": "/data/rmsnorm_evals/rmsnorm_weight.safetensors",
        "tensor_key": "weight"
      }
    }
  },
  "evaluation": {
    "status": "PASSED",
    "log": "...",
    "correctness": {
      "max_relative_error": 1.15e-05,
      "max_absolute_error": 0.89e-05
    },
    "performance": {
      "latency_ms": 0.008,
      "reference_latency_ms": 0.019,
      "speedup_factor": 2.375
    },
    "environment": {
      "hardware": "NVIDIA_H100",
      "libs": {
        "cuda": "12.6",
        "torch": "2.6.0",
        "triton": "2.4.0"
      }
    },
    "timestamp": "2025-06-27T12:45:00Z"
  }
}
```
