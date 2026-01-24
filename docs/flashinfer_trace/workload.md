# Workload

## Overview

This document describes the JSON schema for a **Workload**.

A `Workload` defines a concrete, executable instance of a [Definition](definition.md) by binding specific values to all variable axes and specifying the data source for all inputs. It represents the exact configuration under which a `Solution` is benchmarked.

**Storage Format:** In the FlashInfer-Bench dataset, a standalone Workload is stored using the [Trace](trace.md) data structure with only the `definition` and `workload` fields populated, while `solution` and `evaluation` are set to `null`.

In FlashInfer Trace dataset, all workloads of the same definition are stored in a single JSONL file where each line is a `Workload` object.

## JSON Schema Description

### Top-Level Object Structure

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `uuid` | string | Yes | A unique identifier for this workload configuration. |
| `axes` | object | Yes | An object mapping `var` axis names from the `Definition` to their concrete integer values. |
| `inputs` | object | Yes | An object describing the data source for each input. |

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

### Example: RMSNorm Workload

```json
{
  "definition": "rmsnorm_d4096",
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
      },
      "eps": {
        "type": "scalar",
        "value": 1e-6
      }
    }
  },
  "solution": null,
  "evaluation": null
}
```

### Example: GEMM Workload with Random Inputs

```json
{
  "definition": "gemm_n_4096_k_4096",
  "workload": {
    "uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "axes": {
      "M": 1024
    },
    "inputs": {
      "A": {
        "type": "random"
      },
      "B": {
        "type": "random"
      }
    }
  },
  "solution": null,
  "evaluation": null
}
```
