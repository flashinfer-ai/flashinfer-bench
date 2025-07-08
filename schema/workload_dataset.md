# Workload Dataset

## Overview

This document describes the schema for a **Workload Dataset**.

The `Workload Dataset`  component provides the concrete data required to instantiate a `Workload Definition` for benchmarking. While the `Definition` describes a workload with abstract, variable dimensions (e.g., `batch_size`), the `Dataset` provides specific values for these dimensions and points to the actual tensor data.

The primary goal is to ensure that `Implementations` are benchmarked against data that is representative of real-world scenarios, making the resulting performance metrics truly meaningful. For this reason, `Dataset` s are ideally captured from production systems or carefully designed to stress-test specific edge cases.

## JSON Schema Description

### Top-Level Object Structure

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `name` | string | Yes | A unique, human-readable name for this dataset (e.g., `rmsnorm_sglang_sharegpt_h100`). |
| `workload` | string | Yes | The`name`of the`Workload Definition`this dataset  applies to. |
| `description` | string | No | A brief description or note of this dataset (origin, purpose, etc.). |
| `environment` | object | Yes | An object that logs the single, specific hardware and software environment where this entire dataset is collected on. |
| `entries` | array | Yes | An array of objects, where each object represents a single, distinct dataset entry. |

### **`environment`: Environment Definition Object**

The `environment` object specifies the exact execution environment for all `entries` within this file.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `device` | string | Yes | The name of the hardware device, e.g., `"NVIDIA_H100"`. |
| `libs` | object | Yes | The relevant software libraries and their versions. Keys are library names, and values are version strings. |
| `framework` | string | No | The serving framework that was used. Could be `SGLang`, `vLLM`, `MLC-LLM`, `TRT-LLM`. Not required for framework and model independant data. |
| `model` | string | No | The HuggingFace model name or identifier. Required if `framework` is specified. |

### `entries` : Array of Dataset Cases

Each object in the `entries` array is a self-contained test case.

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `axes` | object | Yes | An object mapping the `var` axis names from the `Workload Definition` to their concrete integer values for this case. |
| `inputs` | object | Yes | An object describing the location and format of the required input tensor data files for this case. |

### **`axes` : Concrete Dimension Values**

It provides a concrete integer value for every dimension that was declared as `type: "var"` in the corresponding `Workload Definition`.

| **Field** | **Type** | **Required** | **Description** |
| --- | --- | --- | --- |
| `key`  | string | Yes | The name of the variable (e.g. “batch_size”). |
| value | integer | Yes | Concrete value of the axis. |

Dimensions declared as `const` in the `Workload Definition` get their values from the definition itself and should not be specified here.

### `inputs` : Data File Descriptors

The keys of this object must correspond to the input tensor names defined in the `Workload Definition`. The value for each key is a "file descriptor" object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `format` | string | Yes | The format of the data file (e.g., `pytorch_pt`, `safetensors`, `numpy_npy`). |
| `path` | string | Yes | The relative path or URI to the binary file containing the tensor data. |
| `tensor_key` | string | No | If the data file is a container for multiple tensors (like `.npz` or `.safetensors`), this key specifies which tensor to load. |

```json
{
	"name": "rmsnorm_sglang_llama_sharegpt_h100"
  "workload": "rmsnorm",
  "description": "RMS Norm evaluation dataset collected by gemini_dataset_agent running Llama-3.1-8B on SGLang, H100, and ShareGPT iuputs."
  "environment": {
    "device": "NVIDIA_H100",
    "libs": {
      "cuda": "12.6",
      "torch": "2.6.0",
      "sglang": "0.4.8"
    },
    "framework": "SGLang",
    "model": "meta-llama/Llama-3.1-8B"
  },
  "entries": [
    {
      "axes": {
        "batch_size": 4
      },
      "inputs": {
        "input": {
          "format": "safetensors",
          "path": "/Upload/rmsnorm_evals/b4_input.safetensors",
          "tensor_key": "input"
        },
        "weight": {
          "format": "safetensors",
          "path": "/Upload/rmsnorm_evals/rmsnorm_weight.safetensors",
          "tensor_key": "weight"
        }
      }
    },
    {
      "axes": {
        "batch_size": 32
      },
      "inputs": {
        "input": {
          "format": "safetensors",
          "path": "/Upload/rmsnorm_evals/b32_input.safetensors",
          "tensor_key": "input"
        },
        "weight": {
          "format": "safetensors",
          "path": "/Upload/rmsnorm_evals/rmsnorm_weight.safetensors",
          "tensor_key": "weight"
        }
      }
    },
    {
      "axes": {
        "batch_size": 128
      },
      "inputs": {
        "input": {
          "format": "safetensors",
          "path": "/Upload/rmsnorm_evals/b128_input.safetensors",
          "tensor_key": "input"
        },
        "weight": {
          "format": "safetensors",
          "path": "/Upload/rmsnorm_evals/rmsnorm_weight.safetensors",
          "tensor_key": "weight"
        }
      }
    }
  ]
}

```