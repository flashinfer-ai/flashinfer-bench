---
license: apache-2.0
---

# FlashInfer Trace

We provide an official dataset called **FlashInfer Trace** with kernels and workloads in real-world AI system deployment environments. FlashInfer-Bench can use this dataset to measure and compare the performance of kernels. It follows the [FlashInfer Trace Schema](https://bench.flashinfer.ai/docs/flashinfer_trace/flashinfer_trace).

It is organized as follows:

```
flashinfer_trace/   # Here
├── definitions/
└── workloads/

flashinfer-trace/   # On Hugging Face
├── solutions/
└── traces/
```

Example `solutions` and `traces` directories, featuring reference implementations and benchmark logs, are available on Hugging Face: https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace

* Each **Definition** describes a computation task and reference logic.
* Each **Solution** specifies a kernel or agent implementation for a definition.
* Each **Workload** contains the inputs for a definition during real inference.
* Each **Trace** records a benchmark result: input config, performance, correctness, environment, etc.

# Components

## Definition

This component provides a formal definition for a specific computational workload encountered in a model's forward pass. It specifies the expected input and output formats. We also include a mathematical specification of the workload in the form of PyTorch code. This serves as both a precise description of the computation and a standard reference implementation.

The Definition directly guides the subsequent Solution and Trace components.

## Solution

This component represents a single, high-performance solution implementation of a given Definition, contributed by either human experts or autonomous agent systems. A solution must strictly adhere to the corresponding Definition, including input/output shapes and constant values. Its computation must be functionally equivalent to the mathematical specification.

The implementation is not restricted to any specific language, framework, or platform, but it must provide an entry-point function with a strictly matching signature. Once submitted, solutions are benchmarked to generate a Trace. By applying pre-collected input data to the entry point, we verify its correctness and measure its performance metrics.

## Workload

This component encapsulates the concrete input data and configurations used to execute a Definition during real inference scenarios. Each Workload instance contains specific input tensors, shapes, and any relevant parameters that define how the computation should be performed.

## Trace

This component is an atomic and immutable record of a single benchmark run of a Solution. A Trace serves as a detailed log entry, precisely linking a Solution to a Definition for a specific workload configuration (i.e., concrete shapes and input data), and contains the complete evaluation result.

The collection of Traces is the central artifact of the FlashInfer-Bench ecosystem, creating a complete, queryable performance database that enables both high-level analysis and the programmatic discovery of the optimal Solution for any given Definition and environment.
