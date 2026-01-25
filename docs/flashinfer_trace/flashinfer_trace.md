---
license: apache-2.0
---

# FlashInfer Trace Schema

We organize the FlashInfer-Bench dataset into the following four core components:

# Definition

This component provides a formal definition for a specific computational workload encountered in a model's forward pass. It specifies the expected input and output formats. We also include a mathematical specification of the workload in the form of PyTorch code. This serves as both a precise description of the computation and a standard reference implementation.

The Definition directly guides the subsequent Solution and Trace components.

**Formal Specification:** [Definition](definition.md)

# Workload

This component defines a concrete, executable instance of a Definition by binding specific values to all variable axes and specifying the data source for all inputs. A Workload represents the exact configuration under which a Solution is benchmarked.

In the dataset, a standalone Workload is stored using the Trace data structure with only the `definition` and `workload` fields populated (`solution` and `evaluation` are `null`).

**Formal Specification:** [Workload](workload.md)

# Solution

This component represents a single, high-performance solution implementation of a given Definition, contributed by either human experts or autonomous agent systems. A solution must strictly adhere to the corresponding Definition, including input/output shapes and constant values. Its computation must be functionally equivalent to the mathematical specification.

The implementation is not restricted to any specific language, framework, or platform, but it must provide an entry-point function with a strictly matching signature. Once submitted, solutions are benchmarked to generate a Trace. By applying pre-collected input data to the entry point, we verify its correctness and measure its performance metrics.

**Formal Specification:** [Solution](solution.md)

# Trace

This component describes a benchmark result of a solution on a definition with a specific workload. The collection of all Trace files forms the database of benchmark results.

In a `Trace` object, `solution` and `evaluation` are optional. If they are not provided, it describes
a workload entry in the dataset.

**Formal Specification:** [Trace](trace.md)
