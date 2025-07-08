# FlashInfer Bench Schema

We organize the FlashInfer Bench dataset into the following three core components:

# Definition

This component provides a formal definition for a specific computational workload encountered in a model's forward pass. It specifies the expected input and output formats. We also include a mathematical specification of the workload in the form of PyTorch code. This serves as both a precise description of the computation and a standard reference implementation.

The Definition directly guides the subsequent Implementation and Dataset components.

**Formal Specification:** [Workload Definition](workload_definition.md)
    

# Implementation


This component represents a single, high-performance implementation of a given workload, contributed by either human experts or autonomous agent systems. An implementation must strictly adhere to the corresponding Definition, including input/output shapes and constant values. Its computation must be functionally equivalent to the mathematical specification.

The implementation is not restricted to any specific language, framework, or platform, but it must provide an entry-point function with a strictly matching signature. Once submitted, implementations are benchmarked. By applying pre-collected evaluation Dataset to the entry point, we verify its correctness and measure its performance speedup, which is then recorded.

**Formal Specification:** [Workload Implementation](workload_implementation.md)
    

# Dataset

This component contains real-world evaluation dataset collected for a specific workload. This data includes the variable dimensions of inputs and outputs and, for cases where latency is correlated with the input distribution, the specific input values themselves.

Dataset is workload-specific, and a single workload should have multiple Data entries. To ensure that benchmark results are relevant to real-world scenarios, we aim to collect this data from production scheduling algorithms and use cases (e.g., SGLang with ShareGPT).

**Formal Specification:** [Workload Dataset](workload_dataset.md)