# FlashInfer Bench Schema

We organize the FlashInfer Bench dataset into the following three core components:

# Definition

This component provides a formal definition for a specific computational workload encountered in a model's forward pass. It specifies the expected input and output formats. We also include a mathematical specification of the workload in the form of PyTorch code. This serves as both a precise description of the computation and a standard reference implementation.

The Definition directly guides the subsequent Solution and Trace components.

**Formal Specification:** [Definition](definition.md)
    

# Solution


This component represents a single, high-performance solution implementation of a given workload, contributed by either human experts or autonomous agent systems. A solution must strictly adhere to the corresponding Definition, including input/output shapes and constant values. Its computation must be functionally equivalent to the mathematical specification.

The implementation is not restricted to any specific language, framework, or platform, but it must provide an entry-point function with a strictly matching signature. Once submitted, solutions are benchmarked to generate a Trace. By applying pre-collected input data to the entry point, we verify its correctness and measure its performance speedup.

**Formal Specification:** [Solution](solution.md)
    

# Trace

This component is an atomic and immutable record of a single benchmark run of a Solution. A Trace serves as a detailed log entry, precisely linking a Solution to a Definition for a specific workload configuration (i.e., concrete shapes and input data), and contains the complete evaluation result.

The collection of Traces is the central artifact of the FlashInfer Bench ecosystem, creating a complete, queryable performance database that enables both high-level analysis and the programmatic discovery of the optimal Solution for any given Definition and environment.

**Formal Specification:** [Trace](trace.md)