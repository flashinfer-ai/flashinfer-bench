# FlashInfer Trace Schema

FlashInfer-Bench provides a schema for the FlashInfer Trace database. This document includes
the Python API for the schema, including

- The {py:class}`~flashinfer_bench.data.Definition` class, which defines the kernel specification.
- The {py:class}`~flashinfer_bench.data.Solution` class, which defines the kernel implementation.
- The {py:class}`~flashinfer_bench.data.Workload` class, which defines the kernel's input tensors.
- The {py:class}`~flashinfer_bench.data.Trace` class, which defines the kernel execution trace.
- The {py:class}`~flashinfer_bench.data.TraceSet` class, which defines a set of kernel execution traces.

```{toctree}
:maxdepth: 2

schema_definition
schema_solution
schema_workload
schema_trace
schema_trace_set
```
