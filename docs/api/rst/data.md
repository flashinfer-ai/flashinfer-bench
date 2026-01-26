# flashinfer_bench.data

This is the Python API for the FlashInfer Trace Schema. FlashInfer-Bench uses Pydantic to
describe all schemas in the dataset. The following classes are defined:

- The {py:class}`~flashinfer_bench.data.Definition` class, which defines the kernel specification.
- The {py:class}`~flashinfer_bench.data.Solution` class, which defines the kernel implementation.
- The {py:class}`~flashinfer_bench.data.Workload` class, which defines the kernel's input tensors.
- The {py:class}`~flashinfer_bench.data.Trace` class, which defines the kernel execution trace.
- The {py:class}`~flashinfer_bench.data.TraceSet` class, which describes the whole dataset, including
multiple definitions, solutions, workloads, and traces.

```{toctree}
:maxdepth: 2

data_definition
data_solution
data_workload
data_trace
data_trace_set
```
