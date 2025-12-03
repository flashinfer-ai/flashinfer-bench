# flashinfer_bench.tracing

`flashinfer_bench.tracing` provides tools for tracing kernel executions during LLM inference
and collecting workload traces for the FlashInfer Trace database. This module enables:

1. **Workload Collection**: Capture kernel inputs and execution patterns during runtime
2. **Configurable Tracing**: Control what data to collect and how to deduplicate or filter traces
3. **Filter Policies**: Apply policies to reduce redundant traces and manage dataset size

```{eval-rst}
.. currentmodule:: flashinfer_bench

.. autofunction:: enable_tracing

.. autofunction:: disable_tracing
```
