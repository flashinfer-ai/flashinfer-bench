# flashinfer_bench.agents

`flashinfer_bench.agents` provides tools for kernel agent development and debugging.
This module provides the following tools:

1. **Profiling Tools**: Run NVIDIA Nsight Compute, Compute Sanitizer, etc. on solutions
2. **FFI Prompts**: Provide context about the FlashInfer Bench API for LLM agents

This package also provides JSON Schema version of the tools by calling {py:func}`~flashinfer_bench.agents.function_to_schema` and {py:func}`~flashinfer_bench.agents.get_all_tool_schemas`.

```{eval-rst}
.. currentmodule:: flashinfer_bench.agents

.. autofunction:: flashinfer_bench_run_ncu

.. autofunction:: flashinfer_bench_list_ncu_options

.. autofunction:: function_to_schema

.. autofunction:: get_all_tool_schemas

.. automodule:: flashinfer_bench.agents.ffi_prompt
   :members:
   :no-value:
```
