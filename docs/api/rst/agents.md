# flashinfer_bench.agents

`flashinfer_bench.agents` provides tools for LLM-based kernel development and debugging.
This module enables LLM agents to:

1. **Profiling Tools**: Run NVIDIA Nsight Compute, Compute Sanitizer, etc. on solutions
2. **Schema Generation**: Generate JSON Schema from function signatures for tool calling
3. **FFI Prompts**: Provide context about the FlashInfer Bench API for LLM agents

```{eval-rst}
.. currentmodule:: flashinfer_bench.agents

.. autofunction:: flashinfer_bench_run_ncu

.. autofunction:: flashinfer_bench_list_ncu_options

.. autofunction:: function_to_schema

.. autofunction:: get_all_tool_schemas

.. autodata:: FFI_PROMPT

.. autodata:: FFI_PROMPT_SIMPLE
```
