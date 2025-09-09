# Environment Variables

The following environment variables are used to configure the behavior of the FlashInfer-Bench:

- `FIB_HOME`: The root directory of the FlashInfer-Bench. It will contain these data:
    - The flashinfer trace dataset: kernel definitions, solutions, workloads, and traces.
    - `.cache/`: Store the compilation results.
- `FIB_ENABLE_APPLY`: Enable automatic substitution through the [`flashinfer_bench.apply`](flashinfer_bench.apply) API.
- `FIB_ENABLE_TRACE`: Enable automatic tracing through the [`flashinfer_bench.apply`](flashinfer_bench.apply) API
