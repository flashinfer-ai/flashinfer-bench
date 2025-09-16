# Welcome to FlashInfer-Bench documentation!

[FlashInfer Blog](https://flashinfer.ai/) | [GitHub](https://github.com/flashinfer-ai/flashinfer-bench/) | [Join Slack](https://join.slack.com/t/flashinfer/shared_invite/zt-379wct3hc-D5jR~1ZKQcU00WHsXhgvtA)

FlashInfer-Bench is a comprehensive benchmark and infrastructure designed to create a "virtuous cycle" where AI can automatically optimize and improve the core GPU kernels of the AI systems it runs on. It provides a systematic framework to identify performance bottlenecks, generate solutions, and deploy them immediately into production.

- **Standardized Schema**: It introduces "FlashInfer Trace," a standardized format to clearly describe GPU kernel workloads, solutions, and results for both AI agents and human engineers.

- **Real-World Benchmarks**: The benchmark datasets are curated from actual, production-grade LLM serving traffic, ensuring that optimizations are relevant and impactful.

- **Seamless Deployment**: It enables "zero-day" integration, allowing new, high-performance kernels to be immediately swapped into live LLM engines with minimal effort.

- **Performance Tracking**: A public leaderboard visualizes and ranks kernel performance, helping developers prioritize optimization efforts on the most critical components.

```{toctree}
:maxdepth: 2
:caption: Get Started

start/quick_start
start/installation
```

```{toctree}
:maxdepth: 2
:caption: Schema

flashinfer_trace/flashinfer_trace
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/bring_your_own_kernel
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/api_apply
api/api_schema
```
