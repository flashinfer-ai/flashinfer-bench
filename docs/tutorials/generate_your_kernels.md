# Generate Your Kernels with LLMs

This tutorial shows you how to use Large Language Models (LLMs) to automatically generate optimized kernel solutions for your FlashInfer-Bench definitions. The framework supports multiple LLM providers and can iteratively optimize solutions based on benchmark feedback.

## Overview

The LLM-based kernel generation system consists of:

- **`KernelGenerator`**: Core class that interfaces with LLMs to generate and optimize solutions
- **`example.py`**: Main script demonstrating end-to-end solution generation for all definitions in a traceset
- **Optimization Loop**: Iterative improvement using benchmark feedback to refine solutions
- **Multi-provider Support**: Works with OpenAI, Anthropic, and other compatible API providers

---

## Quick Start

### 1. Environment Setup

First, create a `.env` file in your project root with your LLM API credentials:

```bash
# .env file
LLM_API_KEY=your_api_key_here
BASE_URL=https://api.openai.com/v1  # Optional: custom API endpoint
```

**Supported API providers:**
- OpenAI (GPT-5, GPT-4o, o1, o3)
- Anthropic Claude (via OpenAI-compatible API)
- Google Gemini
- Any OpenAI-compatible API endpoint

### 2. Install Dependencies

```bash
pip install python-dotenv openai flashinfer-bench
```

### 3. Basic Usage

```python
from flashinfer_bench import TraceSet
from kernel_generator import KernelGenerator
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load your traceset
traceset = TraceSet.from_path("/path/to/your/flashinfer-trace")

# Initialize the generator
generator = KernelGenerator(
    model_name="gpt-4o",
    language="triton",
    target_gpu="H100",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("BASE_URL")  # Optional
)

# Generate solution for a specific definition
definition = traceset.definitions["your_kernel_name"]
solution = generator.optimized_generate(
    traceset=traceset,
    definition=definition,
    rounds=5  # Number of optimization rounds
)

# Save the solution
solution_path = f"solutions/{definition.type}/{definition.name}/{solution.name}.json"
with open(solution_path, 'w') as f:
    f.write(solution.to_json())
```

---

## The Example Script Walkthrough

The `examples/example.py` script demonstrates a complete workflow for generating solutions for all definitions in your traceset.

### Key Features

**Automatic Discovery**: Scans your traceset and processes all available definitions

**Error Handling**: Includes retry logic and graceful failure handling

**Progress Tracking**: Shows detailed progress and success/failure statistics

**Organized Output**: Saves solutions in a structured directory layout

### Running the Example

1. **Configure the script** (edit `example.py`):
```python
# Configuration at the top of main()
model_name = "gpt-4o"  # Choose your model
language = "triton"    # "triton", "cuda", or "python"
target_gpu = "H100"    # Target architecture
traceset_path = "/your/path/to/flashinfer-trace"
```

2. **Set up environment**:
```bash
export LLM_API_KEY="your_api_key_here"
export BASE_URL="https://api.openai.com/v1"  # Optional
```

3. **Run the script**:
```bash
cd flashinfer-bench/examples
python example.py
```

### Example Output

```
Loading TraceSet from: /home/user/flashinfer-trace
All definitions found: 15

============================================================
Generating solutions for 15 definitions...
============================================================

[1/15] Processing definition: gqa_paged_decode_h32_kv4_d128
Definition type: gqa
Found 8 workloads for this definition

Attempt 1/2 for gqa_paged_decode_h32_kv4_d128

=== Optimization Round 1/5 ===
Evaluating solution...
Evaluation status: PASSED
Solution PASSED! Speedup: 2.3x

Solution saved to: solutions/gqa/gqa_paged_decode_h32_kv4_d128/gpt-4o_optimized_solution.json
...

============================================================
GENERATION COMPLETE
============================================================
Total definitions processed: 15
Successful generations: 13
Failed generations: 2
Success rate: 86.7%
```

---

## Understanding KernelGenerator

### Initialization Options

```python
generator = KernelGenerator(
    model_name="gpt-4o",           # Model to use
    language="triton",             # Target language: "triton", "cuda", "python"
    target_gpu="H100",             # Target hardware: "H100", "A100", "V100", "RTX4090"
    api_key="your_key",            # API key (or from environment)
    base_url="custom_endpoint"     # Custom API endpoint (optional)
)
```

### Generation Methods

#### Basic Generation (`generate`)
Generates multiple independent solutions:

```python
solutions = generator.generate(definition, pass_k=3)
# Returns 3 independent solution attempts
```

#### Optimized Generation (`optimized_generate`)
Uses iterative optimization with benchmark feedback:

```python
solution = generator.optimized_generate(
    traceset=traceset,
    definition=definition,
    rounds=5  # Maximum optimization rounds
)
```

### The Optimization Loop

The `optimized_generate` method implements a sophisticated feedback loop:

1. **Initial Generation**: Creates a baseline solution from the definition
2. **Benchmark Evaluation**: Tests the solution against a representative workload
3. **Feedback Analysis**: Extracts error messages, performance metrics, and compilation issues
4. **Iterative Improvement**: Generates improved versions based on specific feedback
5. **Early Success**: Returns immediately when a solution passes all tests

```python
# Example optimization process:
# Round 1: Generate initial Triton kernel
# -> Evaluation: COMPILE_ERROR (missing import)
# Round 2: Fix imports, improve memory access patterns  
# -> Evaluation: RUNTIME_ERROR (shape mismatch)
# Round 3: Fix tensor shapes, add bounds checking
# -> Evaluation: PASSED (2.3x speedup) ✓
```

---

## Advanced Configuration

### Custom Model Support

For newer models or custom endpoints:

```python
# For OpenAI o3 models
generator = KernelGenerator(
    model_name="o3",  # Uses different API format
    language="triton",
    target_gpu="B200"
)

# For custom API endpoints
generator = KernelGenerator(
    model_name="custom-model",
    base_url="https://your-custom-api.com/v1",
    api_key="your_custom_key"
)
```

### Language-Specific Generation

```python
# Generate CUDA kernels
cuda_generator = KernelGenerator(
    model_name="gpt-4o",
    language="cuda",
    target_gpu="H100"
)

# Generate Python fallbacks
python_generator = KernelGenerator(
    model_name="gpt-4o", 
    language="python",
    target_gpu="H100"
)
```

### Batch Processing Multiple Tracesets

```python
tracesets = [
    TraceSet.from_path("/path/to/traceset1"),
    TraceSet.from_path("/path/to/traceset2"),
]

for i, traceset in enumerate(tracesets):
    print(f"Processing traceset {i+1}/{len(tracesets)}")
    
    for definition_name, definition in traceset.definitions.items():
        try:
            solution = generator.optimized_generate(
                traceset=traceset,
                definition=definition,
                rounds=3
            )
            # Save solution...
        except Exception as e:
            print(f"Failed to generate {definition_name}: {e}")
```

---

## Output Structure

Generated solutions are saved in a structured directory layout:

```
your-traceset/
└── solutions/
    ├── gqa/                          # Definition type
    │   ├── gqa_paged_decode_h32_kv4_d128/  # Definition name
    │   │   ├── gpt-4o_optimized_solution_1.json
    │   │   └── gpt-4o_optimized_solution_2.json
    │   └── gqa_prefill_h16_d64/
    │       └── claude-4_optimized_solution.json
    ├── gemm/
    │   └── gemm_n_4096_k_14336/
    │       └── o3_optimized_solution.json
    └── mla/
        └── mla_decode_h64_kv8_d256/
            └── gpt-4o_optimized_solution.json
```

Each solution JSON contains:
- **Generated source code** (Triton/CUDA/Python)
- **Build specification** (entry points, target hardware)
- **Metadata** (author model, optimization round)
- **Complete solution definition**

---

## Troubleshooting

### Common Issues

**API Key Not Found**
```
Please set LLM_API_KEY environment variable or modify this script to pass api_key explicitly
```
→ Set `LLM_API_KEY` in your `.env` file or environment

**No Workloads Found**
```
No workloads found for definition 'kernel_name' - SKIPPING
```
→ Ensure you have captured workloads for your definitions (see [tracing tutorial](bring_your_own_kernel.md#component-3-workload))

**Generation Failures**
```
All attempts failed for definition_name - SKIPPING
```
→ Check your API key, model availability, and definition complexity

### Debug Tips

1. **Reduce scope**: Start with a single definition to test your setup
2. **Check logs**: Look for specific error messages in the evaluation feedback
3. **Verify traceset**: Ensure your definitions and workloads are valid
4. **Model selection**: Try different models if one consistently fails

---

## Next Steps

1. **Benchmark your solutions**: Use the generated solutions with the benchmarking system
2. **Apply at runtime**: Enable runtime substitution to use your best solutions in production
3. **Iterate and improve**: Use benchmark results to refine your optimization approach

For benchmarking generated solutions, see the [benchmarking documentation](bring_your_own_kernel.md#component-4-evaluation).

For runtime application, see the [apply tutorial](bring_your_own_kernel.md#end-to-end-apply).
