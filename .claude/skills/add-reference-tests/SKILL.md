---
name: add-reference-tests
description: Add pytest tests to validate reference implementations in flashinfer_trace against FlashInfer or SGLang ground truth. Use when validating kernel definitions, adding tests for new op_types, or verifying reference implementations are correct.
---

# Add Reference Tests

Add tests to validate reference implementations in `./flashinfer_trace/`. Ground truth is sourced from FlashInfer repository or SGLang when FlashInfer doesn't have the implementation.

## Description

This skill creates test cases under `./flashinfer_trace/tests/references/` to validate that reference implementations in Definition JSON files produce correct outputs. The ground truth comes from:

1. **FlashInfer repository** (preferred): Official optimized GPU kernels in `third_party/flashinfer/`
2. **SGLang repository** (fallback): When FlashInfer doesn't have the kernel, use `third_party/sglang/`

## Usage

```bash
# Test all definitions of a specific op_type
/add-reference-tests --op-type mla_paged
/add-reference-tests --op-type moe
/add-reference-tests --op-type gqa_paged
/add-reference-tests --op-type rmsnorm

# Test a specific definition
/add-reference-tests --definition-name mla_paged_decode_h16_ckv512_kpe64_ps1

# Test all definitions in the definitions directory
/add-reference-tests --all

# Test with custom tolerance
/add-reference-tests --definition-name rmsnorm_h4096 --tolerance 1e-4
```

## Parameters

- `definition_name` (optional): Specific definition to test (e.g., "mla_paged_decode_h16_ckv512_kpe64_ps1")
- `op_type` (optional): Test all definitions of a specific op_type (e.g., "mla_paged", "moe", "rmsnorm")
- `all` (optional): Test all definitions in the definitions directory
- `test_sizes` (optional): List of test sizes ["small", "medium", "large"] (default: ["small", "medium"])
- `tolerance` (optional): Numerical tolerance for comparison (default: 1e-3 for fp16, 1e-5 for fp32)

## Prerequisites

Run `/clone-repos` first to set up the `third_party/` directory with SGLang and FlashInfer repositories. The `flashinfer_trace/` directory is already included in this project.

## What This Skill Does

### Phase 1: Definition Discovery

1. **Load Target Definitions**:
   - If `definition_name` specified: load single definition
   - If `op_type` specified: load all definitions matching op_type from `flashinfer_trace/definitions/{op_type}/`
   - If `all`: scan all definitions

2. **Check Existing Tests**:
   - Scan `flashinfer_trace/tests/references/` for existing test files
   - Skip definitions that already have tests (unless force=true)

3. **Parse Definition Schema**:
   - Extract axes (const/var), inputs, outputs
   - Identify required shapes and dtypes
   - Parse reference implementation code

### Phase 2: Ground Truth Discovery

For each definition, locate ground truth implementation using this priority order:

#### For Model Constants: SGLang Model Config (Required)
- **Location**: `third_party/sglang/python/sglang/srt/models/{model_name}.py`
- **Use for**: Extracting and validating model-specific constant values
- **Examples**:
  - `num_attention_heads`, `num_key_value_heads`, `head_dim`
  - `hidden_size`, `intermediate_size`
  - `num_experts`, `topk`, `n_group`, `topk_group`
- **Important**: Test constants must match SGLang model config

#### For Ground Truth Execution: FlashInfer API (Primary)
- **When**: FlashInfer has the kernel implementation (MOST kernels)
- **Location**: `third_party/flashinfer/python/flashinfer/`
- **Use for**: Running optimized GPU kernel as ground truth
- **Examples**:
  ```python
  import flashinfer
  # GQA decode
  flashinfer.BatchDecodeWithPagedKVCacheWrapper(...)
  # GQA prefill
  flashinfer.BatchPrefillWithPagedKVCacheWrapper(...)
  # MLA
  flashinfer.mla.BatchMLAPagedAttentionWrapper(...)
  # RMSNorm
  flashinfer.norm.rmsnorm(...)
  ```
- **Important**: FlashInfer is the PRIMARY ground truth for correctness validation

#### For Ground Truth Execution: SGLang (Fallback ONLY)
- **When**: FlashInfer does NOT have the kernel (e.g., some MoE variants)
- **Location**: `third_party/sglang/python/sglang/srt/layers/`
- **Use for**: Ground truth when FlashInfer unavailable
- **Examples**:
  ```
  layers/moe/fused_moe.py         # MoE when FlashInfer MoE unavailable
  ```
- **Important**: ONLY use SGLang as ground truth when FlashInfer doesn't support the kernel

#### Ground Truth Source Mapping

| Op Type | Ground Truth Source | FlashInfer API | Fallback (if FlashInfer unavailable) |
|---------|---------------------|----------------|--------------------------------------|
| `rmsnorm` | FlashInfer | `flashinfer.norm.rmsnorm` | N/A (FlashInfer has it) |
| `fused_add_rmsnorm` | FlashInfer | `flashinfer.norm.fused_add_rmsnorm` | N/A (FlashInfer has it) |
| `gqa_paged` | FlashInfer | `flashinfer.BatchDecodeWithPagedKVCacheWrapper`, `flashinfer.BatchPrefillWithPagedKVCacheWrapper` | N/A |
| `gqa_ragged` | FlashInfer | `flashinfer.BatchPrefillWithRaggedKVCacheWrapper` | N/A |
| `mla_paged` | FlashInfer | `flashinfer.mla.BatchMLAPagedAttentionWrapper` | N/A (FlashInfer has it) |
| `moe` | SGLang (fallback) | N/A (FlashInfer MoE may not cover all variants) | `sglang/layers/moe/fused_moe.py` |
| `gemm` | PyTorch | N/A | `torch.nn.functional.linear` |
| `sampling` | FlashInfer | `flashinfer.sampling.*` | N/A |

#### Reference `run()` Function Sources

The reference implementation in Definition JSON should be sourced from FlashInfer unit tests. **SGLang should only be used as a fallback when FlashInfer doesn't have the kernel.**

1. **FlashInfer Unit Tests (REQUIRED - Primary Source)**:
   - **Location**: `third_party/flashinfer/tests/`
   - **Why**: Contains ground-truth vanilla PyTorch implementations
   - **How to find**:
     ```bash
     # Search for reference implementations
     grep -r "def ref_" third_party/flashinfer/tests/
     ```
   - **Kernel to test file mapping**:
     | Kernel Type | Test File | Reference Function |
     |-------------|-----------|-------------------|
     | GQA decode | `test_batch_decode.py` | `ref_attention()` |
     | GQA prefill | `test_batch_prefill.py` | `ref_attention()` |
     | MLA | `test_mla.py` | `ref_mla()` |
     | RMSNorm | `test_norm.py` | `ref_rmsnorm()` |
   - **Important**: Adapt FlashInfer's reference to use constants from SGLang model config

2. **SGLang Vanilla Implementation (FALLBACK ONLY)**:
   - **When to use**: ONLY when FlashInfer does NOT have a unit test for the kernel
   - **Location**: `third_party/sglang/python/sglang/srt/layers/`
   - **Common fallback cases**:
     - MoE kernels: `layers/moe/fused_moe.py`
     - Custom quantized kernels not in FlashInfer
   - **Check before using**:
     ```bash
     # First verify FlashInfer doesn't have a test
     ls third_party/flashinfer/tests/ | grep -i "kernel_name"
     ```

### Phase 3: Test Generation

For each definition, generate test file following the standards below.

## Test File Standards

### File Structure

Each test file should follow this structure:

```python
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

# Ground truth imports (with availability checks)
try:
    import flashinfer
    from flashinfer.xxx import some_kernel
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

# Module-level constants from definition
HIDDEN_SIZE = 7168
NUM_EXPERTS = 256
# ... other constants

TRACE_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_JSONL_PATH = TRACE_ROOT / "workloads" / "op_type" / "definition_name.jsonl"


@torch.no_grad()
def run(...):
    """Reference implementation matching the definition."""
    # Check constants
    assert hidden_size == HIDDEN_SIZE
    ...


def generate_random_inputs(..., device="cuda"):
    """Generate random inputs for testing."""
    ...
    return {...}


def test_correctness(..., atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against ground truth."""
    ...


def main():
    """Run comprehensive tests."""
    ...


if __name__ == "__main__":
    main()
```

### Coding Style Patterns

1. **Constants at Module Level**: Define model-specific constants at the top
   ```python
   # DeepSeek V3/R1 MoE constants
   HIDDEN_SIZE = 7168
   INTERMEDIATE_SIZE = 2048
   NUM_EXPERTS_GLOBAL = 256
   NUM_LOCAL_EXPERTS = 32  # EP=8
   ```

2. **Use `@torch.no_grad()` Decorator**: For all reference implementations and test functions

3. **Input Generator Function**: Separate function `generate_random_inputs(...)` that returns a dict

4. **Test Function Pattern**:
   ```python
   def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
       """Test correctness of reference implementation against ground truth."""
       print(f"\n{'='*60}")
       print(f"Testing {description}: {params}")
       print(f"{'='*60}")

       device = "cuda" if torch.cuda.is_available() else "cpu"
       if device == "cpu":
           print("WARNING: CUDA not available, skipping test")
           return

       # Generate inputs
       inputs = generate_random_inputs(...)

       # Run reference implementation
       print("\nRunning reference implementation...")
       ref_output = run(**inputs)

       # Run ground truth
       print("Running ground truth (FlashInfer/SGLang)...")
       gt_output = ground_truth_fn(**inputs)

       # Compare outputs
       print("\nComparing outputs...")
       # ... detailed comparison
   ```

### Correctness Checking Patterns

#### Standard Tolerance Check (FP16/BF16)

```python
# Convert to float32 for comparison
ref_f32 = ref_output.float()
gt_f32 = gt_output.float()

# Compute detailed error metrics
abs_diff = torch.abs(ref_f32 - gt_f32)
rel_diff = abs_diff / (torch.abs(gt_f32) + 1e-8)

max_abs_diff = abs_diff.max().item()
max_rel_diff = rel_diff.max().item()
mean_abs_diff = abs_diff.mean().item()
mean_rel_diff = rel_diff.mean().item()

print(f"\nOutput tensor comparison:")
print(f"Max absolute difference: {max_abs_diff:.6e}")
print(f"Max relative difference: {max_rel_diff:.6e}")
print(f"Mean absolute difference: {mean_abs_diff:.6e}")
print(f"Mean relative difference: {mean_rel_diff:.6e}")

# Cosine similarity and MSE
cos_sim = torch.nn.functional.cosine_similarity(
    ref_f32.flatten(), gt_f32.flatten(), dim=0
).item()
mse = torch.mean((ref_f32 - gt_f32) ** 2).item()
print(f"Cosine similarity: {cos_sim:.6f}")
print(f"MSE: {mse:.6e}")

# Check tolerance
all_close = torch.allclose(ref_f32, gt_f32, atol=atol, rtol=rtol)
if all_close:
    print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
else:
    print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")
```

#### Hit Ratio Check (for FP8/Quantized Kernels)

For quantized kernels with higher variance, use hit ratio instead of strict allclose:

```python
# Check what percentage of elements pass the tolerance check
left = (ref_f32 - gt_f32).abs()
right = atol + rtol * gt_f32.abs()
ok = left <= right
hit_ratio = ok.float().mean().item()
print(f"\nHit ratio: {hit_ratio * 100:.2f}%  (need >= {percent * 100:.2f}%)")

return hit_ratio >= percent  # e.g., 85%
```

#### Error Location Debugging

When tests fail, show top error locations:

```python
if not all_close:
    flat = abs_diff.flatten()
    k = min(5, flat.numel())
    topv, topi = torch.topk(flat, k)
    print(f"\nTop-{k} absolute error locations:")
    for rank in range(k):
        idx = topi[rank].item()
        # Convert flat index to multi-dimensional
        # ... compute indices
        print(f"  [{indices}]: ref={ref_val:.6e}, gt={gt_val:.6e}, diff={topv[rank].item():.6e}")
```

### Tolerance Guidelines

| Data Type | atol | rtol | Notes |
|-----------|------|------|-------|
| float32 | 1e-5 | 1e-5 | Strictest |
| float16 | 1e-3 | 1e-3 | Standard |
| bfloat16 | 8e-3 | 1e-2 | 0.8% abs, 1% rel |
| float8_e4m3fn | 1e-1 | 2e-1 | Use hit ratio ≥85% |
| nvfp4 | 1e-1 | 2e-1 | Use hit ratio ≥85% |

## Multi-Ground-Truth Testing

### Pattern for Multiple Ground Truths

When both FlashInfer and SGLang implementations are available, test against both:

```python
# Ground truth imports
try:
    from flashinfer.xxx import flashinfer_kernel
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

try:
    from sglang.srt.layers.xxx import sglang_kernel
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False


def test_correctness_vs_flashinfer(...):
    """Test reference against FlashInfer ground truth."""
    if not FLASHINFER_AVAILABLE:
        pytest.skip("FlashInfer not available")

    ref_output = run(**inputs)
    fi_output = flashinfer_kernel(**inputs)

    assert_close(ref_output, fi_output, atol=atol, rtol=rtol)


def test_correctness_vs_sglang(...):
    """Test reference against SGLang ground truth."""
    if not SGLANG_AVAILABLE:
        pytest.skip("SGLang not available")

    ref_output = run(**inputs)
    sg_output = sglang_kernel(**inputs)

    assert_close(ref_output, sg_output, atol=atol, rtol=rtol)


def test_ground_truths_match(...):
    """Test that FlashInfer and SGLang produce consistent results."""
    if not (FLASHINFER_AVAILABLE and SGLANG_AVAILABLE):
        pytest.skip("Both ground truths not available")

    fi_output = flashinfer_kernel(**inputs)
    sg_output = sglang_kernel(**inputs)

    assert_close(fi_output, sg_output, atol=atol, rtol=rtol)
```

### Comprehensive Test Runner

```python
def main():
    """Run comprehensive tests against all available ground truths."""
    print("Testing Reference Implementation")

    # Test configurations
    test_configs = [
        (1, 16),   # Small
        (4, 32),   # Medium
        (8, 64),   # Large
    ]

    results = {"flashinfer": [], "sglang": []}

    for config in test_configs:
        if FLASHINFER_AVAILABLE:
            try:
                ok = test_correctness_vs_flashinfer(*config)
                results["flashinfer"].append(ok)
            except Exception as e:
                print(f"FlashInfer test failed: {e}")
                results["flashinfer"].append(False)

        if SGLANG_AVAILABLE:
            try:
                ok = test_correctness_vs_sglang(*config)
                results["sglang"].append(ok)
            except Exception as e:
                print(f"SGLang test failed: {e}")
                results["sglang"].append(False)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for source, passed_list in results.items():
        if passed_list:
            print(f"  {source}: {sum(passed_list)}/{len(passed_list)} tests passed")
    print(f"{'='*60}")
```

## Standard Test Generation

For each definition, generate test file:

1. **Create Test Class** with:
   - Fixture for loading definition
   - Fixture for compiling reference implementation
   - Fixture for ground truth function

2. **Generate Test Inputs**:
   - Parse definition schema for input shapes and dtypes
   - Generate random tensors matching specifications
   - Handle both constant and variable axes

3. **Create Test Methods**:
   - `test_output_shape`: Verify output shapes match definition
   - `test_output_dtype`: Verify output dtypes match definition
   - `test_numerical_correctness`: Compare reference vs ground truth
   - `test_determinism`: Verify reproducible results

### Phase 4: Test Cases

Generate multiple test cases with varying sizes:

```python
# Small: Quick smoke tests
SMALL_SIZES = {
    "batch_size": [1, 2],
    "seq_len": [1, 16],
    "num_pages": [1, 4],
}

# Medium: Standard tests
MEDIUM_SIZES = {
    "batch_size": [4, 8, 16],
    "seq_len": [64, 128, 256],
    "num_pages": [16, 32, 64],
}
```

### Phase 5: Write Test Files

Output to `flashinfer_trace/tests/references/`

## Output Structure

```
flashinfer_trace/tests/references/
├── conftest.py                    # Shared fixtures and utilities
├── test_rmsnorm.py                # RMSNorm tests
├── test_gqa_paged.py              # GQA paged tests
├── test_mla_paged.py              # MLA paged tests
├── test_moe.py                    # MoE tests
└── test_gemm.py                   # GEMM tests
```

## Test File Template

```python
"""Tests for {definition_name} reference implementation."""
import math
from pathlib import Path

import numpy as np
import torch

# Ground truth imports with availability checks
try:
    import flashinfer
    from flashinfer.xxx import some_kernel
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

try:
    from sglang.srt.layers.xxx import sglang_kernel
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False


# Module-level constants (from definition)
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 1

TRACE_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    """Reference implementation matching the definition schema."""
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    # Check constants
    assert num_qo_heads == NUM_QO_HEADS
    assert num_kv_heads == NUM_KV_HEADS
    assert head_dim == HEAD_DIM
    assert page_size == PAGE_SIZE

    device = q.device

    # Reference computation (pure PyTorch)
    output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    # ... detailed step-by-step computation ...

    return output, lse


def generate_random_inputs(
    batch_size,
    max_seq_len,
    num_attention_heads=NUM_QO_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    page_size=PAGE_SIZE,
    device="cuda",
):
    """Generate random inputs for testing."""
    # Generate tensors matching definition schema
    q = torch.randn(batch_size, num_attention_heads, head_dim, dtype=torch.bfloat16, device=device)
    # ... generate other inputs ...

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "sm_scale": sm_scale,
    }


def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    # Generate inputs
    inputs = generate_random_inputs(batch_size, max_seq_len, device=device)
    print(f"Generated sequences with shapes: q={inputs['q'].shape}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_output, ref_lse = run(**{k: v for k, v in inputs.items() if k != 'extra_keys'})

    # Run FlashInfer ground truth
    print("Running FlashInfer...")
    # ... FlashInfer setup and execution ...

    # Compare outputs
    print("\nComparing outputs...")

    # Convert to float32 for comparison
    ref_f32 = ref_output.float()
    fi_f32 = fi_output.float()

    # Compute errors
    abs_diff = torch.abs(ref_f32 - fi_f32)
    rel_diff = abs_diff / (torch.abs(fi_f32) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"\nOutput tensor comparison:")
    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"Mean relative difference: {mean_rel_diff:.6e}")

    # Cosine similarity and MSE
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), fi_f32.flatten(), dim=0
    ).item()
    mse = torch.mean((ref_f32 - fi_f32) ** 2).item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"MSE: {mse:.6e}")

    # Check if outputs match within tolerance
    all_close = torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol)

    if all_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

        # Show top error locations for debugging
        flat = abs_diff.flatten()
        k = min(5, flat.numel())
        topv, topi = torch.topk(flat, k)
        print(f"\nTop-{k} absolute error locations:")
        for rank in range(k):
            idx = topi[rank].item()
            print(f"  idx={idx}: ref={ref_f32.flatten()[idx].item():.6e}, "
                  f"fi={fi_f32.flatten()[idx].item():.6e}, diff={topv[rank].item():.6e}")

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing Reference Implementation vs FlashInfer")

    # Test configurations
    test_configs = [
        (1, 16),   # Single batch
        (4, 32),   # Small batch
        (8, 64),   # Medium batch
        (16, 128), # Large batch
    ]

    passed = 0
    total = len(test_configs)

    for batch_size, max_seq_len in test_configs:
        try:
            if test_correctness(batch_size, max_seq_len):
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
```

## conftest.py Template

```python
"""Shared test fixtures for reference implementation tests."""
import json
import math
import pytest
import torch
from pathlib import Path


DEFINITIONS_DIR = Path(__file__).parent.parent / "definitions"
WORKLOADS_DIR = Path(__file__).parent.parent / "workloads"


@pytest.fixture
def device():
    """Get test device (CUDA if available)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_definition(name: str) -> dict:
    """Load a definition JSON by name."""
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                with open(def_file) as f:
                    return json.load(f)
    raise FileNotFoundError(f"Definition {name} not found")


def compile_reference(reference_code: str):
    """Compile reference implementation to callable function."""
    namespace = {"torch": torch, "math": math, "np": __import__("numpy")}
    exec(reference_code, namespace)
    return namespace["run"]


def assert_close(actual, expected, rtol=1e-3, atol=1e-3):
    """Assert tensors are close within tolerance."""
    if isinstance(actual, tuple):
        for a, e in zip(actual, expected):
            assert_close(a, e, rtol, atol)
    elif isinstance(actual, dict):
        for k in actual:
            assert_close(actual[k], expected[k], rtol, atol)
    else:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def compute_error_metrics(ref, gt, name="output"):
    """Compute and print detailed error metrics."""
    ref_f32 = ref.float()
    gt_f32 = gt.float()

    abs_diff = torch.abs(ref_f32 - gt_f32)
    rel_diff = abs_diff / (torch.abs(gt_f32) + 1e-8)

    print(f"\n{name} comparison:")
    print(f"  Max absolute difference: {abs_diff.max().item():.6e}")
    print(f"  Max relative difference: {rel_diff.max().item():.6e}")
    print(f"  Mean absolute difference: {abs_diff.mean().item():.6e}")
    print(f"  Mean relative difference: {rel_diff.mean().item():.6e}")

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), gt_f32.flatten(), dim=0
    ).item()
    mse = torch.mean((ref_f32 - gt_f32) ** 2).item()
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  MSE: {mse:.6e}")

    return abs_diff, rel_diff


def check_hit_ratio(ref, gt, atol, rtol, required_percent=0.85):
    """Check if hit ratio meets threshold (for quantized kernels)."""
    ref_f32 = ref.float()
    gt_f32 = gt.float()

    left = (ref_f32 - gt_f32).abs()
    right = atol + rtol * gt_f32.abs()
    ok = left <= right
    hit_ratio = ok.float().mean().item()

    print(f"\nHit ratio: {hit_ratio * 100:.2f}%  (need >= {required_percent * 100:.2f}%)")
    return hit_ratio >= required_percent
```

## Implementation Steps

When executing this skill:

1. **Identify definitions to test**:
   ```bash
   ls flashinfer_trace/definitions/{op_type}/
   ```

2. **Check for existing tests**:
   ```bash
   ls flashinfer_trace/tests/references/
   ```

3. **For each definition**:
   - Read the definition JSON
   - Identify ground truth source (FlashInfer or SGLang)
   - Generate test class with appropriate fixtures
   - Generate test methods for shape, dtype, and numerical correctness

4. **Create test file**:
   ```bash
   # Create tests directory if needed
   mkdir -p flashinfer_trace/tests/references/
   ```

5. **Write test file**:
   - If testing multiple definitions of same op_type, combine into one file
   - Each definition gets its own test class

6. **Create/update conftest.py** with shared fixtures

## Running Tests

After generating tests, run from the project root:

```bash
# Run all reference tests
pytest flashinfer_trace/tests/references/ -v

# Run specific test file
pytest flashinfer_trace/tests/references/test_mla_paged.py -v

# Run with GPU
pytest flashinfer_trace/tests/references/ -v --device cuda

# Run with verbose output
pytest flashinfer_trace/tests/references/ -v -s
```

## Error Handling

### Ground Truth Not Available
- **Error**: Ground truth implementation not found
- **Handling**: Follow priority order:
  1. Check SGLang vanilla implementation
  2. Check SGLang FlashInfer API integration
  3. Check FlashInfer API directly
  4. If none available, mark test as skip with reason

### Definition Parse Error
- **Error**: Invalid definition JSON
- **Handling**: Report validation errors, skip test generation

### Shape Mismatch
- **Error**: Reference output shape doesn't match definition
- **Handling**: Create failing test, flag for investigation

### Numerical Divergence
- **Error**: Reference differs from ground truth beyond tolerance
- **Handling**: Create failing test with detailed diff report

## Integration with Other Skills

```bash
# Complete workflow
/clone-repos

# Extract definitions
/extract-kernel-definitions --model-name deepseek_v3

# Add tests for new definitions
/add-reference-tests --op-type mla_paged
/add-reference-tests --op-type moe

# Run tests from project root
pytest flashinfer_trace/tests/references/ -v
```

## Notes

- Tests run on GPU by default; CPU fallback for CI environments
- Tolerance varies by dtype: looser for fp16 (1e-3), stricter for fp32 (1e-5)
- Some kernels may not have FlashInfer ground truth yet
- Test parametrization covers common batch/sequence sizes
- Tests marked with `@pytest.mark.slow` for large sizes

## See Also

- [clone-repos](../clone-repos/SKILL.md)
- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md)
- [workflow](../workflow.md)
