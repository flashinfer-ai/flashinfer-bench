# Add Reference Tests

Add tests to validate reference implementations in flashinfer-trace. Ground truth is sourced from FlashInfer repository or SGLang when FlashInfer doesn't have the implementation.

## Description

This skill creates test cases under `./tests/references` in the `third_party/flashinfer-trace` dataset to validate that reference implementations in Definition JSON files produce correct outputs. The ground truth comes from:

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

Run `/clone-repos` first to set up the `third_party/` directory with all required repositories.

## What This Skill Does

### Phase 1: Definition Discovery

1. **Load Target Definitions**:
   - If `definition_name` specified: load single definition
   - If `op_type` specified: load all definitions matching op_type from `third_party/flashinfer-trace/definitions/{op_type}/`
   - If `all`: scan all definitions

2. **Check Existing Tests**:
   - Scan `third_party/flashinfer-trace/tests/references/` for existing test files
   - Skip definitions that already have tests (unless force=true)

3. **Parse Definition Schema**:
   - Extract axes (const/var), inputs, outputs
   - Identify required shapes and dtypes
   - Parse reference implementation code

### Phase 2: Ground Truth Discovery

For each definition, locate ground truth implementation:

1. **Search FlashInfer First** (`third_party/flashinfer/`):
   ```
   python/flashinfer/
   ├── attention/          # GQA, MLA implementations
   │   ├── prefill.py
   │   └── decode.py
   ├── norm/               # RMSNorm, LayerNorm
   ├── gemm/               # Matrix multiplication
   └── moe/                # MoE kernels
   ```

2. **Fallback to SGLang** (`third_party/sglang/`):
   ```
   python/sglang/srt/layers/
   ├── attention/          # Attention implementations
   ├── moe/                # MoE implementations
   ├── layernorm.py        # Normalization
   └── linear.py           # GEMM operations
   ```

3. **Ground Truth Source Mapping**:

   | Op Type | FlashInfer Location | SGLang Fallback |
   |---------|---------------------|-----------------|
   | `rmsnorm` | `norm/rmsnorm.py` | `layers/layernorm.py` |
   | `fused_add_rmsnorm` | `norm/fused_add_rmsnorm.py` | `layers/layernorm.py` |
   | `gqa_paged` | `attention/decode.py`, `attention/prefill.py` | `layers/attention/` |
   | `gqa_ragged` | `attention/prefill.py` | `layers/attention/` |
   | `mla_paged` | `attention/mla.py` | `layers/attention/mla_decode.py` |
   | `moe` | `moe/` | `layers/moe/fused_moe.py` |
   | `gemm` | `gemm/` | `torch.nn.functional.linear` |

### Phase 3: Test Generation

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

Output to `third_party/flashinfer-trace/tests/references/`

## Output Structure

```
third_party/flashinfer-trace/tests/references/
├── conftest.py                    # Shared fixtures and utilities
├── test_rmsnorm.py                # RMSNorm tests
├── test_gqa_paged.py              # GQA paged tests
├── test_mla_paged.py              # MLA paged tests
├── test_moe.py                    # MoE tests
└── test_gemm.py                   # GEMM tests
```

## Test File Template

```python
"""Tests for {op_type} reference implementations."""
import pytest
import torch
import json
from pathlib import Path

# Try to import ground truth from FlashInfer
try:
    from flashinfer.attention import batch_decode_with_paged_kv_cache
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

# Fallback to SGLang if needed
try:
    from sglang.srt.layers.attention import decode_attention
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False


DEFINITIONS_DIR = Path(__file__).parent.parent / "definitions"


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
    namespace = {"torch": torch, "math": __import__("math")}
    exec(reference_code, namespace)
    return namespace["run"]


class TestMlaPagedDecodeH16Ckv512Kpe64Ps1:
    """Tests for mla_paged_decode_h16_ckv512_kpe64_ps1."""

    DEFINITION_NAME = "mla_paged_decode_h16_ckv512_kpe64_ps1"
    GROUND_TRUTH_SOURCE = "flashinfer"  # or "sglang"

    # Constant axes from definition
    NUM_QO_HEADS = 16
    HEAD_DIM_CKV = 512
    HEAD_DIM_KPE = 64
    PAGE_SIZE = 1

    @pytest.fixture
    def definition(self):
        return load_definition(self.DEFINITION_NAME)

    @pytest.fixture
    def reference_fn(self, definition):
        return compile_reference(definition["reference"])

    def generate_inputs(self, batch_size, num_pages, device="cuda"):
        """Generate random test inputs matching definition schema."""
        return {
            "q_nope": torch.randn(
                batch_size, self.NUM_QO_HEADS, self.HEAD_DIM_CKV,
                dtype=torch.float16, device=device
            ),
            "q_pe": torch.randn(
                batch_size, self.NUM_QO_HEADS, self.HEAD_DIM_KPE,
                dtype=torch.float16, device=device
            ),
            # ... more inputs based on definition
        }

    @pytest.mark.parametrize("batch_size,num_pages", [
        (1, 4),
        (2, 8),
        (4, 16),
    ])
    def test_output_shape(self, definition, reference_fn, batch_size, num_pages):
        """Test that reference produces correct output shapes."""
        inputs = self.generate_inputs(batch_size, num_pages)
        output, lse = reference_fn(**inputs)

        assert output.shape == (batch_size, self.NUM_QO_HEADS, self.HEAD_DIM_CKV)
        assert lse.shape == (batch_size, self.NUM_QO_HEADS)

    @pytest.mark.skipif(not FLASHINFER_AVAILABLE, reason="FlashInfer not installed")
    @pytest.mark.parametrize("batch_size,num_pages", [
        (1, 4),
        (2, 8),
        (4, 16),
    ])
    def test_numerical_correctness(self, definition, reference_fn, batch_size, num_pages):
        """Test reference matches ground truth."""
        inputs = self.generate_inputs(batch_size, num_pages)

        # Get reference output
        ref_output, ref_lse = reference_fn(**inputs)

        # Get ground truth from FlashInfer
        gt_output, gt_lse = get_flashinfer_ground_truth(**inputs)

        torch.testing.assert_close(ref_output, gt_output, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ref_lse, gt_lse, rtol=1e-3, atol=1e-3)
```

## conftest.py Template

```python
"""Shared test fixtures for reference implementation tests."""
import json
import pytest
import torch
from pathlib import Path


DEFINITIONS_DIR = Path(__file__).parent.parent / "definitions"


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
    namespace = {"torch": torch, "math": __import__("math")}
    exec(reference_code, namespace)
    return namespace["run"]


def assert_close(actual, expected, rtol=1e-3, atol=1e-3):
    """Assert tensors are close within tolerance."""
    if isinstance(actual, tuple):
        for a, e in zip(actual, expected):
            assert_close(a, e, rtol, atol)
    else:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
```

## Implementation Steps

When executing this skill:

1. **Identify definitions to test**:
   ```bash
   ls third_party/flashinfer-trace/definitions/{op_type}/
   ```

2. **Check for existing tests**:
   ```bash
   ls third_party/flashinfer-trace/tests/references/
   ```

3. **For each definition**:
   - Read the definition JSON
   - Identify ground truth source (FlashInfer or SGLang)
   - Generate test class with appropriate fixtures
   - Generate test methods for shape, dtype, and numerical correctness

4. **Create test file**:
   ```bash
   # Create tests directory if needed
   mkdir -p third_party/flashinfer-trace/tests/references/
   ```

5. **Write test file**:
   - If testing multiple definitions of same op_type, combine into one file
   - Each definition gets its own test class

6. **Create/update conftest.py** with shared fixtures

## Running Tests

After generating tests:

```bash
cd third_party/flashinfer-trace

# Run all reference tests
pytest tests/references/ -v

# Run specific test file
pytest tests/references/test_mla_paged.py -v

# Run with GPU
pytest tests/references/ -v --device cuda

# Run with verbose output
pytest tests/references/ -v -s
```

## Error Handling

### Ground Truth Not Available
- **Error**: FlashInfer kernel not found
- **Handling**: Fall back to SGLang; if neither available, mark test as skip with reason

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

# Run tests
cd third_party/flashinfer-trace
pytest tests/references/ -v
```

## Notes

- Tests run on GPU by default; CPU fallback for CI environments
- Tolerance varies by dtype: looser for fp16 (1e-3), stricter for fp32 (1e-5)
- Some kernels may not have FlashInfer ground truth yet
- Test parametrization covers common batch/sequence sizes
- Tests marked with `@pytest.mark.slow` for large sizes

## See Also

- [clone-repos](./clone-repos.md)
- [extract-kernel-definitions](./extract-kernel-definitions.md)
- [workflow](./workflow.md)
