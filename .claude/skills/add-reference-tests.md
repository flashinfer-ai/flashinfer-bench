# Add Reference Tests

Add tests to validate reference implementations in flashinfer-trace. Ground truth is sourced from FlashInfer repository or SGLang when FlashInfer doesn't have the implementation.

## Description

This skill creates test cases under `./tests/references` in the flashinfer-trace dataset to validate that reference implementations in Definition JSON files produce correct outputs. The ground truth comes from:

1. **FlashInfer repository** (preferred): Official optimized GPU kernels
2. **SGLang repository** (fallback): When FlashInfer doesn't have the kernel

## Parameters

- `definition_name` (optional): Specific definition to test (e.g., "mla_paged_decode_h16_ckv512_kpe64_ps1")
- `definitions_dir` (optional): Path to definitions directory (default: from repos_config)
- `op_type` (optional): Test all definitions of a specific op_type (e.g., "mla_paged", "moe", "rmsnorm")
- `repos_config` (optional): Path to repos_config.json from clone-repos skill
- `flashinfer_path` (optional): Direct path to FlashInfer repository
- `sglang_path` (optional): Direct path to SGLang repository
- `output_dir` (optional): Test output directory (default: flashinfer-trace/tests/references)
- `test_sizes` (optional): List of test sizes ["small", "medium", "large"] (default: ["small", "medium"])
- `tolerance` (optional): Numerical tolerance for comparison (default: 1e-3 for fp16, 1e-5 for fp32)

## Usage

```bash
# Test a specific definition
/add-reference-tests --definition-name mla_paged_decode_h16_ckv512_kpe64_ps1

# Test all definitions of an op_type
/add-reference-tests --op-type mla_paged

# Test all new definitions (no existing tests)
/add-reference-tests --definitions-dir ./repos/flashinfer-trace/definitions

# Test with custom tolerance
/add-reference-tests \
  --definition-name rmsnorm_h4096 \
  --tolerance 1e-4
```

## What This Skill Does

### Phase 1: Definition Discovery

1. **Load Target Definitions**:
   - If `definition_name` specified: load single definition
   - If `op_type` specified: load all definitions matching op_type
   - Otherwise: scan all definitions in `definitions_dir`

2. **Check Existing Tests**:
   - Scan `tests/references/` for existing test files
   - Skip definitions that already have tests (unless force=true)
   - Report test coverage statistics

3. **Parse Definition Schema**:
   - Extract axes (const/var), inputs, outputs
   - Identify required shapes and dtypes
   - Parse reference implementation code

### Phase 2: Ground Truth Discovery

For each definition, locate ground truth implementation:

1. **Search FlashInfer First**:
   ```
   flashinfer/python/flashinfer/
   ├── attention/          # GQA, MLA implementations
   │   ├── prefill.py
   │   └── decode.py
   ├── norm/               # RMSNorm, LayerNorm
   ├── gemm/               # Matrix multiplication
   └── moe/                # MoE kernels
   ```

2. **Fallback to SGLang**:
   ```
   sglang/python/sglang/srt/layers/
   ├── attention/          # Attention implementations
   ├── moe/                # MoE implementations
   ├── layernorm.py        # Normalization
   └── linear.py           # GEMM operations
   ```

3. **Map Definition to Ground Truth**:
   | Op Type | FlashInfer Location | SGLang Location |
   |---------|---------------------|-----------------|
   | `gqa_paged` | `attention/decode.py`, `attention/prefill.py` | `layers/attention/` |
   | `mla_paged` | `attention/mla.py` | `layers/attention/mla_decode.py` |
   | `moe` | `moe/` | `layers/moe/fused_moe.py` |
   | `rmsnorm` | `norm/rmsnorm.py` | `layers/layernorm.py` |
   | `gemm` | `gemm/` | `torch.nn.functional.linear` |

### Phase 3: Test Generation

For each definition, generate test file:

1. **Create Test Class**:
   ```python
   class TestMlaPaedDecodeH16Ckv512Kpe64Ps1:
       """Tests for mla_paged_decode_h16_ckv512_kpe64_ps1 definition."""

       @pytest.fixture
       def definition(self):
           return load_definition("mla_paged_decode_h16_ckv512_kpe64_ps1")

       @pytest.fixture
       def ground_truth_fn(self):
           # Return ground truth function from FlashInfer/SGLang
           return get_ground_truth("mla_paged", "decode")
   ```

2. **Generate Test Inputs**:
   ```python
   def generate_inputs(self, definition, size="small"):
       """Generate random inputs matching definition schema."""
       inputs = {}
       for name, spec in definition["inputs"].items():
           shape = resolve_shape(spec["shape"], size)
           dtype = torch_dtype(spec["dtype"])
           inputs[name] = torch.randn(shape, dtype=dtype, device="cuda")
       return inputs
   ```

3. **Create Test Methods**:
   - `test_output_shape`: Verify output shapes match definition
   - `test_output_dtype`: Verify output dtypes match definition
   - `test_numerical_correctness`: Compare reference vs ground truth
   - `test_determinism`: Verify reproducible results
   - `test_edge_cases`: Empty inputs, single element, etc.

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

# Large: Stress tests
LARGE_SIZES = {
    "batch_size": [32, 64],
    "seq_len": [512, 1024, 2048],
    "num_pages": [128, 256],
}
```

### Phase 5: Write Test Files

1. **File Structure**:
   ```
   tests/references/
   ├── conftest.py                    # Shared fixtures
   ├── test_rmsnorm.py                # RMSNorm tests
   ├── test_gqa_paged.py              # GQA paged tests
   ├── test_mla_paged.py              # MLA paged tests
   ├── test_moe.py                    # MoE tests
   └── test_gemm.py                   # GEMM tests
   ```

2. **Test File Template**:
   ```python
   """Tests for {op_type} reference implementations."""
   import pytest
   import torch
   from flashinfer_trace.testing import (
       load_definition,
       compile_reference,
       get_ground_truth,
       assert_close,
   )


   class Test{DefinitionName}:
       """Tests for {definition_name}."""

       DEFINITION_NAME = "{definition_name}"
       GROUND_TRUTH_SOURCE = "{source}"  # "flashinfer" or "sglang"

       @pytest.fixture
       def definition(self):
           return load_definition(self.DEFINITION_NAME)

       @pytest.fixture
       def reference_fn(self, definition):
           return compile_reference(definition["reference"])

       @pytest.fixture
       def ground_truth_fn(self):
           return get_ground_truth(self.GROUND_TRUTH_SOURCE, "{op_type}")

       @pytest.mark.parametrize("batch_size,seq_len", [
           (1, 16),
           (4, 64),
           (8, 128),
       ])
       def test_numerical_correctness(
           self, definition, reference_fn, ground_truth_fn, batch_size, seq_len
       ):
           inputs = self.generate_inputs(definition, batch_size, seq_len)

           ref_outputs = reference_fn(**inputs)
           gt_outputs = ground_truth_fn(**inputs)

           assert_close(ref_outputs, gt_outputs, rtol=1e-3, atol=1e-3)
   ```

## Output Format

### Test File Example

```python
"""Tests for MLA paged decode reference implementations."""
import pytest
import torch
from flashinfer_trace.testing import (
    load_definition,
    compile_reference,
    assert_close,
)

# Import ground truth from FlashInfer
try:
    from flashinfer.attention import mla_decode_with_paged_kv_cache
    GROUND_TRUTH_AVAILABLE = True
except ImportError:
    GROUND_TRUTH_AVAILABLE = False


@pytest.mark.skipif(not GROUND_TRUTH_AVAILABLE, reason="FlashInfer not installed")
class TestMlaPagedDecodeH16Ckv512Kpe64Ps1:
    """Tests for mla_paged_decode_h16_ckv512_kpe64_ps1."""

    DEFINITION_NAME = "mla_paged_decode_h16_ckv512_kpe64_ps1"

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
        """Generate random test inputs."""
        # Calculate derived dimensions
        num_kv_indices = num_pages  # Simplified

        return {
            "q_nope": torch.randn(
                batch_size, self.NUM_QO_HEADS, self.HEAD_DIM_CKV,
                dtype=torch.float16, device=device
            ),
            "q_pe": torch.randn(
                batch_size, self.NUM_QO_HEADS, self.HEAD_DIM_KPE,
                dtype=torch.float16, device=device
            ),
            "ckv_cache": torch.randn(
                num_pages, self.PAGE_SIZE, self.HEAD_DIM_CKV,
                dtype=torch.float16, device=device
            ),
            "kpe_cache": torch.randn(
                num_pages, self.PAGE_SIZE, self.HEAD_DIM_KPE,
                dtype=torch.float16, device=device
            ),
            "kv_indptr": torch.arange(
                batch_size + 1, dtype=torch.int32, device=device
            ) * (num_pages // batch_size),
            "kv_indices": torch.arange(
                num_kv_indices, dtype=torch.int32, device=device
            ),
            "sm_scale": 1.0 / (self.HEAD_DIM_CKV ** 0.5),
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

    @pytest.mark.parametrize("batch_size,num_pages", [
        (1, 4),
        (2, 8),
        (4, 16),
    ])
    def test_numerical_correctness(self, definition, reference_fn, batch_size, num_pages):
        """Test reference matches FlashInfer ground truth."""
        inputs = self.generate_inputs(batch_size, num_pages)

        # Get reference output
        ref_output, ref_lse = reference_fn(**inputs)

        # Get ground truth from FlashInfer
        gt_output, gt_lse = mla_decode_with_paged_kv_cache(
            q_nope=inputs["q_nope"],
            q_pe=inputs["q_pe"],
            kv_cache=(inputs["ckv_cache"], inputs["kpe_cache"]),
            kv_indptr=inputs["kv_indptr"],
            kv_indices=inputs["kv_indices"],
            sm_scale=inputs["sm_scale"],
        )

        assert_close(ref_output, gt_output, rtol=1e-3, atol=1e-3)
        assert_close(ref_lse, gt_lse, rtol=1e-3, atol=1e-3)

    def test_determinism(self, definition, reference_fn):
        """Test that reference is deterministic."""
        inputs = self.generate_inputs(batch_size=2, num_pages=8)

        out1, lse1 = reference_fn(**inputs)
        out2, lse2 = reference_fn(**inputs)

        assert torch.equal(out1, out2)
        assert torch.equal(lse1, lse2)
```

### conftest.py (Shared Fixtures)

```python
"""Shared test fixtures for reference implementation tests."""
import json
import pytest
import torch
from pathlib import Path


DEFINITIONS_DIR = Path(__file__).parent.parent / "definitions"


def load_definition(name: str) -> dict:
    """Load a definition JSON by name."""
    # Search in all op_type subdirectories
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                with open(def_file) as f:
                    return json.load(f)
    raise FileNotFoundError(f"Definition {name} not found")


def compile_reference(reference_code: str):
    """Compile reference implementation to callable function."""
    namespace = {"torch": torch}
    exec(reference_code, namespace)
    return namespace["run"]


def assert_close(actual, expected, rtol=1e-3, atol=1e-3):
    """Assert tensors are close within tolerance."""
    if isinstance(actual, tuple):
        for a, e in zip(actual, expected):
            assert_close(a, e, rtol, atol)
    else:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.fixture
def device():
    """Get test device (CUDA if available)."""
    return "cuda" if torch.cuda.is_available() else "cpu"
```

## Ground Truth Source Mapping

| Op Type | Primary Source | Function/Method | Fallback |
|---------|---------------|-----------------|----------|
| `rmsnorm` | FlashInfer | `flashinfer.norm.rmsnorm` | `torch.nn.functional.rms_norm` |
| `fused_add_rmsnorm` | FlashInfer | `flashinfer.norm.fused_add_rmsnorm` | SGLang layers |
| `gqa_paged` | FlashInfer | `flashinfer.attention.BatchDecodeWithPagedKVCache` | SGLang attention |
| `gqa_ragged` | FlashInfer | `flashinfer.attention.BatchPrefillWithRaggedKVCache` | SGLang attention |
| `mla_paged` | FlashInfer | `flashinfer.attention.mla_decode` | SGLang MLA layers |
| `moe` | SGLang | `sglang.srt.layers.moe.fused_moe` | Custom reference |
| `gemm` | PyTorch | `torch.matmul` | `torch.nn.functional.linear` |

## Requirements

- Python packages:
  - `pytest`
  - `torch` (with CUDA support for GPU tests)
  - `flashinfer` (optional, for ground truth)
- Access to FlashInfer and/or SGLang repositories
- Write access to flashinfer-trace tests directory

## Error Handling

### Ground Truth Not Available
- **Error**: FlashInfer kernel not found
- **Handling**: Fall back to SGLang, or mark test as skip

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
/clone-repos --target-dir ./repos

# Extract definitions
/extract-kernel-definitions --model-name deepseek_v3

# Add tests for new definitions
/add-reference-tests --definitions-dir ./repos/flashinfer-trace/definitions

# Run tests
cd ./repos/flashinfer-trace
pytest tests/references/ -v
```

## Notes

- Tests run on GPU by default; CPU fallback for CI environments
- Tolerance varies by dtype: looser for fp16, stricter for fp32
- Some kernels may not have FlashInfer ground truth yet
- Test parametrization covers common batch/sequence sizes
- Tests marked with `@pytest.mark.slow` for large sizes

## See Also

- [clone-repos](./clone-repos.md)
- [extract-kernel-definitions](./extract-kernel-definitions.md)
- [Definition Schema](../../docs/flashinfer_trace/definition.md)
