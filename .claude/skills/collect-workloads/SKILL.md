---
name: collect-workloads
description: Auto-collect workloads from SGLang inference runs using FlashInfer logging API. Dumps tensors, sanitizes them according to kernel definitions, and submits PR to flashinfer-trace workload repo.
---

# Collect Workloads

Automatically collect real-world workloads by running SGLang inference with FlashInfer Level 10 logging, then sanitize and submit to the flashinfer-ai/flashinfer-trace HuggingFace dataset repository.

## Description

This skill automates the complete workload collection pipeline:
1. **Setup FlashInfer logging**: Enable Level 10 logging to dump all tensor inputs/outputs
2. **Run SGLang inference**: Execute ShareGPT inference job to capture real workloads
3. **Dump tensors locally**: Collect tensor dumps from FlashInfer logs
4. **Sanitize workloads**: Convert raw tensor dumps to standard workload JSONL format according to kernel definitions
5. **Submit PR**: Create pull request to flashinfer-ai/flashinfer-trace dataset repo

## Usage

```bash
# Collect workloads for specific definitions
/collect-workloads --definition-names mla_paged_decode_h16_ckv512_kpe64_ps1 rmsnorm_h7168

# Collect for all definitions of an op_type
/collect-workloads --op-type mla_paged --model-name deepseek-v3

# Collect for all definitions (comprehensive collection)
/collect-workloads --all --model-name llama-3.1-8b

# Collect without submitting PR (local testing)
/collect-workloads --op-type gqa_paged --submit-pr false

# Custom dataset and sample size
/collect-workloads --op-type rmsnorm --dataset /path/to/custom_sharegpt.jsonl --num-samples 500
```

## Parameters

- `definition_names` (optional): List of specific definition names to collect workloads for (e.g., ["mla_paged_decode_h16_ckv512_kpe64_ps1", "rmsnorm_h7168"])
- `op_type` (optional): Collect workloads for all definitions of a specific op_type (e.g., "mla_paged", "gqa_paged", "rmsnorm")
- `all` (optional): Collect workloads for ALL definitions in definitions directory (default: false)
- `model_name` (required): Model to run inference on (e.g., "deepseek-v3", "llama-3.1-8b", "qwen2.5-7b")
- `dataset` (optional): Path to ShareGPT-format JSONL dataset (default: download from Hugging Face)
- `num_samples` (optional): Number of inference samples to process (default: 100)
- `submit_pr` (optional): Whether to submit PR to flashinfer-trace repo (default: true)
- `pr_title` (optional): Custom PR title (default: auto-generated)
- `pr_branch` (optional): Custom branch name (default: auto-generated from definitions)

## Prerequisites

Run `/clone-repos` first to set up the `tmp/` directory with SGLang and FlashInfer (the `flashinfer_trace/` directory is already part of this repository).

## What This Skill Does

### Phase 1: Environment Setup

1. **Verify Prerequisites**:
   - Check SGLang installation: `python -c "import sglang; print(sglang.__version__)"`
   - Check FlashInfer installation: `python -c "import flashinfer; print(flashinfer.__version__)"`
   - Verify model availability or download if needed

2. **Load Target Definitions**:
   - If `definition_names` specified: load specific definitions
   - If `op_type` specified: load all definitions matching op_type from `flashinfer_trace/definitions/{op_type}/`
   - If `all`: scan all definitions in `flashinfer_trace/definitions/`
   - Parse each definition to understand axes, inputs, and constraints

3. **Prepare Dataset**:
   - If custom dataset path provided, verify file exists
   - Otherwise, download ShareGPT dataset:
     ```python
     from datasets import load_dataset
     dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
     # Save to local JSONL for faster access
     ```
   - Validate dataset format (ShareGPT conversation format)

### Phase 2: FlashInfer Logging Configuration

#### Step 2a: Extract fi_api Patterns from Definition Files

Each definition JSON has `fi_api:<dotted.api.name>` tags that identify which FlashInfer API to capture. Parse these **before** starting the server to build a precise `FLASHINFER_DUMP_INCLUDE` filter:

```python
import json
from pathlib import Path

def get_dump_include_pattern(def_files: list[str]) -> str:
    """
    Parse fi_api tags from definition JSON files and return a
    FLASHINFER_DUMP_INCLUDE glob pattern string.

    Wrapper/class APIs (e.g. BatchMLAPagedAttentionWrapper) are captured
    via their .run() method, so '.run' is appended automatically.
    Plain function APIs (e.g. flashinfer.norm.rmsnorm) are used as-is.
    """
    apis = set()
    for path in def_files:
        defn = json.loads(Path(path).read_text())
        for tag in defn.get("tags", []):
            if tag.startswith("fi_api:"):
                apis.add(tag[len("fi_api:"):])

    patterns = []
    for api in sorted(apis):
        last_component = api.split(".")[-1]
        # Class/Wrapper APIs are invoked through their .run() method
        if last_component[0].isupper():
            patterns.append(f"{api}.run")
        else:
            patterns.append(api)

    return ",".join(patterns)
```

Equivalent one-liner for use in shell scripts:

```bash
# Collect fi_api values from a set of definition files and build the pattern
FI_APIS=$(python3 -c "
import json, sys, pathlib
apis = set()
for f in sys.argv[1:]:
    d = json.loads(pathlib.Path(f).read_text())
    for t in d.get('tags', []):
        if t.startswith('fi_api:'):
            apis.add(t[7:])
patterns = []
for api in sorted(apis):
    last = api.split('.')[-1]
    patterns.append(f'{api}.run' if last[0].isupper() else api)
print(','.join(patterns))
" flashinfer_trace/definitions/mla_paged/*.json flashinfer_trace/definitions/rmsnorm/*.json)

echo "FLASHINFER_DUMP_INCLUDE=$FI_APIS"
# Example output:
# flashinfer.mla.BatchMLAPagedAttentionWrapper.run,flashinfer.norm.fused_add_rmsnorm,flashinfer.norm.rmsnorm
```

#### Step 2b: Set Environment Variables

```bash
# Core configuration
export FLASHINFER_LOGLEVEL=10
export FLASHINFER_DUMP_DIR=./workload_dumps_$(date +%Y%m%d_%H%M%S)
export FLASHINFER_DUMP_MAX_SIZE_GB=50
export FLASHINFER_DUMP_MAX_COUNT=10000

# Format selection (use safetensors for better compatibility)
export FLASHINFER_DUMP_SAFETENSORS=1

# Include only APIs identified from fi_api tags in target definitions.
# This dramatically reduces dump size by skipping unrelated kernels.
export FLASHINFER_DUMP_INCLUDE="$FI_APIS"

# Always exclude constructor and plan/planner calls — they carry no useful
# tensor data and would bloat the dump directory.
export FLASHINFER_DUMP_EXCLUDE="*.__init__,*.plan"
```

**Filter pattern rules**:
- `FLASHINFER_DUMP_INCLUDE` uses comma-separated glob patterns matched against the fully-qualified API call path (e.g. `flashinfer.norm.rmsnorm`)
- `FLASHINFER_DUMP_EXCLUDE` is evaluated after INCLUDE; matching calls are always skipped
- Wrapper `.run()` calls must be matched explicitly (e.g. `flashinfer.mla.BatchMLAPagedAttentionWrapper.run`)
- Wildcards (`*`) match any characters including dots

**Reference**: [FlashInfer Logging Documentation](https://docs.flashinfer.ai/logging.html)

**Important Notes**:
- Level 10 enables "Flight Recorder (Metadata + Tensors)" mode
- Dumps are saved in `.safetensors` format when `FLASHINFER_DUMP_SAFETENSORS=1`
- Each API call creates:
  - `session.jsonl`: Central log with all API events
  - Per-call directories with `metadata.jsonl`, `inputs.safetensors`, `outputs.safetensors`

### Phase 3: SGLang Inference Execution

1. **Launch SGLang Server** with FlashInfer backend:

   ```bash
   # Start SGLang server in background with model
   python -m sglang.launch_server \
     --model-path {model_path} \
     --host 0.0.0.0 \
     --port 30000 \
     --tp 1 \
     --attention-backend flashinfer \
     --disable-cuda-graph \
     --log-level info \
     &

   # Wait for server to be ready
   sleep 30
   ```

   **Key flags**:
   - `--attention-backend flashinfer`: Use FlashInfer kernels (enables logging)
   - `--disable-cuda-graph`: Ensures every kernel call is logged
   - `--tp 1`: Tensor parallelism (adjust based on GPU availability)

2. **Run Inference Requests**:

   ```python
   import requests
   import json
   from tqdm import tqdm

   # Load ShareGPT dataset
   with open(dataset_path) as f:
       conversations = [json.loads(line) for line in f][:num_samples]

   # Process each conversation
   for conv in tqdm(conversations):
       # Format as chat completion
       messages = conv["conversations"]

       # Send to SGLang server
       response = requests.post(
           "http://localhost:30000/v1/chat/completions",
           json={
               "model": model_name,
               "messages": messages,
               "max_tokens": 512,
               "temperature": 0.7,
           }
       )

       # Brief pause between requests
       time.sleep(0.1)
   ```

3. **Shutdown Server**:
   ```bash
   # Gracefully shutdown SGLang server
   pkill -f "sglang.launch_server"
   ```

### Phase 4: Tensor Dump Processing

1. **Locate Dump Directory**:
   ```bash
   DUMP_DIR=$(ls -td workload_dumps_* | head -1)
   echo "Processing dumps from: $DUMP_DIR"
   ```

2. **Parse Session Log**:
   - Read `session.jsonl` to understand API call sequence
   - Extract metadata for each call: function name, timestamp, call ID
   - Map to kernel definitions based on function names

3. **Load Tensor Dumps**:
   ```python
   from safetensors import safe_open

   # For each API call directory
   call_dirs = sorted(Path(DUMP_DIR).glob("call_*"))

   for call_dir in call_dirs:
       # Load metadata
       with open(call_dir / "metadata.jsonl") as f:
           metadata = json.loads(f.readline())

       # Load input tensors
       inputs = {}
       with safe_open(call_dir / "inputs.safetensors", framework="pt") as f:
           for key in f.keys():
               inputs[key] = f.get_tensor(key)

       # Identify which definition this matches
       definition_name = match_to_definition(metadata["function_name"], inputs)
   ```

### Phase 5: Workload Sanitization

**Map raw tensor dumps to workload specifications**:

1. **Match to Definition Schema**:
   - Load definition JSON for target kernel
   - Extract axes (constant and variable)
   - Extract input/output specifications

2. **Extract Variable Axes**:
   ```python
   def extract_axes(definition, tensors):
       """Extract variable axis values from actual tensor shapes."""
       axes = {}

       # Parse definition axes
       for axis_name, axis_spec in definition["axes"].items():
           if axis_spec["type"] == "const":
               # Verify constant matches expected value
               continue
           elif axis_spec["type"] == "var":
               # Extract from tensor shapes
               axes[axis_name] = infer_axis_value(tensors, axis_name, definition)

       return axes
   ```

3. **Handle Tensor Storage**:

   **For small tensors (< 1MB)**: Use `"type": "random"` for reproducibility

   **For large tensors or unique patterns**: Save to safetensors
   ```python
   def create_workload_entry(definition_name, tensors, definition):
       """Create a workload JSONL entry."""
       # Extract axes
       axes = extract_axes(definition, tensors)

       # Prepare inputs
       inputs = {}
       for input_name, input_spec in definition["inputs"].items():
           tensor = tensors[input_name]

           # Decision: random vs saved tensor
           if tensor.numel() < 262144:  # < 1MB for fp16
               inputs[input_name] = {"type": "random"}
           else:
               # Save to safetensors
               safetensor_path = f"tensors/{definition_name}/{uuid}.safetensors"
               save_file({input_name: tensor}, safetensor_path)
               inputs[input_name] = {
                   "type": "safetensors",
                   "path": safetensor_path,
                   "tensor_key": input_name
               }

       # Create workload
       return {
           "definition": definition_name,
           "solution": None,
           "workload": {
               "uuid": str(uuid.uuid4()),
               "axes": axes,
               "inputs": inputs
           },
           "evaluation": None
       }
   ```

4. **Generate Workload JSONL**:
   ```python
   # Group by definition
   workloads_by_def = defaultdict(list)

   for call_dir in processed_calls:
       entry = create_workload_entry(...)
       workloads_by_def[entry["definition"]].append(entry)

   # Write to JSONL files
   for def_name, entries in workloads_by_def.items():
       op_type = load_definition(def_name)["op_type"]
       output_path = f"flashinfer_trace/workloads/{op_type}/{def_name}.jsonl"

       # Append to existing or create new
       with open(output_path, "a") as f:
           for entry in entries:
               f.write(json.dumps(entry) + "\n")
   ```

5. **Deduplication**:
   - Check for duplicate workloads (same axes + input patterns)
   - Only keep unique configurations
   - Prioritize diversity in batch_size, seq_len, and other variable axes

### Phase 6: Submit Pull Request

1. **Clone flashinfer-trace Repository**:
   ```bash
   # Clone HF dataset repo (requires git-lfs)
   cd tmp/
   git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
   cd flashinfer-trace
   ```

2. **Create Branch**:
   ```bash
   # Generate branch name from definitions
   BRANCH_NAME="workloads-$(date +%Y%m%d)-${op_type:-mixed}"
   git checkout -b $BRANCH_NAME
   ```

3. **Copy Workload Files**:
   ```bash
   # Copy new/updated workload JSONL files
   for def_name in $COLLECTED_DEFINITIONS; do
       op_type=$(get_op_type $def_name)
       cp ../../flashinfer-bench/flashinfer_trace/workloads/$op_type/$def_name.jsonl \
          workloads/$op_type/$def_name.jsonl
   done

   # Copy any new tensor safetensors files
   if [ -d "../../flashinfer-bench/flashinfer_trace/workloads/tensors/" ]; then
       cp -r ../../flashinfer-bench/flashinfer_trace/workloads/tensors/* \
             workloads/tensors/
   fi
   ```

4. **Commit Changes**:
   ```bash
   # Stage all workload changes
   git add workloads/

   # Create commit
   git commit -m "Add workloads for ${op_type:-multiple op_types}

   Collected from ${model_name} inference run with ${num_samples} samples.

   Definitions covered:
   $(echo "$COLLECTED_DEFINITIONS" | sed 's/^/- /')

   Stats:
   - Total workloads: ${total_count}
   - Op types: ${op_types_list}
   - Source model: ${model_name}
   - Dataset: ShareGPT (${num_samples} samples)

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
   ```

5. **Push and Create PR**:
   ```bash
   # Push branch
   git push origin $BRANCH_NAME

   # Create PR using Hugging Face CLI (if available) or GitHub CLI
   # Note: HuggingFace datasets use GitHub backend
   gh pr create \
     --repo flashinfer-ai/flashinfer-trace \
     --title "Add ${op_type:-mixed} workloads from ${model_name}" \
     --body "$(cat <<EOF
   ## Summary

   This PR adds real-world workloads collected from ${model_name} inference runs.

   ### Collection Details
   - **Model**: ${model_name}
   - **Dataset**: ShareGPT (${num_samples} samples)
   - **Definitions**: ${#COLLECTED_DEFINITIONS[@]} definitions across ${#op_types[@]} op_types
   - **Total Workloads**: ${total_count}

   ### Definitions Covered
   $(echo "$COLLECTED_DEFINITIONS" | sed 's/^/- `/' | sed 's/$/`/')

   ### Op Types
   $(echo "$op_types_list" | sed 's/^/- /')

   ### Workload Statistics
   \`\`\`
   $(for def in $COLLECTED_DEFINITIONS; do
       echo "$def: $(wc -l < workloads/$op_type/$def.jsonl) workloads"
   done)
   \`\`\`

   ### Validation
   - [x] All workloads follow JSONL format
   - [x] Axes match definition schemas
   - [x] Input specifications are valid
   - [x] Deduplication applied
   - [x] Tensor files saved (if applicable)

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```

## Output Format

### Console Output

During execution, print:

```
=============================================================
FlashInfer Workload Collection
=============================================================

Phase 1: Environment Setup
  ✓ SGLang version: 0.4.1
  ✓ FlashInfer version: 0.6.2
  ✓ Model: deepseek-v3 (loaded)
  ✓ Definitions to collect: 5
    - mla_paged_decode_h16_ckv512_kpe64_ps1
    - mla_paged_prefill_h16_ckv512_kpe64_ps1
    - rmsnorm_h7168
    - fused_add_rmsnorm_h7168
    - moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Phase 2: FlashInfer Logging Configuration
  ✓ Parsed fi_api tags from 5 definition files
  ✓ FLASHINFER_LOGLEVEL=10
  ✓ FLASHINFER_DUMP_DIR=./workload_dumps_20260202_143022
  ✓ FLASHINFER_DUMP_SAFETENSORS=1
  ✓ FLASHINFER_DUMP_INCLUDE=flashinfer.fused_moe.trtllm_fp8_block_scale_moe,flashinfer.mla.BatchMLAPagedAttentionWrapper.run,flashinfer.norm.fused_add_rmsnorm,flashinfer.norm.rmsnorm
  ✓ FLASHINFER_DUMP_EXCLUDE=*.__init__,*.plan

Phase 3: SGLang Inference Execution
  ✓ Server started on port 30000
  ✓ Processing 100 ShareGPT samples
  [████████████████████] 100/100 samples (2m 15s)
  ✓ Server shutdown

Phase 4: Tensor Dump Processing
  ✓ Dump directory: workload_dumps_20260202_143022
  ✓ Found 2,547 API calls
  ✓ Matched calls to definitions:
    - mla_paged_decode_h16_ckv512_kpe64_ps1: 845 calls
    - rmsnorm_h7168: 1,280 calls
    - moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048: 422 calls

Phase 5: Workload Sanitization
  ✓ Extracted axes and inputs
  ✓ Applied deduplication (2547 → 234 unique workloads)
  ✓ Workload distribution:
    - mla_paged_decode_h16_ckv512_kpe64_ps1: 78 workloads
    - rmsnorm_h7168: 102 workloads
    - moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048: 54 workloads
  ✓ Saved to flashinfer_trace/workloads/{op_type}/{def_name}.jsonl

Phase 6: Submit Pull Request
  ✓ Cloned flashinfer-ai/flashinfer-trace
  ✓ Created branch: workloads-20260202-mla_paged
  ✓ Copied workload files
  ✓ Committed changes
  ✓ Pushed to origin
  ✓ Created PR: https://github.com/flashinfer-ai/flashinfer-trace/pull/123

=============================================================
Summary
=============================================================
✓ Collected 234 unique workloads across 3 definitions
✓ PR submitted: https://github.com/flashinfer-ai/flashinfer-trace/pull/123

Local workload files:
  flashinfer_trace/workloads/mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps1.jsonl
  flashinfer_trace/workloads/rmsnorm/rmsnorm_h7168.jsonl
  flashinfer_trace/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl
=============================================================
```

### Generated Files

**Workload JSONL files** (appended to existing):
```
flashinfer_trace/workloads/
├── mla_paged/
│   ├── mla_paged_decode_h16_ckv512_kpe64_ps1.jsonl  (78 new entries)
│   └── mla_paged_prefill_h16_ckv512_kpe64_ps1.jsonl
├── rmsnorm/
│   ├── rmsnorm_h7168.jsonl  (102 new entries)
│   └── fused_add_rmsnorm_h7168.jsonl
└── moe/
    └── moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl  (54 new entries)
```

**Safetensors files** (if large tensors saved):
```
flashinfer_trace/workloads/tensors/
├── mla_paged_decode_h16_ckv512_kpe64_ps1/
│   ├── a1b2c3d4-e5f6-7890-abcd-ef1234567890.safetensors
│   └── ...
└── moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/
    └── ...
```

## Implementation Steps

When executing this skill:

1. **Verify prerequisites**:
   ```bash
   python -c "import sglang, flashinfer; print(f'SGLang: {sglang.__version__}, FlashInfer: {flashinfer.__version__}')"
   ```

2. **Load target definitions**:
   ```bash
   # List definitions to collect for
   ls flashinfer_trace/definitions/{op_type}/*.json
   ```

3. **Download ShareGPT dataset** (if not provided):
   ```python
   from datasets import load_dataset
   dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
   dataset.to_json("sharegpt.jsonl")
   ```

4. **Set FlashInfer logging environment**:

   First, parse `fi_api` tags from the target definition files to build the include filter:
   ```bash
   # Collect definition files for the target op_type(s)
   DEF_FILES=$(ls flashinfer_trace/definitions/${op_type}/*.json)

   # Parse fi_api tags to get precise API filter patterns
   FI_APIS=$(python3 -c "
   import json, sys, pathlib
   apis = set()
   for f in sys.argv[1:]:
       d = json.loads(pathlib.Path(f).read_text())
       for t in d.get('tags', []):
           if t.startswith('fi_api:'):
               apis.add(t[7:])
   patterns = []
   for api in sorted(apis):
       last = api.split('.')[-1]
       patterns.append(f'{api}.run' if last[0].isupper() else api)
   print(','.join(patterns))
   " $DEF_FILES)

   export FLASHINFER_LOGLEVEL=10
   export FLASHINFER_DUMP_DIR=./workload_dumps_$(date +%Y%m%d_%H%M%S)
   export FLASHINFER_DUMP_SAFETENSORS=1
   export FLASHINFER_DUMP_MAX_COUNT=10000
   export FLASHINFER_DUMP_INCLUDE="$FI_APIS"
   export FLASHINFER_DUMP_EXCLUDE="*.__init__,*.plan"
   echo "Capturing APIs: $FI_APIS"
   ```

5. **Launch SGLang server**:
   ```bash
   python -m sglang.launch_server \
     --model-path $MODEL_PATH \
     --host 0.0.0.0 --port 30000 \
     --attention-backend flashinfer \
     --disable-cuda-graph \
     --log-level info &

   # Wait for server ready
   sleep 30
   ```

6. **Run inference requests**:
   ```python
   # Send ShareGPT conversations to server
   for conv in tqdm(conversations[:num_samples]):
       requests.post("http://localhost:30000/v1/chat/completions", json=payload)
   ```

7. **Process tensor dumps**:
   ```python
   # Parse session.jsonl, load safetensors, match to definitions
   workloads = process_dumps(DUMP_DIR, target_definitions)
   ```

8. **Sanitize and save workloads**:
   ```python
   # Write to JSONL format
   for def_name, entries in workloads.items():
       save_workload_jsonl(def_name, entries)
   ```

9. **Submit PR**:
   ```bash
   cd tmp/flashinfer-trace
   git checkout -b $BRANCH_NAME
   # Copy files, commit, push, create PR
   ```

## ShareGPT Dataset Format

The ShareGPT dataset uses conversation format:

```json
{
  "id": "conversation_id",
  "conversations": [
    {"from": "human", "value": "User message"},
    {"from": "gpt", "value": "Assistant response"},
    ...
  ]
}
```

**Alternative datasets**:
- Alpaca format
- OpenAI chat format
- Custom JSONL with similar structure

## Kernel Name Mapping

Map FlashInfer API function names to definitions:

| FlashInfer API | Definition Op Type | Example Definition |
|---------------|-------------------|-------------------|
| `batch_decode_with_paged_kv_cache` | `gqa_paged` | `gqa_paged_decode_h32_kv8_d128_ps1` |
| `batch_prefill_with_paged_kv_cache` | `gqa_paged` | `gqa_paged_prefill_causal_h32_kv8_d128_ps1` |
| `mla_decode_with_paged_kv_cache` | `mla_paged` | `mla_paged_decode_h16_ckv512_kpe64_ps1` |
| `rmsnorm` | `rmsnorm` | `rmsnorm_h7168` |
| `fused_add_rmsnorm` | `rmsnorm` | `fused_add_rmsnorm_h7168` |
| `moe_*` | `moe` | `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` |
| `gemm` / `linear` | `gemm` | `gemm_n6144_k4096` |

## Error Handling

### SGLang Server Fails to Start
- **Error**: Model loading failure or OOM
- **Handling**:
  - Check GPU memory availability
  - Reduce `--tp` (tensor parallelism) if needed
  - Try smaller model variant

### No Tensor Dumps Generated
- **Error**: Dump directory empty after inference
- **Handling**:
  - Verify `FLASHINFER_LOGLEVEL=10` is set
  - Check `FLASHINFER_DUMP_INCLUDE` matches the actual API names — run the `fi_api` parser script against your definition files to regenerate the correct patterns
  - Check `FLASHINFER_DUMP_EXCLUDE` isn't inadvertently matching target APIs
  - Ensure `--attention-backend flashinfer` flag is used
  - Check SGLang is actually using FlashInfer (not falling back to other backends)

### Definition Mismatch
- **Error**: Tensor shapes don't match definition schema
- **Handling**:
  - Print detailed shape comparison
  - Check if model uses different tensor parallel configuration
  - Flag for manual review (may need new definition variant)

### PR Submission Fails
- **Error**: Authentication or push failure
- **Handling**:
  - Verify Hugging Face authentication: `huggingface-cli login`
  - Check write permissions to flashinfer-ai/flashinfer-trace
  - Provide local workload file paths for manual PR creation

## Integration with Other Skills

```bash
# Complete workflow: Extract definitions → Collect workloads → Add tests

# 1. Clone repos
/clone-repos

# 2. Extract kernel definitions from model
/extract-kernel-definitions --model-name deepseek_v3

# 3. Collect workloads for new definitions
/collect-workloads --op-type mla_paged --model-name deepseek-v3

# 4. Add reference tests
/add-reference-tests --op-type mla_paged

# 5. Run tests with real workloads
pytest flashinfer_trace/tests/references/test_mla_paged*.py -v
```

## Advanced Usage

### Collecting for Multiple Models

```bash
# Collect from different models for better workload diversity
/collect-workloads --op-type gqa_paged --model-name llama-3.1-8b --num-samples 100
/collect-workloads --op-type gqa_paged --model-name qwen2.5-7b --num-samples 100
/collect-workloads --op-type gqa_paged --model-name mistral-7b --num-samples 100
```

### Custom Filtering with FLASHINFER_DUMP_INCLUDE / FLASHINFER_DUMP_EXCLUDE

The skill automatically builds `FLASHINFER_DUMP_INCLUDE` by parsing `fi_api` tags from the target definition files. You can also set these manually for ad-hoc collection:

```bash
# Automatically derived from fi_api tags (preferred approach)
export FLASHINFER_DUMP_INCLUDE="flashinfer.mla.BatchMLAPagedAttentionWrapper.run,flashinfer.norm.rmsnorm"
export FLASHINFER_DUMP_EXCLUDE="*.__init__,*.plan"

# Wildcard shorthand — useful for quick ad-hoc runs
export FLASHINFER_DUMP_INCLUDE="*decode*"
export FLASHINFER_DUMP_EXCLUDE="*.__init__,*.plan"

# Only capture Wrapper .run() calls (all attention types)
export FLASHINFER_DUMP_INCLUDE="*Wrapper.run"
export FLASHINFER_DUMP_EXCLUDE="*.__init__,*.plan"

# Combine include and exclude to fine-tune what gets captured
export FLASHINFER_DUMP_INCLUDE="*BatchPrefill*,*rmsnorm*"
export FLASHINFER_DUMP_EXCLUDE="*.__init__,*.plan,*prefill_ragged*"
```

**Filter semantics**:
- Patterns are comma-separated globs matched against the fully-qualified API path
- `FLASHINFER_DUMP_INCLUDE`: only matching calls are recorded (omit to capture everything)
- `FLASHINFER_DUMP_EXCLUDE`: matching calls are always skipped, even if INCLUDE matches
- Always exclude `*.__init__,*.plan` — these carry no tensor data and inflate the dump

### Replay and Validation

After collecting, validate workloads using FlashInfer replay:

```bash
# Replay a dump directory to verify correctness
flashinfer replay --dir workload_dumps_20260202_143022 --compare-outputs
```

## Notes

- FlashInfer Level 10 logging can generate **large amounts of data** (GBs per inference run)
  - Use `FLASHINFER_DUMP_MAX_SIZE_GB` and `FLASHINFER_DUMP_MAX_COUNT` to limit
  - Apply `FLASHINFER_DUMP_INCLUDE` filters to target specific kernels
- Deduplication is **critical** to avoid bloating the dataset with redundant workloads
- Prefer `"type": "random"` for small tensors (faster benchmarking, reproducible)
- Use `"type": "safetensors"` only for large tensors or when patterns are important
- SGLang's `--disable-cuda-graph` flag is **required** to ensure all kernels are logged individually
- The flashinfer-trace repo may require approval for contributions (contact @flashinfer-ai team)

## Troubleshooting

### Issue: Workloads have all random inputs

**Cause**: Tensors were small enough to be marked as random

**Solution**: This is expected behavior; random inputs provide good test coverage while keeping dataset size manageable

### Issue: Variable axes don't match definition

**Cause**: Tensor parallel or model variant differences

**Solution**:
1. Check definition's constant axes match model config
2. May need to create new definition variant for this TP configuration
3. Use `/extract-kernel-definitions` to generate proper definition

### Issue: PR rejected due to dataset quality

**Cause**: Workloads may be too synthetic or redundant

**Solution**:
- Increase `num_samples` for more diversity
- Use different datasets (not just ShareGPT)
- Collect from multiple models
- Ensure deduplication is working properly

## Maintaining This Document

Update this file when:
- FlashInfer logging API changes (new environment variables, file formats)
- Workload schema changes (new InputSpec types)
- HuggingFace dataset repo structure changes
- SGLang server launch parameters change

## References

- [FlashInfer Logging Documentation](https://docs.flashinfer.ai/logging.html)
- [FlashInfer PR #2311 (API Decorator Refactoring)](https://github.com/flashinfer-ai/flashinfer/pull/2311)
- [FlashInfer PR #2206 (Tensor Dump & Replay)](https://github.com/flashinfer-ai/flashinfer/pull/2206)
- [flashinfer-ai/flashinfer-trace Dataset](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
- [SGLang Documentation](https://sgl-project.github.io/)
- [ShareGPT Dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)

## See Also

- [clone-repos](../clone-repos/SKILL.md)
- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md)
- [add-reference-tests](../add-reference-tests/SKILL.md)
