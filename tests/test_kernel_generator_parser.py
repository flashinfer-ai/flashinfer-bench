from __future__ import annotations

import sys
from pathlib import Path

from flashinfer_bench.data import Definition

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "kernel_generator"
sys.path.insert(0, str(EXAMPLES_DIR))
from kernel_generator import KernelGenerator  # noqa: E402


def _make_generator(language: str = "cuda") -> KernelGenerator:
    generator = KernelGenerator.__new__(KernelGenerator)
    generator.language = language
    generator.model_name = "test-model"
    generator.target_gpu = "A800"
    generator.use_ffi = True
    generator.reasoning_effort = "high"
    return generator


def _make_definition() -> Definition:
    return Definition(
        name="test_kernel",
        op_type="test",
        axes={"N": {"type": "const", "value": 1}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"y": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x\n",
    )


def test_cuda_cleaner_parses_xml_and_escaped_xml() -> None:
    generator = _make_generator()

    code = """
```xml
&lt;header_file name='kernel.h'&gt;
#pragma once
void launch();
&lt;/header_file&gt;
&lt;cuda_file name='kernel.cu'&gt;
__global__ void kernel() {}
&lt;/cuda_file&gt;
&lt;cpp_file name='main.cpp'&gt;
void run() {}
&lt;/cpp_file&gt;
```
"""

    cleaned = generator._clean_generated_code(code)

    assert cleaned == {
        "kernel.h": "#pragma once\nvoid launch();",
        "kernel.cu": "__global__ void kernel() {}",
        "main.cpp": "void run() {}",
    }


def test_cuda_cleaner_parses_markdown_file_blocks() -> None:
    generator = _make_generator()

    code = """
File: kernel.h
```cpp
#pragma once
void launch();
```

### kernel.cu
```cuda
__global__ void kernel() {}
```

binding.cpp
```cpp
void run() {}
```
"""

    cleaned = generator._clean_generated_code(code)

    assert cleaned["kernel.h"] == "#pragma once\nvoid launch();"
    assert cleaned["kernel.cu"] == "__global__ void kernel() {}"
    assert cleaned["binding.cpp"] == "void run() {}"


def test_create_solution_uses_detected_host_file_for_entry_point() -> None:
    generator = _make_generator()
    definition = _make_definition()

    solution = generator._create_solution_from_code(
        {
            "kernel.cu": "__global__ void kernel() {}",
            "binding.cpp": "void run() {}",
        },
        definition,
        round_num=1,
    )

    assert solution.spec.entry_point == "binding.cpp::run"
    assert {source.path for source in solution.sources} == {"kernel.cu", "binding.cpp"}


def test_cuda_cleaner_falls_back_to_unlabeled_fenced_blocks() -> None:
    generator = _make_generator()

    code = """
```cuda
__global__ void kernel() {}
```

```cpp
void run() {}
```
"""

    cleaned = generator._clean_generated_code(code)

    assert cleaned["kernel.cu"] == "__global__ void kernel() {}"
    assert cleaned["main.cpp"] == "void run() {}"
