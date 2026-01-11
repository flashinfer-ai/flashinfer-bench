"""
Test script to evaluate LLMs' ability to generate CUDA kernels with TVM FFI bindings.
Tests elementwise add kernel generation across multiple models.
"""

import os
import re
import sys
from pathlib import Path

import pytest
import torch
import tvm_ffi.cpp
from tvm_ffi import Module

from flashinfer_bench.agents.ffi_prompt import FFI_PROMPT_SIMPLE

# System prompt for elementwise add
ELEMENTWISE_ADD_PROMPT = """Write a CUDA kernel function that performs elementwise addition of two tensors.

The function should:
- Take three TensorView arguments: input tensor a, input tensor b, and output tensor c
- Compute c[i] = a[i] + b[i] for all elements
- Support 1D float32 tensors
- Use proper input validation
- Export the function with name "elementwise_add"
"""


def get_model_config(model_name: str):
    """Get API configuration for a given model."""
    # OpenAI models
    openai_key = os.getenv("OPENAI_API_KEY")
    # Anthropic models
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if model_name in ["gpt-5-2025-08-07", "o3", "gpt-5-mini-2025-08-07", "o4-mini-2025-04-16"]:
        return {"provider": "openai", "api_key": openai_key, "model": model_name}
    elif model_name in ["claude-opus-4-1-20250805", "claude-sonnet-4-5-20250805"]:
        return {"provider": "anthropic", "api_key": anthropic_key, "model": model_name}
    else:
        raise ValueError(f"Unknown model: {model_name}")


def call_openai_model(model_name: str, api_key: str, prompt: str) -> str:
    """Call OpenAI API to generate code."""
    import openai

    client = openai.OpenAI(api_key=api_key)

    if model_name in ["o3", "o4-mini-2025-04-16"]:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": ELEMENTWISE_ADD_PROMPT},
                {"role": "user", "content": FFI_PROMPT_SIMPLE},
            ],
        )

    return response.choices[0].message.content


def call_anthropic_model(model_name: str, api_key: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model_name,
        max_tokens=4096,
        system=ELEMENTWISE_ADD_PROMPT,
        messages=[{"role": "user", "content": FFI_PROMPT_SIMPLE}],
    )

    return response.content[0].text


def extract_cuda_code(response: str) -> str:
    patterns = [r"```(?:cpp|cuda|c\+\+)\n(.*?)```", r"```\n(.*?)```"]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            for match in matches:
                if "TVM_FFI" in match or "TensorView" in match:
                    return match.strip()

    return response.strip()


@pytest.fixture
def output_dir():
    """Create and return output directory for test results."""
    dir_path = Path(__file__).parent / "test_results"
    dir_path.mkdir(exist_ok=True)
    return dir_path


@pytest.fixture(
    params=[
        "gpt-5-2025-08-07",
        "o3",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-5-20250805",
        "gpt-5-mini-2025-08-07",
        "o4-mini-2025-04-16",
    ]
)
def model_name(request):
    """Parametrize tests across all supported models."""
    return request.param


def verify_kernel_small_tensor(mod: Module):
    """Test the generated kernel with small tensor."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda")
    b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float32, device="cuda")
    c = torch.empty_like(a)

    mod.elementwise_add(a, b, c)
    expected = a + b

    torch.testing.assert_close(c, expected)


def verify_kernel_large_tensor(mod: Module):
    """Test the generated kernel with large tensor."""
    n = 10000
    a = torch.randn(n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, dtype=torch.float32, device="cuda")
    c = torch.empty_like(a)

    mod.elementwise_add(a, b, c)
    expected = a + b

    torch.testing.assert_close(c, expected)


def generate_code_from_model(model_name: str) -> str:
    """Generate CUDA code from the specified model."""
    config = get_model_config(model_name)

    if config["api_key"] is None:
        pytest.skip(f"API key not available for {config['provider']}")

    full_prompt = ELEMENTWISE_ADD_PROMPT + "\n\n" + FFI_PROMPT_SIMPLE

    if config["provider"] == "openai":
        response = call_openai_model(config["model"], config["api_key"], full_prompt)
    elif config["provider"] == "anthropic":
        response = call_anthropic_model(config["model"], config["api_key"], full_prompt)
    else:
        raise ValueError(f"Unknown provider: {config['provider']}")

    return extract_cuda_code(response)


def save_test_results(
    output_file: Path,
    model_name: str,
    cuda_code: str,
    compilation_success: bool,
    error_msg: str = None,
    small_tensor_passed: bool = False,
    large_tensor_passed: bool = False,
):
    """Save test results to file."""
    with open(output_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"{'='*80}\n\n")
        f.write("Generated Code:\n")
        f.write("=" * 80 + "\n")
        f.write(cuda_code)
        f.write("\n\n")
        f.write("Test Results:\n")
        f.write("=" * 80 + "\n")

        if compilation_success:
            f.write("Compilation: SUCCESS\n")
            f.write(f"Test 1 (small tensor): {'PASS' if small_tensor_passed else 'FAIL'}\n")
            f.write(f"Test 2 (large tensor): {'PASS' if large_tensor_passed else 'FAIL'}\n")
        else:
            f.write("Compilation: FAILED\n")
            if error_msg:
                f.write(f"Error: {error_msg}\n")


def test_llm_code_generation_compilation(model_name, output_dir):
    """Test that the LLM can generate compilable CUDA code with TVM FFI bindings."""
    print(f"\n{'='*80}")
    print(f"Testing model: {model_name}")
    print(f"{'='*80}")

    # Generate code from model
    print(f"Calling API for {model_name}...")
    cuda_code = generate_code_from_model(model_name)
    print(f"Extracted {len(cuda_code)} characters of code")

    # Try to compile the code
    output_file = output_dir / f"{model_name}.txt"
    try:
        mod: Module = tvm_ffi.cpp.load_inline(
            name=f'elementwise_add_{model_name.replace("-", "_")}', cuda_sources=cuda_code
        )
        print("Compilation successful!")

        # Run verification tests
        small_passed = False
        large_passed = False

        try:
            verify_kernel_small_tensor(mod)
            small_passed = True
            print("Small tensor test passed")
        except Exception as e:
            print(f"Small tensor test failed: {e}")

        try:
            verify_kernel_large_tensor(mod)
            large_passed = True
            print("Large tensor test passed")
        except Exception as e:
            print(f"Large tensor test failed: {e}")

        save_test_results(
            output_file,
            model_name,
            cuda_code,
            True,
            small_tensor_passed=small_passed,
            large_tensor_passed=large_passed,
        )

        # Assert both tests passed
        assert small_passed and large_passed, "Kernel verification tests failed"

    except Exception as e:
        print(f"Compilation failed: {e}")
        save_test_results(output_file, model_name, cuda_code, False, str(e))
        pytest.fail(f"Compilation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
