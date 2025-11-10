"""
Test script to evaluate LLMs' ability to generate CUDA kernels with TVM FFI bindings.
Tests elementwise add kernel generation across multiple models.
"""

import os
import re
from pathlib import Path

import torch
import tvm_ffi.cpp
from dotenv import load_dotenv
from ffi_prompt import FFI_PROMPT_SIMPLE
from tvm_ffi import Module

load_dotenv()


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


def test_kernel(mod: Module, test_name: str) -> bool:
    """Test the generated kernel with simple test cases."""
    try:
        print(f"\n  Running {test_name}...")

        if test_name == "test_small":
            # Test 1: Small tensor
            a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda")
            b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float32, device="cuda")
            c = torch.empty_like(a)

            mod.elementwise_add(a, b, c)
            expected = a + b

            torch.testing.assert_close(c, expected)
            print(f"  ✓ {test_name} passed")
            return True

        elif test_name == "test_large":
            # Test 2: Larger tensor
            n = 10000
            a = torch.randn(n, dtype=torch.float32, device="cuda")
            b = torch.randn(n, dtype=torch.float32, device="cuda")
            c = torch.empty_like(a)

            mod.elementwise_add(a, b, c)
            expected = a + b

            torch.testing.assert_close(c, expected)
            print(f"  ✓ {test_name} passed")
            return True

    except Exception as e:
        print(f"  ✗ {test_name} failed: {e}")
        return False


def test_model(model_name: str, output_dir: Path):
    print(f"\n{'='*80}")
    print(f"Testing model: {model_name}")
    print(f"{'='*80}")

    try:
        config = get_model_config(model_name)

        full_prompt = ELEMENTWISE_ADD_PROMPT + "\n\n" + FFI_PROMPT_SIMPLE

        print(f"Calling {config['provider']} API...")
        if config["provider"] == "openai":
            response = call_openai_model(config["model"], config["api_key"], full_prompt)
        elif config["provider"] == "anthropic":
            response = call_anthropic_model(config["model"], config["api_key"], full_prompt)
        else:
            raise ValueError(f"Unknown provider: {config['provider']}")

        print(f"Received response from {model_name}")

        cuda_code = extract_cuda_code(response)
        print(f"Extracted {len(cuda_code)} characters of code")

        output_file = output_dir / f"{model_name}.txt"
        with open(output_file, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"{'='*80}\n\n")
            f.write("Generated Code:\n")
            f.write("=" * 80 + "\n")
            f.write(cuda_code)
            f.write("\n\n")

        print(f"Saved response to {output_file}")

        try:
            mod: Module = tvm_ffi.cpp.load_inline(
                name=f'elementwise_add_{model_name.replace("-", "_")}', cuda_sources=cuda_code
            )
            print("Compilation successful!")

            # Run tests
            print("\nRunning tests...")
            test1_passed = test_kernel(mod, "test_small")
            test2_passed = test_kernel(mod, "test_large")

            with open(output_file, "a") as f:
                f.write("Test Results:\n")
                f.write("=" * 80 + "\n")
                f.write(f"Compilation: SUCCESS\n")
                f.write(f"Test 1 (small tensor): {'PASS' if test1_passed else 'FAIL'}\n")
                f.write(f"Test 2 (large tensor): {'PASS' if test2_passed else 'FAIL'}\n")

            all_passed = test1_passed and test2_passed
            status = "ALL TESTS PASSED" if all_passed else "TESTS FAILED"
            print(f"\n{status}")

            return {
                "model": model_name,
                "compilation": "success",
                "test_small": test1_passed,
                "test_large": test2_passed,
                "all_passed": all_passed,
            }

        except Exception as e:
            print(f"Compilation failed: {e}")
            with open(output_file, "a") as f:
                f.write("Test Results:\n")
                f.write("=" * 80 + "\n")
                f.write(f"Compilation: FAILED\n")
                f.write(f"Error: {str(e)}\n")

            return {
                "model": model_name,
                "compilation": "failed",
                "error": str(e),
                "all_passed": False,
            }

    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        return {"model": model_name, "compilation": "error", "error": str(e), "all_passed": False}


def main():
    models = [
        "gpt-5-2025-08-07",
        "o3",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-5-20250805",
        "gpt-5-mini-2025-08-07",
        "o4-mini-2025-04-16",
    ]

    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Testing LLM FFI Code Generation")
    print("=" * 80)
    print(f"Models to test: {len(models)}")
    print(f"Output directory: {output_dir}")

    results = []

    for idx, model_name in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] Testing {model_name}...")
        result = test_model(model_name, output_dir)
        results.append(result)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    for result in results:
        model = result["model"]
        status = "✓ PASS" if result.get("all_passed", False) else "✗ FAIL"
        print(f"{model:30} {status}")
        if "error" in result:
            print(f"  Error: {result['error'][:100]}")

    total_passed = sum(1 for r in results if r.get("all_passed", False))
    print(f"\nTotal: {total_passed}/{len(models)} models passed all tests")


if __name__ == "__main__":
    main()
