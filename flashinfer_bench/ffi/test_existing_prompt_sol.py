"""
Test script to re-test previously generated CUDA kernels from test_results directory.
Reads .txt files containing generated code and runs the same test cases.
"""

import torch
from tvm_ffi import Module
import tvm_ffi.cpp
from pathlib import Path
import re


def extract_code_from_file(file_path: Path) -> tuple[str, str]:
    """Extract model name and generated code from a test result file.
    
    Returns:
        tuple: (model_name, code)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract model name
    model_match = re.search(r'Model: (.+?)\n', content)
    model_name = model_match.group(1) if model_match else file_path.stem
    
    # Extract code between "Generated Code:" and "Test Results:"
    code_pattern = r'Generated Code:\n={80}\n(.*?)\n(?:Test Results:|$)'
    code_match = re.search(code_pattern, content, re.DOTALL)
    
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Try alternative pattern - just get everything after "Generated Code:"
        alt_pattern = r'Generated Code:\n={80}\n(.*)'
        alt_match = re.search(alt_pattern, content, re.DOTALL)
        if alt_match:
            # Get everything until we see "Test Results:" or end of file
            full_content = alt_match.group(1)
            if 'Test Results:' in full_content:
                code = full_content.split('Test Results:')[0].strip()
            else:
                code = full_content.strip()
        else:
            raise ValueError(f"Could not extract code from {file_path}")
    
    return model_name, code


def test_kernel(mod: Module, test_name: str) -> bool:
    """Test the generated kernel with simple test cases."""
    try:
        print(f"\n  Running {test_name}...")
        
        if test_name == "test_small":
            # Test 1: Small tensor
            a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device='cuda')
            b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float32, device='cuda')
            c = torch.empty_like(a)
            
            mod.elementwise_add(a, b, c)
            expected = a + b
            
            torch.testing.assert_close(c, expected)
            print(f"  ✓ {test_name} passed")
            return True
            
        elif test_name == "test_large":
            # Test 2: Larger tensor
            n = 10000
            a = torch.randn(n, dtype=torch.float32, device='cuda')
            b = torch.randn(n, dtype=torch.float32, device='cuda')
            c = torch.empty_like(a)
            
            mod.elementwise_add(a, b, c)
            expected = a + b
            
            torch.testing.assert_close(c, expected)
            print(f"  ✓ {test_name} passed")
            return True
            
    except Exception as e:
        print(f"  ✗ {test_name} failed: {e}")
        return False


def update_test_results_in_file(file_path: Path, model_name: str, code: str, results: dict):
    """Update the test results section in a result file, overwriting old results."""
    # Build the new content
    content = f"Model: {model_name}\n"
    content += "=" * 80 + "\n\n"
    content += "Generated Code:\n"
    content += "=" * 80 + "\n"
    content += code + "\n\n"
    content += "Test Results:\n"
    content += "=" * 80 + "\n"
    content += f"Compilation: {results['compilation']}\n"
    
    if "error" in results:
        content += f"Error: {results['error']}\n"
    else:
        content += f"Test 1 (small tensor): {results['test_small']}\n"
        content += f"Test 2 (large tensor): {results['test_large']}\n"
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(content)


def test_saved_code(file_path: Path):
    """Test code from a saved result file."""
    print(f"\n{'='*80}")
    print(f"Testing: {file_path.name}")
    print(f"{'='*80}")
    
    try:
        # Extract code
        model_name, cuda_code = extract_code_from_file(file_path)
        print(f"Model: {model_name}")
        print(f"Extracted {len(cuda_code)} characters of code")
        
        # Try to compile
        print("Attempting to compile with tvm_ffi...")
        try:
            mod: Module = tvm_ffi.cpp.load_inline(
                name=f'elementwise_add_{model_name.replace("-", "_")}',
                cuda_sources=cuda_code
            )
            print("✓ Compilation successful!")
            
            # Run tests
            print("\nRunning tests...")
            test1_passed = test_kernel(mod, "test_small")
            test2_passed = test_kernel(mod, "test_large")
            
            all_passed = test1_passed and test2_passed
            status = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
            print(f"\n{status}")
            
            # Update the file with new test results
            update_test_results_in_file(file_path, model_name, cuda_code, {
                "compilation": "SUCCESS",
                "test_small": "PASS" if test1_passed else "FAIL",
                "test_large": "PASS" if test2_passed else "FAIL"
            })
            
            return {
                "file": file_path.name,
                "model": model_name,
                "compilation": "success",
                "test_small": test1_passed,
                "test_large": test2_passed,
                "all_passed": all_passed
            }
            
        except Exception as e:
            print(f"✗ Compilation failed: {e}")
            
            # Update the file with compilation failure
            update_test_results_in_file(file_path, model_name, cuda_code, {
                "compilation": "FAILED",
                "error": str(e)
            })
            
            return {
                "file": file_path.name,
                "model": model_name,
                "compilation": "failed",
                "error": str(e),
                "all_passed": False
            }
            
    except Exception as e:
        print(f"✗ Error processing file: {e}")
        return {
            "file": file_path.name,
            "compilation": "error",
            "error": str(e),
            "all_passed": False
        }


def main():
    """Test all saved code from test_results directory."""
    
    # Get test_results directory
    test_results_dir = Path(__file__).parent / "test_results"
    
    if not test_results_dir.exists():
        print(f"Error: {test_results_dir} does not exist")
        return
    
    # Find all .txt files
    txt_files = sorted(test_results_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {test_results_dir}")
        return
    
    print("="*80)
    print("Re-testing Saved LLM Generated Code")
    print("="*80)
    print(f"Files to test: {len(txt_files)}")
    print(f"Directory: {test_results_dir}")
    
    results = []
    
    for idx, file_path in enumerate(txt_files, 1):
        print(f"\n[{idx}/{len(txt_files)}] Testing {file_path.name}...")
        result = test_saved_code(file_path)
        results.append(result)
    
    # Print summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        file_name = result['file']
        status = "✓ PASS" if result.get('all_passed', False) else "✗ FAIL"
        print(f"{file_name:40} {status}")
        if 'error' in result:
            error_msg = result['error']
            # Truncate long errors
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            print(f"  Error: {error_msg}")
    
    total_passed = sum(1 for r in results if r.get('all_passed', False))
    print(f"\nTotal: {total_passed}/{len(results)} files passed all tests")


if __name__ == "__main__":
    main()

