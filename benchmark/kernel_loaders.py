import os
import importlib.util
import types
import tempfile
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
import torch
import torch.nn as nn

"""The KernelLoader class abstracts dataset loading, allowing flashinfer-bench to support a variety of datasets."""

def strip_code_blocks(source: str) -> str:
    if not source:
        return source
    
    source = source.strip()
    
    if '```' not in source:
        return source
    
    lines = source.split('\n')
    cleaned_lines = []
    in_code_block = False
    found_any_code_block = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if stripped.startswith('```'):
            found_any_code_block = True
            in_code_block = not in_code_block
            continue
        
        if stripped == '```' or stripped.endswith('```'):
            if not found_any_code_block:
                break
            else:
                in_code_block = not in_code_block
                continue
        
        if in_code_block or not found_any_code_block:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    
    if not result.strip():
        return source
        
    return result

class KernelLoader(ABC):    
    @abstractmethod
    def load_original_model_and_inputs(self, source: str) -> Tuple[str, Any, Any, Any]:
        """
        Load original model and input generators from source.
        Returns (code_string, Model_class, get_init_inputs_func, get_inputs_func)
        """
        pass
    
    @abstractmethod
    def load_generated_model(self, source: str, entry_point: str = "ModelNew") -> Any:
        """
        Load generated model class from source.
        Returns model class.
        """
        pass


class KernelBenchLoader(KernelLoader):
    """Adapted from KernelBench https://github.com/ScalingIntelligence/KernelBench/blob/main/src/eval.py"""
    def load_original_model_and_inputs(self, source: str) -> Tuple[str, Any, Any, Any]:
        if os.path.exists(source):
            spec = importlib.util.spec_from_file_location("kernel_module", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            with open(source, 'r') as f:
                code_string = f.read()
        else:
            module = types.ModuleType("kernel_module")
            exec(source, module.__dict__)
            code_string = source
        
        context = module.__dict__
        return self._load_from_context(code_string, context)
    
    def _load_from_context(self, code_string: str, context: dict) -> Tuple[str, Any, Any, Any]:
        try:
            compile(code_string, "<string>", "exec")
        except SyntaxError as e:
            raise ValueError(f"Syntax Error in original code: {e}")
        
        get_init_inputs_fn = context.get("get_init_inputs")
        get_inputs_fn = context.get("get_inputs") 
        Model = context.get("Model")
        
        if Model is None:
            raise ValueError("No 'Model' class found in kernel description")
        if get_inputs_fn is None:
            raise ValueError("No 'get_inputs' function found in kernel description")
        
        enhanced_code = code_string + "\n\n# Auto-generated compatibility wrapper\nModelNew = Model\n"
        
        return enhanced_code, Model, get_init_inputs_fn, get_inputs_fn
    
    def load_generated_model(self, source: str, entry_point: str = "ModelNew") -> Any:
        source = strip_code_blocks(source)
        
        context = {}
        try:
            compile(source, "<string>", "exec")
            exec(source, context)
        except SyntaxError as e:
            raise ValueError(f"Syntax Error in generated code: {e}")
        except Exception as e:
            raise ValueError(f"Error executing generated code: {e}")
        
        ModelNew = context.get(entry_point)
        if ModelNew is None:
            raise ValueError(f"No '{entry_point}' class found in generated code")
        
        return ModelNew


class TritonKernelLoader(KernelLoader):
    """Loader for Triton kernels with temp file loading hack to resolve compile/exec issues with @triton.jit 
    Adapted from: https://github.com/ScalingIntelligence/KernelBench/pull/35"""
    def __init__(self):
        self.temp_files = []
    
    def load_original_model_and_inputs(self, source: str) -> Tuple[str, Any, Any, Any]:
        return KernelBenchLoader().load_original_model_and_inputs(source)
    
    def load_generated_model(self, source: str, entry_point: str = "ModelNew") -> Any:
        source = strip_code_blocks(source)
        
        model_class, temp_file = self._load_custom_model_with_tempfile(source, entry_point)
        self.temp_files.append(temp_file)
        return model_class
    
    def _load_custom_model_with_tempfile(self, model_custom_src: str, entry_point: str = "ModelNew"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
            tmp_file.write(model_custom_src)
            tempfile_path = tmp_file.name
            temp_file = tmp_file

        spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)

        ModelNew = getattr(temp_module, entry_point)
        return ModelNew, temp_file
    
    def cleanup(self):
        for temp_file in self.temp_files:
            try:
                if hasattr(temp_file, 'name') and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp file: {e}")
        self.temp_files.clear()


class FlashinferKernelLoader(KernelLoader):    
    def load_original_model_and_inputs(self, source: str) -> Tuple[str, Any, Any, Any]:
        """Format TBD, TODO: decide on dataset format"""
        if os.path.exists(source):
            spec = importlib.util.spec_from_file_location("kernel_module", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            with open(source, 'r') as f:
                code_string = f.read()
        else:
            module = types.ModuleType("kernel_module")
            exec(source, module.__dict__)
            code_string = source
        
        run_pytorch = getattr(module, 'run_pytorch', None)
        gen_data = getattr(module, 'gen_data', None)
        
        if run_pytorch is None:
            raise ValueError("No 'run_pytorch' function found")
        if gen_data is None:
            raise ValueError("No 'gen_data' function found")
        
        # For flashinfer format, we need to extract the Model class from the code
        # and create a dummy get_init_inputs that returns empty list
        try:
            exec(code_string, module.__dict__)
            Model = getattr(module, 'ModelNew', None)
            if Model is None:
                raise ValueError("No 'ModelNew' class found in flashinfer format")
            
            def dummy_get_init_inputs():
                return []
                
            return code_string, Model, dummy_get_init_inputs, gen_data
            
        except Exception as e:
            raise ValueError(f"Failed to extract Model from flashinfer code: {e}")
    
    def load_generated_model(self, source: str, entry_point: str = "ModelNew") -> Any:
        return KernelBenchLoader().load_generated_model(source, entry_point)


def create_kernel_loader(source: str, loader_type: str = "auto") -> KernelLoader:
    if loader_type == "kernelbench":
        return KernelBenchLoader()
    elif loader_type == "triton":
        return TritonKernelLoader()
    elif loader_type == "flashinfer":
        return FlashinferKernelLoader()
    elif loader_type == "auto":
        try:
            if os.path.exists(source):
                with open(source, 'r') as f:
                    content = f.read()
            else:
                content = source
            
            if "run_pytorch" in content and "gen_data" in content:
                return FlashinferKernelLoader()
=            elif "import triton" in content or "@triton.jit" in content:
                return TritonKernelLoader()
            else:
                return KernelBenchLoader()
        except:
            return KernelBenchLoader()
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")