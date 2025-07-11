"""Benchmark utility classes for FlashInfer Bench."""

import os
import random
from typing import Dict, List, Union, Any, Tuple
import torch
import numpy as np


class SeedManager:
    """For reproducibility"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset_seed()
    
    def reset_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)


class DeviceManager:
    def __init__(self, device: Union[int, torch.device, str] = None, backend: str = "cuda"):
        self.backend = backend
        self.is_triton = backend == "triton"
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        if device is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        if self.device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Invalid device id: {self.device.index}. Only {torch.cuda.device_count()} devices available.")
        
        torch.cuda.set_device(self.device)
        
        if self.is_triton:
            if device is None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device.index)
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    def get_device_info(self) -> Dict:
        device_id = self.device.index
        return {
            "device_name": torch.cuda.get_device_name(device_id),
            "device_id": device_id,
            "device_str": str(self.device),
            "compute_capability": torch.cuda.get_device_capability(device_id),
            "total_memory": torch.cuda.get_device_properties(device_id).total_memory,
            "driver_version": torch.version.cuda,
            "backend": self.backend,
        }


class CorrectnessChecker:    
    @staticmethod
    def max_absolute_diff(output1: torch.Tensor, output2: torch.Tensor) -> float:
        if isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
            max_diffs = []
            for o1, o2 in zip(output1, output2):
                if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    max_diffs.append(torch.max(torch.abs(o1 - o2)).item())
            return max(max_diffs) if max_diffs else float('inf')
        elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            return torch.max(torch.abs(output1 - output2)).item()
        else:
            return float('inf')
    
    @staticmethod
    def max_relative_diff(output1: torch.Tensor, output2: torch.Tensor) -> float:
        if isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
            max_diffs = []
            for o1, o2 in zip(output1, output2):
                if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    abs_diff = torch.abs(o1 - o2)
                    rel_diff = abs_diff / (torch.abs(o1) + 1e-8)
                    max_diffs.append(torch.max(rel_diff).item())
            return max(max_diffs) if max_diffs else float('inf')
        elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            abs_diff = torch.abs(output1 - output2)
            rel_diff = abs_diff / (torch.abs(output1) + 1e-8)
            return torch.max(rel_diff).item()
        else:
            return float('inf')
    
    @staticmethod
    def max_diff(output1: torch.Tensor, output2: torch.Tensor) -> float:
        # For backward compatibility - returns absolute diff
        return CorrectnessChecker.max_absolute_diff(output1, output2)
    
    @staticmethod
    def avg_diff(output1: torch.Tensor, output2: torch.Tensor) -> float:
        if isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
            avg_diffs = []
            for o1, o2 in zip(output1, output2):
                if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    avg_diffs.append(torch.mean(torch.abs(o1 - o2)).item())
            return np.mean(avg_diffs) if avg_diffs else float('inf')
        elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            return torch.mean(torch.abs(output1 - output2)).item()
        else:
            return float('inf')
    
    @staticmethod
    def validate_shapes(output1: torch.Tensor, output2: torch.Tensor) -> Tuple[bool, str]:
        if isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
            if len(output1) != len(output2):
                return False, f"Output length mismatch: Expected {len(output1)}, got {len(output2)}"
            
            for i, (o1, o2) in enumerate(zip(output1, output2)):
                if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    if o1.shape != o2.shape:
                        return False, f"Output[{i}] shape mismatch: Expected {o1.shape}, got {o2.shape}"
                elif type(o1) != type(o2):
                    return False, f"Output[{i}] type mismatch: Expected {type(o1)}, got {type(o2)}"
            return True, ""
            
        elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            if output1.shape != output2.shape:
                return False, f"Output shape mismatch: Expected {output1.shape}, got {output2.shape}"
            return True, ""
        elif type(output1) != type(output2):
            return False, f"Output type mismatch: Expected {type(output1)}, got {type(output2)}"
        else:
            return True, ""
    
    @staticmethod
    def check_correctness(output1: torch.Tensor, output2: torch.Tensor, max_diff_limit: float, 
                         validate_shapes: bool = True) -> Dict[str, Any]:
        result = {
            "correct": False,
            "max_diff": float('inf'),
            "avg_diff": float('inf'),
            "shape_valid": True,
            "shape_error": "",
            "error": None
        }
        
        try:
            # shape validation
            if validate_shapes:
                shape_valid, shape_error = CorrectnessChecker.validate_shapes(output1, output2)
                result["shape_valid"] = shape_valid
                result["shape_error"] = shape_error
                
                if not shape_valid:
                    result["error"] = shape_error
                    return result
            
            # differences
            max_diff = CorrectnessChecker.max_diff(output1, output2)
            avg_diff = CorrectnessChecker.avg_diff(output1, output2)
            
            result["max_diff"] = max_diff
            result["avg_diff"] = avg_diff
            
            result["correct"] = max_diff <= max_diff_limit
            
        except Exception as e:
            result["error"] = str(e)
            result["correct"] = False
        
        return result
    
    @staticmethod
    def is_correct(output1: torch.Tensor, output2: torch.Tensor, max_diff_limit: float) -> bool:
        """Simple correctness check - returns only boolean result"""
        result = CorrectnessChecker.check_correctness(output1, output2, max_diff_limit)
        return result["correct"]
