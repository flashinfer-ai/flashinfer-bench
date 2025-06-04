"""Benchmark utility classes for FlashInfer Bench."""

import os
import random
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch


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
            raise RuntimeError(
                f"Invalid device id: {self.device.index}. Only {torch.cuda.device_count()} devices available."
            )

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
            "cuda": torch.version.cuda,
            "backend": self.backend,
        }


class CorrectnessChecker:
    @staticmethod
    def check_correctness(ref_outputs, impl_outputs, rtol: float = 1e-4, atol: float = 1e-5) -> Tuple[bool, Dict[str, Any]]:
        try:
            torch.testing.assert_close(
                impl_outputs, 
                ref_outputs, 
                rtol=rtol, 
                atol=atol,
                check_device=False,
                check_dtype=True,
                equal_nan=True
            )
            return True, {}
            
        except AssertionError as e:
            error_msg = str(e)
            error_details = CorrectnessChecker._parse_error_message(error_msg)
            return False, error_details
            
        except Exception as e:
            return False, {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "max_absolute_error": 0.0,
                "max_relative_error": 0.0
            }
    
    @staticmethod
    def _parse_error_message(error_msg: str) -> Dict[str, Any]:
        details = {
            "error_message": error_msg,
            "max_absolute_error": 0.0,
            "max_relative_error": 0.0,
            "error_type": "comparison_failed"
        }
        
        abs_match = re.search(r"Greatest absolute difference: ([\d.e+-]+)", error_msg)
        if abs_match:
            details["max_absolute_error"] = float(abs_match.group(1))
        
        rel_match = re.search(r"Greatest relative difference: ([\d.e+-]+)", error_msg)
        if rel_match:
            details["max_relative_error"] = float(rel_match.group(1))
        
        if "shape" in error_msg.lower():
            details["error_type"] = "shape_mismatch"
        
        elif "dtype" in error_msg.lower():
            details["error_type"] = "dtype_mismatch"
            
        elif "device" in error_msg.lower():
            details["error_type"] = "device_mismatch"
        
        return details