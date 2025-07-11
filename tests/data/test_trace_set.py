#!/usr/bin/env python3
"""Test script for the TraceSet class."""

import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from flashinfer_bench.trace_set import TraceSet, build_index, filter_passed_traces, filter_by_error, get_best_trace


class TestTraceSet(unittest.TestCase):
    """Test suite for TraceSet class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_definition = Mock()
        self.mock_definition.name = "test_gemm"
        self.mock_definition.type = "gemm"
        
        self.mock_solution = Mock()
        self.mock_solution.name = "test_gemm_triton" 
        self.mock_solution.definition = "test_gemm"
        
        self.mock_trace_passed = Mock()
        self.mock_trace_passed.definition = "test_gemm"
        self.mock_trace_passed.solution = "test_gemm_triton"
        self.mock_trace_passed.evaluation = {
            "status": "PASSED",
            "correctness": {
                "max_absolute_error": 1e-6,
                "max_relative_error": 1e-5
            },
            "performance": {
                "speedup_factor": 2.0
            }
        }
        
        self.mock_trace_failed = Mock()
        self.mock_trace_failed.definition = "test_gemm"
        self.mock_trace_failed.solution = "test_gemm_triton"
        self.mock_trace_failed.evaluation = {
            "status": "RUNTIME_ERROR",
            "correctness": {
                "max_absolute_error": 0,
                "max_relative_error": 0
            },
            "performance": {
                "speedup_factor": 0
            }
        }
    
    @patch('flashinfer_bench.trace_set.Definition')
    @patch('flashinfer_bench.trace_set.Solution')
    @patch('flashinfer_bench.trace_set.Trace')
    def test_from_path(self, MockTrace, MockSolution, MockDefinition):
        """Test loading TraceSet from a directory path."""
        # Set up mocks
        MockDefinition.side_effect = lambda **kwargs: Mock(name=kwargs.get('name', 'def'))
        MockSolution.side_effect = lambda **kwargs: Mock(name=kwargs.get('name', 'sol'))
        MockTrace.from_dict = lambda data: Mock(
            definition=data.get('definition'),
            solution=data.get('solution'),
            evaluation=data.get('evaluation', {})
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create test files
            with open(tmpdir_path / "definitions.json", "w") as f:
                json.dump([{"name": "test_gemm"}, {"name": "test_conv"}], f)
            
            with open(tmpdir_path / "solutions.json", "w") as f:
                json.dump([{"name": "test_sol1"}, {"name": "test_sol2"}], f)
            
            with open(tmpdir_path / "traces.jsonl", "w") as f:
                f.write(json.dumps({"definition": "test_gemm", "solution": "test_sol1"}) + "\n")
                f.write(json.dumps({"definition": "test_conv", "solution": "test_sol2"}) + "\n")
            
            # Load TraceSet
            trace_set = TraceSet.from_path(tmpdir_path)
            
            # Verify
            self.assertEqual(len(trace_set.definitions), 2)
            self.assertEqual(len(trace_set.solutions), 2)
            self.assertEqual(len(trace_set.traces), 2)
    
    def test_get_definition(self):
        """Test getting a definition by name."""
        trace_set = TraceSet(
            definitions={"test_gemm": self.mock_definition},
            solutions={},
            traces=[]
        )
        
        result = trace_set.get_definition("test_gemm")
        self.assertEqual(result, self.mock_definition)
        
        result_none = trace_set.get_definition("non_existent")
        self.assertIsNone(result_none)
    
    def test_get_solution(self):
        """Test getting a solution by name."""
        trace_set = TraceSet(
            definitions={},
            solutions={"test_gemm_triton": self.mock_solution},
            traces=[]
        )
        
        result = trace_set.get_solution("test_gemm_triton")
        self.assertEqual(result, self.mock_solution)
        
        result_none = trace_set.get_solution("non_existent")
        self.assertIsNone(result_none)
    
    def test_get_traces_for_definition(self):
        """Test getting traces for a specific definition."""
        other_trace = Mock()
        other_trace.definition = "other_def"
        
        trace_set = TraceSet(
            definitions={},
            solutions={},
            traces=[self.mock_trace_passed, self.mock_trace_failed, other_trace]
        )
        
        traces = trace_set.get_traces_for_definition("test_gemm")
        self.assertEqual(len(traces), 2)
        self.assertTrue(all(t.definition == "test_gemm" for t in traces))
        
        traces_empty = trace_set.get_traces_for_definition("non_existent")
        self.assertEqual(len(traces_empty), 0)
    
    def test_get_best_op(self):
        """Test getting the best operation for a definition."""
        # Create traces with different speedup factors
        trace1 = Mock()
        trace1.definition = "test_gemm"
        trace1.evaluation = {
            "status": "PASSED",
            "correctness": {"max_absolute_error": 1e-6, "max_relative_error": 1e-5},
            "performance": {"speedup_factor": 1.5}
        }
        
        trace2 = Mock()
        trace2.definition = "test_gemm"
        trace2.evaluation = {
            "status": "PASSED",
            "correctness": {"max_absolute_error": 1e-6, "max_relative_error": 1e-5},
            "performance": {"speedup_factor": 2.5}
        }
        
        trace3 = Mock()
        trace3.definition = "test_gemm"
        trace3.evaluation = {
            "status": "PASSED",
            "correctness": {"max_absolute_error": 1e-3, "max_relative_error": 1e-2},
            "performance": {"speedup_factor": 3.0}
        }
        
        trace_set = TraceSet(
            definitions={},
            solutions={},
            traces=[trace1, trace2, trace3, self.mock_trace_failed]
        )
        
        # Best with default thresholds - should include high error trace
        best = trace_set.get_best_op("test_gemm")
        self.assertIsNotNone(best)
        self.assertEqual(best.evaluation["performance"]["speedup_factor"], 3.0)
        
        # Best with strict thresholds - should exclude high error trace
        best_strict = trace_set.get_best_op("test_gemm", max_abs_diff=1e-4, max_relative_diff=1e-4)
        self.assertIsNotNone(best_strict)
        self.assertEqual(best_strict.evaluation["performance"]["speedup_factor"], 2.5)
        
        # No valid traces
        best_none = trace_set.get_best_op("test_gemm", max_abs_diff=1e-10, max_relative_diff=1e-10)
        self.assertIsNone(best_none)


class TestHelperFunctions(unittest.TestCase):
    """Test suite for helper functions."""
    
    def test_build_index(self):
        """Test build_index function."""
        items = [
            Mock(name="item1", value=10),
            Mock(name="item2", value=20),
            Mock(name="item3", value=30)
        ]
        
        index = build_index(items, lambda x: x.name)
        
        self.assertEqual(len(index), 3)
        self.assertEqual(index["item1"].value, 10)
        self.assertEqual(index["item2"].value, 20)
        self.assertEqual(index["item3"].value, 30)
    
    def test_filter_passed_traces(self):
        """Test filter_passed_traces function."""
        trace_passed = Mock()
        trace_passed.evaluation = {"status": "PASSED"}
        
        trace_failed = Mock()
        trace_failed.evaluation = {"status": "RUNTIME_ERROR"}
        
        trace_incorrect = Mock()
        trace_incorrect.evaluation = {"status": "INCORRECT"}
        
        traces = [trace_passed, trace_failed, trace_incorrect]
        passed = filter_passed_traces(traces)
        
        self.assertEqual(len(passed), 1)
        self.assertEqual(passed[0], trace_passed)
    
    def test_filter_by_error(self):
        """Test filter_by_error function."""
        trace_low_error = Mock()
        trace_low_error.evaluation = {
            "correctness": {
                "max_absolute_error": 1e-6,
                "max_relative_error": 1e-5
            }
        }
        
        trace_high_error = Mock()
        trace_high_error.evaluation = {
            "correctness": {
                "max_absolute_error": 1e-3,
                "max_relative_error": 1e-2
            }
        }
        
        traces = [trace_low_error, trace_high_error]
        
        # Strict filter
        filtered_strict = filter_by_error(traces, max_abs=1e-5, max_rel=1e-4)
        self.assertEqual(len(filtered_strict), 1)
        self.assertEqual(filtered_strict[0], trace_low_error)
        
        # Loose filter
        filtered_loose = filter_by_error(traces, max_abs=1e-2, max_rel=1e-1)
        self.assertEqual(len(filtered_loose), 2)
    
    def test_get_best_trace(self):
        """Test get_best_trace function."""
        trace1 = Mock()
        trace1.evaluation = {"performance": {"speedup_factor": 1.5}}
        
        trace2 = Mock()
        trace2.evaluation = {"performance": {"speedup_factor": 2.5}}
        
        trace3 = Mock()
        trace3.evaluation = {"performance": {"speedup_factor": 2.0}}
        
        traces = [trace1, trace2, trace3]
        best = get_best_trace(traces)
        
        self.assertIsNotNone(best)
        self.assertEqual(best, trace2)
        
        # Test with empty list
        self.assertIsNone(get_best_trace([]))


if __name__ == "__main__":
    unittest.main() 