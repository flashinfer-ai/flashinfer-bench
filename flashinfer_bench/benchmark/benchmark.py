import time
import logging
from typing import Literal, Optional
from flashinfer_bench.db.database import Database
from flashinfer_bench.specs.trace import Trace, Evaluation, Correctness, Performance, Environment
from flashinfer_bench.benchmark.benchmark_config import BenchmarkConfig

class Benchmark:
    def __init__(self, database: Database, config: Optional[BenchmarkConfig] = None):
        self.database = database
        self.config = config or BenchmarkConfig()
        logging.basicConfig(level=getattr(logging, self.config.log_level.upper()))
        self.logger = logging.getLogger("Benchmark")
        
    @classmethod
    def from_path(cls, path: str, config: Optional[BenchmarkConfig] = None) -> "Benchmark":
        db = Database.from_uri(path)
        return cls(db, config=config)

    def run(self):
        """run all benchmarks and append Trace to self.database.traces"""
        for solution in self.database.solutions:
            definition = self.database.get_definition(solution.definition)
            if not definition:
                self.logger.warning(f"No matching definition found for solution {solution.name}")
                continue

            self.logger.info(f"Benchmarking solution: {solution.name} for workload: {definition.name}")

            # Prepare inputs, reference outputs, etc. (placeholder)
            try:
                inputs = self._load_placeholder_inputs(definition)
                reference_outputs = self._run_reference(definition, inputs)
                self._warmup(solution, inputs)

                latencies = []
                for _ in range(self.config.iterations):
                    start = time.time()
                    outputs = self._run_solution(solution, inputs)
                    end = time.time()
                    latencies.append((end - start) * 1000)  # ms

                # Compute correctness
                max_rel, max_abs = self._compute_differences(reference_outputs, outputs)
                passed = max_abs <= self.config.max_diff_limit and max_rel <= self.config.max_diff_limit

                trace = Trace(
                    definition=definition.name,
                    solution=solution.name,
                    workload={"axes": {}, "inputs": {}},  # TODO: Add real values
                    evaluation=Evaluation(
                        status="PASSED" if passed else "INCORRECT",
                        log_file=f"logs/{definition.name}/{solution.name}.log",
                        correctness=Correctness(
                            max_relative_error=max_rel,
                            max_absolute_error=max_abs
                        ),
                        performance=Performance(
                            latency_ms=sum(latencies) / len(latencies),
                            reference_latency_ms=0.0,   # placeholder
                            speedup_factor=1.0          # placeholder
                        ),
                        environment=Environment(
                            device=self.config.device,
                            libs={"torch": "2.0.0"}     # TODO: get real versions
                        ),
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    )
                )

                self.logger.info(f"Trace generated for {solution.name}")
                self.database.traces.append(trace)

            except Exception as e:
                self.logger.exception(f"Failed to run solution {solution.name}: {e}")
                
        return self.database

    def _warmup(self, solution, inputs):
        for _ in range(self.config.warmup_runs):
            self._run_solution(solution, inputs)

    def _run_reference(self, definition, inputs):
        # TODO: run definition.reference code safely
        return {}

    def _run_solution(self, solution, inputs):
        # TODO: import and run solution.spec.entry_point
        return {}

    def _load_placeholder_inputs(self, definition):
        # TODO: replace with real input loader based on definition.inputs
        return {}

    def _compute_differences(self, ref, out):
        # TODO: compute real numeric diffs
        return 1e-6, 1e-6
