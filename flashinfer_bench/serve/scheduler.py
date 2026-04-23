"""GPU worker scheduling for the benchmark server."""

import logging
import queue
import threading
from typing import Dict, List, Optional

from flashinfer_bench.agents.ncu import flashinfer_bench_run_ncu
from flashinfer_bench.agents.sanitizer import flashinfer_bench_run_sanitizer
from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner.persistent_runner import PersistentSubprocessWorker
from flashinfer_bench.bench.runner.runner import BaselineHandle
from flashinfer_bench.data import Definition, EvaluationStatus, Solution, Trace, TraceSet, Workload
from flashinfer_bench.serve.task_store import RunLog, Task, TaskKind, TaskStore
from flashinfer_bench.utils import kill_all_tracked_subprocesses

logger = logging.getLogger(__name__)


class Scheduler:
    """Manages GPU workers and dispatches evaluate/profile/sanitize tasks."""

    def __init__(self, trace_set: TraceSet, config: BenchmarkConfig, devices: List[str]):
        self._trace_set = trace_set
        self._config = config
        self._task_store = TaskStore()
        self._queue: queue.Queue[str] = queue.Queue()
        self._shutdown = threading.Event()

        self._workers: List[_GPUWorkerThread] = []
        for device in devices:
            worker = _GPUWorkerThread(
                device=device,
                task_queue=self._queue,
                task_store=self._task_store,
                trace_set=trace_set,
                config=config,
                shutdown_event=self._shutdown,
            )
            worker.start()
            self._workers.append(worker)

        logger.info(f"Scheduler started with {len(devices)} GPU workers: {devices}")

    @property
    def trace_set(self) -> TraceSet:
        return self._trace_set

    @property
    def task_store(self) -> TaskStore:
        return self._task_store

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def workers(self) -> List["_GPUWorkerThread"]:
        return self._workers

    def submit_evaluate(
        self, solution: Solution, workload_uuids: Optional[List[str]] = None
    ) -> str:
        """Submit a solution for evaluation. Returns task_id."""
        task_id = self._task_store.create_task(solution, workload_uuids, kind=TaskKind.EVALUATE)
        self._queue.put(task_id)
        return task_id

    def submit_profile(
        self,
        solution: Solution,
        workload_uuids: Optional[List[str]] = None,
        *,
        ncu_set: str = "detailed",
        ncu_sections: Optional[List[str]] = None,
        ncu_kernel_name: Optional[str] = None,
        ncu_page: str = "details",
        ncu_path: str = "ncu",
        ncu_timeout: int = 60,
        ncu_max_lines: Optional[int] = None,
    ) -> str:
        """Submit a solution for NCU profiling. Returns task_id."""
        task_id = self._task_store.create_task(
            solution,
            workload_uuids,
            kind=TaskKind.PROFILE,
            ncu_set=ncu_set,
            ncu_sections=ncu_sections,
            ncu_kernel_name=ncu_kernel_name,
            ncu_page=ncu_page,
            ncu_path=ncu_path,
            ncu_timeout=ncu_timeout,
            ncu_max_lines=ncu_max_lines,
        )
        self._queue.put(task_id)
        return task_id

    def submit_sanitize(
        self,
        solution: Solution,
        workload_uuids: Optional[List[str]] = None,
        *,
        sanitizer_types: Optional[List[str]] = None,
        sanitizer_path: str = "compute-sanitizer",
        sanitizer_timeout: int = 120,
        sanitizer_max_lines: Optional[int] = None,
        sanitizer_print_limit: Optional[int] = None,
    ) -> str:
        """Submit a solution for compute-sanitizer checks. Returns task_id."""
        task_id = self._task_store.create_task(
            solution,
            workload_uuids,
            kind=TaskKind.SANITIZE,
            sanitizer_types=sanitizer_types,
            sanitizer_path=sanitizer_path,
            sanitizer_timeout=sanitizer_timeout,
            sanitizer_max_lines=sanitizer_max_lines,
            sanitizer_print_limit=sanitizer_print_limit,
        )
        self._queue.put(task_id)
        return task_id

    def shutdown(self) -> None:
        self._shutdown.set()
        # Worker threads may be blocked in subprocess.run for compute-sanitizer
        # or ncu; killing the tracked process groups unblocks communicate() so
        # the threads can observe _shutdown and exit promptly. Without this,
        # each such thread would sit in the join timeout and the subprocess
        # (plus its _solution_runner grandchild) would keep running on the GPU
        # after the serve process exits.
        killed = kill_all_tracked_subprocesses()
        if killed:
            logger.info("Terminated %d in-flight managed subprocess group(s) on shutdown", killed)
        for worker in self._workers:
            worker.join(timeout=10)
        for worker in self._workers:
            worker.close()
        logger.info("Scheduler shut down")


class _GPUWorkerThread(threading.Thread):
    """Background thread owning a PersistentSubprocessWorker, processing tasks from the queue."""

    def __init__(
        self,
        device: str,
        task_queue: queue.Queue,
        task_store: TaskStore,
        trace_set: TraceSet,
        config: BenchmarkConfig,
        shutdown_event: threading.Event,
    ):
        super().__init__(daemon=True, name=f"gpu-worker-{device}")
        self._device = device
        self._queue = task_queue
        self._store = task_store
        self._trace_set = trace_set
        self._config = config
        self._shutdown = shutdown_event
        self._gpu_worker: Optional[PersistentSubprocessWorker] = None
        self._ref_cache: Dict[tuple[str, str], BaselineHandle] = {}

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_healthy(self) -> bool:
        return self._gpu_worker is not None and self._gpu_worker.is_healthy()

    def close(self) -> None:
        if self._gpu_worker:
            self._gpu_worker.close()
            self._gpu_worker = None

    def run(self) -> None:
        try:
            self._gpu_worker = PersistentSubprocessWorker(self._device)
        except Exception as e:
            logger.error(f"Failed to start GPU worker on {self._device}: {e}")
            return

        while not self._shutdown.is_set():
            try:
                task_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task = self._store.get_task(task_id)
            if task is None:
                continue

            self._store.mark_running(task_id)
            try:
                if task.kind == TaskKind.EVALUATE:
                    traces = self._evaluate_task(task)
                    self._store.complete_task(task_id, traces=traces)
                elif task.kind == TaskKind.PROFILE:
                    logs = self._profile_task(task)
                    self._store.complete_task(task_id, logs=logs)
                elif task.kind == TaskKind.SANITIZE:
                    logs = self._sanitize_task(task)
                    self._store.complete_task(task_id, logs=logs)
                else:
                    raise ValueError(f"Unknown task kind: {task.kind}")
            except Exception as e:
                logger.error(f"Task {task_id} failed on {self._device}: {e}")
                self._store.fail_task(task_id, str(e))
                if not self._gpu_worker.is_healthy():
                    logger.warning(f"Worker on {self._device} unhealthy, restarting")
                    if self._gpu_worker.restart():
                        self._ref_cache.clear()
                    else:
                        logger.error(f"Failed to restart worker on {self._device}, exiting")
                        return

    def _resolve_workloads(self, task: Task) -> List[Workload]:
        """Return the list of Workload objects for this task's definition + uuid filter."""
        workload_traces = self._trace_set.workloads.get(task.definition_name, [])
        if task.workload_uuids:
            uuid_set = set(task.workload_uuids)
            workload_traces = [t for t in workload_traces if t.workload.uuid in uuid_set]

        if not workload_traces:
            raise ValueError(f"No workloads found for definition: {task.definition_name}")

        return [t.workload for t in workload_traces]

    def _evaluate_task(self, task: Task) -> List[Trace]:
        definition = self._trace_set.definitions.get(task.definition_name)
        if definition is None:
            raise ValueError(f"Definition not found: {task.definition_name}")

        workloads = self._resolve_workloads(task)

        traces = []
        for workload in workloads:
            ref_handle = self._get_or_build_ref(definition, workload)
            evaluation = self._gpu_worker.run_solution(task.solution, ref_handle, self._config)
            trace = Trace(
                definition=task.definition_name,
                workload=workload,
                solution=task.solution.name,
                evaluation=evaluation,
            )
            traces.append(trace)

            # Check for CUDA context corruption after RUNTIME_ERROR
            if evaluation.status == EvaluationStatus.RUNTIME_ERROR:
                if not self._gpu_worker.is_healthy():
                    logger.warning(
                        f"Worker on {self._device} unhealthy after RUNTIME_ERROR, restarting"
                    )
                    if self._gpu_worker.restart():
                        self._ref_cache.clear()
                    else:
                        logger.error(f"Failed to restart worker on {self._device}")
                        raise RuntimeError(f"Worker on {self._device} failed to restart")

        return traces

    def _profile_task(self, task: Task) -> List[RunLog]:
        """Run NCU profiling per workload and return logs."""
        if task.definition_name not in self._trace_set.definitions:
            raise ValueError(f"Definition not found: {task.definition_name}")

        workloads = self._resolve_workloads(task)
        trace_set_path = str(self._trace_set.root) if self._trace_set.root else None

        logs: List[RunLog] = []
        for workload in workloads:
            log = flashinfer_bench_run_ncu(
                task.solution,
                workload,
                device=self._device,
                trace_set_path=trace_set_path,
                set=task.ncu_set,
                sections=task.ncu_sections,
                kernel_name=task.ncu_kernel_name,
                page=task.ncu_page,
                ncu_path=task.ncu_path,
                timeout=task.ncu_timeout,
                max_lines=task.ncu_max_lines,
            )
            logs.append(
                RunLog(
                    definition=task.definition_name,
                    workload=workload.model_dump(mode="json"),
                    solution=task.solution.name,
                    log=log,
                )
            )
        return logs

    def _sanitize_task(self, task: Task) -> List[RunLog]:
        """Run compute-sanitizer per workload and return logs."""
        if task.definition_name not in self._trace_set.definitions:
            raise ValueError(f"Definition not found: {task.definition_name}")

        workloads = self._resolve_workloads(task)
        trace_set_path = str(self._trace_set.root) if self._trace_set.root else None

        logs: List[RunLog] = []
        for workload in workloads:
            log = flashinfer_bench_run_sanitizer(
                task.solution,
                workload,
                device=self._device,
                trace_set_path=trace_set_path,
                sanitizer_types=task.sanitizer_types,  # type: ignore[arg-type]
                sanitizer_path=task.sanitizer_path,
                timeout=task.sanitizer_timeout,
                max_lines=task.sanitizer_max_lines,
                print_limit=task.sanitizer_print_limit,
            )
            logs.append(
                RunLog(
                    definition=task.definition_name,
                    workload=workload.model_dump(mode="json"),
                    solution=task.solution.name,
                    log=log,
                )
            )

            # Sanitizer runs kernels that may have crashed; check worker health.
            if not self._gpu_worker.is_healthy():
                logger.warning(f"Worker on {self._device} unhealthy after sanitize, restarting")
                if self._gpu_worker.restart():
                    self._ref_cache.clear()
                else:
                    logger.error(f"Failed to restart worker on {self._device}")
                    raise RuntimeError(f"Worker on {self._device} failed to restart")

        return logs

    def _get_or_build_ref(self, definition: Definition, workload: Workload) -> BaselineHandle:
        """Get cached reference or build a new one."""
        key = (definition.name, workload.uuid)
        if key in self._ref_cache:
            return self._ref_cache[key]

        handle = self._gpu_worker.run_ref(definition, workload, self._config, self._trace_set.root)
        self._ref_cache[key] = handle
        return handle
