"""GPU worker scheduling for the benchmark server."""

import logging
import queue
import threading
from typing import Dict, List, Optional

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner.persistent_runner import PersistentSubprocessWorker
from flashinfer_bench.bench.runner.runner import BaselineHandle
from flashinfer_bench.data import Definition, EvaluationStatus, Solution, Trace, TraceSet, Workload
from flashinfer_bench.serve.task_store import Task, TaskStore

logger = logging.getLogger(__name__)


class Scheduler:
    """Manages GPU workers and dispatches evaluation tasks."""

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

    def submit(self, solution: Solution, workload_uuids: Optional[List[str]] = None) -> str:
        """Submit a solution for evaluation. Returns task_id."""
        task_id = self._task_store.create_task(solution, workload_uuids)
        self._queue.put(task_id)
        return task_id

    def shutdown(self) -> None:
        self._shutdown.set()
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
                traces = self._evaluate_task(task)
                self._store.complete_task(task_id, traces)
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

    def _evaluate_task(self, task: Task) -> List[Trace]:
        definition = self._trace_set.definitions.get(task.definition_name)
        if definition is None:
            raise ValueError(f"Definition not found: {task.definition_name}")

        workload_traces = self._trace_set.workloads.get(task.definition_name, [])
        if task.workload_uuids:
            uuid_set = set(task.workload_uuids)
            workload_traces = [t for t in workload_traces if t.workload.uuid in uuid_set]

        if not workload_traces:
            raise ValueError(f"No workloads found for definition: {task.definition_name}")

        traces = []
        for wl_trace in workload_traces:
            workload = wl_trace.workload
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

    def _get_or_build_ref(self, definition: Definition, workload: Workload) -> BaselineHandle:
        """Get cached reference or build a new one."""
        key = (definition.name, workload.uuid)
        if key in self._ref_cache:
            return self._ref_cache[key]

        handle = self._gpu_worker.run_ref(definition, workload, self._config, self._trace_set.root)
        self._ref_cache[key] = handle
        return handle
