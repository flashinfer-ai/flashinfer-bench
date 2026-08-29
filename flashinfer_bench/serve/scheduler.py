"""GPU worker scheduling for the benchmark server."""

import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from typing import Dict, List, Optional

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner.persistent_runner import PersistentSubprocessWorker
from flashinfer_bench.bench.runner.runner import BaselineHandle
from flashinfer_bench.data import Definition, EvaluationStatus, Solution, Trace, TraceSet, Workload
from flashinfer_bench.serve.task_store import Task, TaskStore

logger = logging.getLogger(__name__)


class SchedulerDrainingError(RuntimeError):
    """Raised when a draining scheduler receives a new task."""


class Scheduler:
    """Manages GPU workers and dispatches evaluation tasks."""

    def __init__(self, trace_set: TraceSet, config: BenchmarkConfig, devices: List[str]):
        self._trace_set = trace_set
        self._config = config
        self._task_store = TaskStore()
        self._queue: queue.Queue[str] = queue.Queue()
        self._shutdown = threading.Event()
        self._accepting = True
        self._admission_lock = threading.Lock()
        self._instance_id = uuid.uuid4().hex
        self._started_at = time.time()
        self._dataset_id = _dataset_id(trace_set)

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
    def active_tasks(self) -> int:
        return self._task_store.count_active()

    @property
    def accepting(self) -> bool:
        with self._admission_lock:
            return self._accepting

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def started_at(self) -> float:
        return self._started_at

    @property
    def workers(self) -> List["_GPUWorkerThread"]:
        return self._workers

    def submit(
        self,
        solution: Solution,
        workload_uuids: Optional[List[str]] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Submit a solution for evaluation. Returns task_id."""
        with self._admission_lock:
            if not self._accepting:
                raise SchedulerDrainingError("Server is draining and not accepting new tasks")
            existing = self._task_store.get_task(task_id) if task_id is not None else None
            task_id = self._task_store.create_task(solution, workload_uuids, task_id=task_id)
            if existing is None:
                self._queue.put(task_id)
            return task_id

    def drain(self) -> None:
        """Stop accepting new tasks while allowing admitted work to finish."""
        with self._admission_lock:
            self._accepting = False
        logger.info("Scheduler is draining")

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
        self._worker_lock = threading.RLock()
        self._healthy = False
        self._ref_cache: Dict[tuple[str, str], BaselineHandle] = {}

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_healthy(self) -> bool:
        # The worker thread maintains this cached value. HTTP health requests must not send
        # concurrent commands over the same process pipe while a benchmark is running.
        return self._healthy

    def close(self) -> None:
        with self._worker_lock:
            if self._gpu_worker:
                self._gpu_worker.close()
                self._gpu_worker = None
            self._healthy = False

    def run(self) -> None:
        retry_delay = 1.0
        while not self._shutdown.is_set():
            if self._gpu_worker is None:
                try:
                    with self._worker_lock:
                        self._gpu_worker = PersistentSubprocessWorker(self._device)
                        self._healthy = True
                    retry_delay = 1.0
                except Exception as e:
                    logger.error(
                        f"Failed to start GPU worker on {self._device}; "
                        f"retrying in {retry_delay:.0f}s: {e}"
                    )
                    self._shutdown.wait(retry_delay)
                    retry_delay = min(retry_delay * 2, 30.0)
                    continue

            try:
                task_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                self._maintain_idle_worker()
                continue

            task = self._store.get_task(task_id)
            if task is None:
                continue

            self._store.mark_running(task_id)
            try:
                with self._worker_lock:
                    traces = self._evaluate_task(task)
                self._store.complete_task(task_id, traces)
            except Exception as e:
                logger.error(f"Task {task_id} failed on {self._device}: {e}")
                self._store.fail_task(task_id, str(e))
                self._maintain_idle_worker()

    def _maintain_idle_worker(self) -> None:
        """Restart an unhealthy idle worker, falling back to fresh construction."""
        with self._worker_lock:
            if self._gpu_worker is None:
                self._healthy = False
                return
            self._healthy = self._gpu_worker.is_healthy()
            if self._healthy:
                return
            logger.warning(f"Worker on {self._device} unhealthy, restarting")
            if self._gpu_worker.restart():
                self._ref_cache.clear()
                self._healthy = True
                return
            logger.error(f"Failed to restart worker on {self._device}; retrying startup")
            self.close()
            self._ref_cache.clear()

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


def _dataset_id(trace_set: TraceSet) -> str:
    """Return a stable digest for routing-compatible definitions and workloads."""
    payload = {
        "definitions": {
            name: definition.model_dump(mode="json")
            for name, definition in sorted(trace_set.definitions.items())
        },
        "workloads": {
            name: sorted(
                (trace.workload.model_dump(mode="json") for trace in traces),
                key=lambda workload: workload["uuid"],
            )
            for name, traces in sorted(trace_set.workloads.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
