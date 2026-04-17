"""Task lifecycle management for the benchmark server."""

import asyncio
import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.data import Solution, Trace


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """A single evaluation task for one solution."""

    id: str
    solution: Solution
    definition_name: str
    workload_uuids: Optional[List[str]]
    status: TaskStatus = TaskStatus.PENDING
    traces: List[Trace] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    config_override: Optional[BenchmarkConfig] = None
    sanitize: bool = False
    sanitizer_types: Optional[List[str]] = None
    sanitizer_print_limit: Optional[int] = None
    sanitizer_max_lines: Optional[int] = None


class TaskStore:
    """Thread-safe task storage with TTL-based cleanup."""

    def __init__(self, ttl_seconds: int = 3600):
        self._tasks: Dict[str, Task] = {}
        self._events: Dict[str, threading.Event] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def create_task(
        self,
        solution: Solution,
        workload_uuids: Optional[List[str]] = None,
        config_override: Optional[BenchmarkConfig] = None,
        sanitize: bool = False,
        sanitizer_types: Optional[List[str]] = None,
        sanitizer_print_limit: Optional[int] = None,
        sanitizer_max_lines: Optional[int] = None,
    ) -> str:
        """Create a single evaluation task. Returns task_id."""
        task_id = uuid.uuid4().hex
        task = Task(
            id=task_id,
            solution=solution,
            definition_name=solution.definition,
            workload_uuids=workload_uuids,
            config_override=config_override,
            sanitize=sanitize,
            sanitizer_types=sanitizer_types,
            sanitizer_print_limit=sanitizer_print_limit,
            sanitizer_max_lines=sanitizer_max_lines,
        )
        with self._lock:
            self._tasks[task_id] = task
            self._events[task_id] = threading.Event()
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def mark_running(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.RUNNING

    def complete_task(self, task_id: str, traces: List[Trace]) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.traces = traces
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                assert task_id in self._events, f"Event missing for task {task_id}"
                self._events[task_id].set()

    def fail_task(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.error = error
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                assert task_id in self._events, f"Event missing for task {task_id}"
                self._events[task_id].set()

    def wait_for_all(self, task_ids: List[str], timeout: float) -> bool:
        """Block until all tasks complete or timeout. Returns True if all done."""
        deadline = time.time() + timeout
        for task_id in task_ids:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            event = self._events.get(task_id)
            if event and not event.is_set():
                event.wait(timeout=remaining)
        return all(
            self._tasks[tid].status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for tid in task_ids
            if tid in self._tasks
        )

    async def async_wait_for_all(self, task_ids: List[str], timeout: float) -> bool:
        """Async wait that cooperates with cancellation (client disconnect).

        Polls threading.Events in short intervals so that asyncio cancellation
        (triggered when a client closes the connection) is honoured promptly.
        """
        deadline = time.monotonic() + timeout
        poll_interval = 0.5
        for task_id in task_ids:
            event = self._events.get(task_id)
            if event is None or event.is_set():
                continue
            while not event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                await asyncio.sleep(min(poll_interval, remaining))
        return all(
            self._tasks[tid].status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for tid in task_ids
            if tid in self._tasks
        )

    def cleanup(self) -> int:
        """Remove completed tasks older than TTL. Returns count removed."""
        now = time.time()
        to_remove = []
        with self._lock:
            for task_id, task in self._tasks.items():
                if task.completed_at and (now - task.completed_at) > self._ttl:
                    to_remove.append(task_id)
            for task_id in to_remove:
                del self._tasks[task_id]
                self._events.pop(task_id, None)
        return len(to_remove)
