"""Task lifecycle management for the benchmark server."""

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from flashinfer_bench.data import Solution, Trace


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskConflictError(ValueError):
    """Raised when a task identifier is reused for a different request."""


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
        task_id: Optional[str] = None,
    ) -> str:
        """Create an evaluation task, or return an identical task created earlier.

        Supplying ``task_id`` makes submission idempotent. This lets a router safely reconcile a
        request after losing the server response. Reusing an identifier with a different payload
        raises :class:`TaskConflictError`.
        """
        task_id = task_id or uuid.uuid4().hex
        task = Task(
            id=task_id,
            solution=solution,
            definition_name=solution.definition,
            workload_uuids=workload_uuids,
        )
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing is not None:
                if (
                    existing.solution.model_dump(mode="json") == solution.model_dump(mode="json")
                    and existing.workload_uuids == workload_uuids
                ):
                    return task_id
                raise TaskConflictError(
                    f"Task ID already exists with a different request: {task_id}"
                )
            self._tasks[task_id] = task
            self._events[task_id] = threading.Event()
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def mark_running(self, task_id: str) -> None:
        with self._lock:
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
        with self._lock:
            return all(
                self._tasks[tid].status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                for tid in task_ids
                if tid in self._tasks
            )

    def count_active(self) -> int:
        """Return the number of pending or running tasks."""
        with self._lock:
            return sum(
                task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                for task in self._tasks.values()
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
