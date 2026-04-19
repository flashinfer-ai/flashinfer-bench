"""Task lifecycle management for the benchmark server."""

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flashinfer_bench.data import Solution, Trace


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskKind(str, enum.Enum):
    EVALUATE = "evaluate"
    PROFILE = "profile"
    SANITIZE = "sanitize"


@dataclass
class RunLog:
    """Per-workload log output from profile/sanitize runs."""

    definition: str
    workload: Dict[str, Any]
    solution: str
    log: str


@dataclass
class Task:
    """A single task (evaluate / profile / sanitize) for one solution."""

    id: str
    solution: Solution
    definition_name: str
    workload_uuids: Optional[List[str]]
    kind: TaskKind = TaskKind.EVALUATE
    status: TaskStatus = TaskStatus.PENDING
    traces: List[Trace] = field(default_factory=list)
    logs: List[RunLog] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Profile (NCU) configuration
    ncu_set: str = "detailed"
    ncu_sections: Optional[List[str]] = None
    ncu_kernel_name: Optional[str] = None
    ncu_page: str = "details"
    ncu_path: str = "ncu"
    ncu_timeout: int = 60
    ncu_max_lines: Optional[int] = None

    # Sanitizer configuration
    sanitizer_types: Optional[List[str]] = None
    sanitizer_path: str = "compute-sanitizer"
    sanitizer_timeout: int = 300
    sanitizer_max_lines: Optional[int] = None
    sanitizer_print_limit: Optional[int] = None


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
        *,
        kind: TaskKind = TaskKind.EVALUATE,
        # Profile (NCU) configuration
        ncu_set: str = "detailed",
        ncu_sections: Optional[List[str]] = None,
        ncu_kernel_name: Optional[str] = None,
        ncu_page: str = "details",
        ncu_path: str = "ncu",
        ncu_timeout: int = 60,
        ncu_max_lines: Optional[int] = None,
        # Sanitizer configuration
        sanitizer_types: Optional[List[str]] = None,
        sanitizer_path: str = "compute-sanitizer",
        sanitizer_timeout: int = 300,
        sanitizer_max_lines: Optional[int] = None,
        sanitizer_print_limit: Optional[int] = None,
    ) -> str:
        """Create a task. Returns task_id."""
        task_id = uuid.uuid4().hex
        task = Task(
            id=task_id,
            solution=solution,
            definition_name=solution.definition,
            workload_uuids=workload_uuids,
            kind=kind,
            ncu_set=ncu_set,
            ncu_sections=ncu_sections,
            ncu_kernel_name=ncu_kernel_name,
            ncu_page=ncu_page,
            ncu_path=ncu_path,
            ncu_timeout=ncu_timeout,
            ncu_max_lines=ncu_max_lines,
            sanitizer_types=sanitizer_types,
            sanitizer_path=sanitizer_path,
            sanitizer_timeout=sanitizer_timeout,
            sanitizer_max_lines=sanitizer_max_lines,
            sanitizer_print_limit=sanitizer_print_limit,
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

    def complete_task(
        self,
        task_id: str,
        *,
        traces: Optional[List[Trace]] = None,
        logs: Optional[List[RunLog]] = None,
    ) -> None:
        """Mark a task COMPLETED and attach its result payload.

        Exactly one of `traces` (evaluate) or `logs` (profile/sanitize) should be supplied.
        """
        if (traces is None) == (logs is None):
            raise ValueError("complete_task requires exactly one of `traces` or `logs`")
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                if traces is not None:
                    task.traces = traces
                else:
                    assert logs is not None
                    task.logs = logs
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
