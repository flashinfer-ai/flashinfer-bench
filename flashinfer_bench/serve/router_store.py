"""Durable task state for the benchmark server router."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ACTIVE_STATUSES = ("pending", "assigned", "running")
ASSIGNED_STATUSES = ("assigned", "running")
TERMINAL_STATUSES = ("completed", "failed")


class RouterQueueFull(RuntimeError):
    """Raised when router admission would exceed the configured active-task limit."""


class RouterTaskConflict(ValueError):
    """Raised when a client task identifier is reused for a different request."""


@dataclass(frozen=True)
class RouterTask:
    """One logical router task loaded from persistent storage."""

    id: str
    request: Dict[str, Any]
    definition: str
    solution_name: str
    status: str
    backend_id: Optional[str]
    backend_instance_id: Optional[str]
    attempts: int
    response: Optional[Dict[str, Any]]
    last_error: Optional[str]
    created_at: float
    updated_at: float


class RouterTaskStore:
    """SQLite-backed task registry for a single router process."""

    def __init__(self, path: Path | str, max_active_tasks: int = 10_000):
        self._path = str(path)
        self._max_active_tasks = max_active_tasks
        if max_active_tasks < 1:
            raise ValueError("max_active_tasks must be at least one")
        self._lock = threading.RLock()
        database_existed = self._path == ":memory:" or Path(self._path).exists()
        if self._path != ":memory:":
            db_path = Path(self._path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        if not database_existed:
            os.chmod(self._path, 0o600)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS router_tasks (
                    id TEXT PRIMARY KEY,
                    request_json TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    solution_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    backend_id TEXT,
                    backend_instance_id TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    response_json TEXT,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS router_tasks_status_idx "
                "ON router_tasks(status, created_at)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS router_tasks_backend_idx "
                "ON router_tasks(backend_id, status)"
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_task(
        self,
        request: Dict[str, Any],
        definition: str,
        solution_name: str,
        task_id: Optional[str] = None,
    ) -> RouterTask:
        """Persist a request, or return an identical client-identified request."""
        now = time.time()
        task_id = task_id or uuid.uuid4().hex
        request_json = json.dumps(request, sort_keys=True, separators=(",", ":"))
        with self._transaction():
            existing = self._connection.execute(
                "SELECT * FROM router_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if existing is not None:
                existing_task = self._from_row(existing)
                if (
                    existing["request_json"] == request_json
                    and existing_task.definition == definition
                    and existing_task.solution_name == solution_name
                ):
                    return existing_task
                raise RouterTaskConflict(
                    f"Task ID already exists with a different request: {task_id}"
                )
            active = self._connection.execute(
                "SELECT COUNT(*) FROM router_tasks WHERE status IN (?, ?, ?)", ACTIVE_STATUSES
            ).fetchone()[0]
            if active >= self._max_active_tasks:
                raise RouterQueueFull(
                    f"Router has reached its active task limit ({self._max_active_tasks})"
                )
            self._connection.execute(
                """
                INSERT INTO router_tasks (
                    id, request_json, definition, solution_name, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                (task_id, request_json, definition, solution_name, now, now),
            )
        task = self.get_task(task_id)
        assert task is not None
        return task

    def get_task(self, task_id: str) -> Optional[RouterTask]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM router_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list_pending(self, limit: int) -> List[RouterTask]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM router_tasks WHERE status = 'pending' "
                "ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def list_assigned(self, backend_id: Optional[str] = None) -> List[RouterTask]:
        with self._lock:
            if backend_id is None:
                rows = self._connection.execute(
                    "SELECT * FROM router_tasks WHERE status IN (?, ?) ORDER BY created_at",
                    ASSIGNED_STATUSES,
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT * FROM router_tasks WHERE backend_id = ? "
                    "AND status IN (?, ?) ORDER BY created_at",
                    (backend_id, *ASSIGNED_STATUSES),
                ).fetchall()
        return [self._from_row(row) for row in rows]

    def assignment_counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT backend_id, COUNT(*) AS count FROM router_tasks "
                "WHERE status IN (?, ?) GROUP BY backend_id",
                ASSIGNED_STATUSES,
            ).fetchall()
        return {row["backend_id"]: row["count"] for row in rows if row["backend_id"]}

    def assign(self, task_id: str, backend_id: str, backend_instance_id: str) -> bool:
        """Atomically move a pending task to a specific server generation."""
        now = time.time()
        with self._transaction():
            cursor = self._connection.execute(
                """
                UPDATE router_tasks
                SET status = 'assigned', backend_id = ?, backend_instance_id = ?,
                    attempts = attempts + 1, last_error = NULL, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (backend_id, backend_instance_id, now, task_id),
            )
        return cursor.rowcount == 1

    def mark_remote_status(self, task: RouterTask, status: str) -> bool:
        """Record an observed non-terminal server state if the assignment is still current."""
        if status not in ("pending", "running"):
            raise ValueError(f"Not a non-terminal task status: {status}")
        stored_status = "running" if status == "running" else "assigned"
        with self._transaction():
            cursor = self._connection.execute(
                """
                UPDATE router_tasks SET status = ?, updated_at = ?
                WHERE id = ? AND backend_id = ? AND backend_instance_id = ?
                  AND status IN (?, ?)
                """,
                (
                    stored_status,
                    time.time(),
                    task.id,
                    task.backend_id,
                    task.backend_instance_id,
                    *ASSIGNED_STATUSES,
                ),
            )
        return cursor.rowcount == 1

    def finish(self, task: RouterTask, response: Dict[str, Any]) -> bool:
        """Persist a terminal server response if the assignment is still current."""
        status = response.get("status")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Not a terminal task response: {status}")
        response_json = json.dumps(response, sort_keys=True, separators=(",", ":"))
        with self._transaction():
            cursor = self._connection.execute(
                """
                UPDATE router_tasks
                SET status = ?, response_json = ?, last_error = NULL, updated_at = ?
                WHERE id = ? AND backend_id = ? AND backend_instance_id = ?
                  AND status IN (?, ?)
                """,
                (
                    status,
                    response_json,
                    time.time(),
                    task.id,
                    task.backend_id,
                    task.backend_instance_id,
                    *ASSIGNED_STATUSES,
                ),
            )
        return cursor.rowcount == 1

    def fail_assigned(self, task: RouterTask, error: str) -> bool:
        """Persist a non-retryable dispatch failure for the current assignment."""
        response = {
            "task_id": task.id,
            "status": "failed",
            "definition": task.definition,
            "solution": task.solution_name,
            "traces": None,
            "error": error,
        }
        return self.finish(task, response)

    def requeue(self, task: RouterTask, error: str) -> bool:
        """Return the current assignment to the durable pending queue."""
        with self._transaction():
            cursor = self._connection.execute(
                """
                UPDATE router_tasks
                SET status = 'pending', backend_id = NULL, backend_instance_id = NULL,
                    last_error = ?, updated_at = ?
                WHERE id = ? AND backend_id = ? AND backend_instance_id = ?
                  AND status IN (?, ?)
                """,
                (
                    error,
                    time.time(),
                    task.id,
                    task.backend_id,
                    task.backend_instance_id,
                    *ASSIGNED_STATUSES,
                ),
            )
        return cursor.rowcount == 1

    def requeue_backend(self, backend_id: str, error: str) -> int:
        """Requeue all unfinished assignments owned by an unavailable server."""
        with self._transaction():
            cursor = self._connection.execute(
                """
                UPDATE router_tasks
                SET status = 'pending', backend_id = NULL, backend_instance_id = NULL,
                    last_error = ?, updated_at = ?
                WHERE backend_id = ? AND status IN (?, ?)
                """,
                (error, time.time(), backend_id, *ASSIGNED_STATUSES),
            )
        return cursor.rowcount

    def requeue_other_generation(self, backend_id: str, instance_id: str) -> int:
        """Requeue work recorded against an older process at the same address."""
        with self._transaction():
            cursor = self._connection.execute(
                """
                UPDATE router_tasks
                SET status = 'pending', backend_id = NULL, backend_instance_id = NULL,
                    last_error = ?, updated_at = ?
                WHERE backend_id = ? AND status IN (?, ?)
                  AND (backend_instance_id IS NULL OR backend_instance_id != ?)
                """,
                (
                    f"Server {backend_id} restarted before the task result was recovered",
                    time.time(),
                    backend_id,
                    *ASSIGNED_STATUSES,
                    instance_id,
                ),
            )
        return cursor.rowcount

    def fail_exhausted(self, max_attempts: int) -> int:
        """Fail pending tasks that exhausted infrastructure replay attempts."""
        now = time.time()
        with self._transaction():
            rows = self._connection.execute(
                "SELECT * FROM router_tasks WHERE status = 'pending' AND attempts >= ?",
                (max_attempts,),
            ).fetchall()
            for row in rows:
                task = self._from_row(row)
                response = {
                    "task_id": task.id,
                    "status": "failed",
                    "definition": task.definition,
                    "solution": task.solution_name,
                    "traces": None,
                    "error": f"Infrastructure retry limit reached after {task.attempts} attempts",
                }
                self._connection.execute(
                    "UPDATE router_tasks SET status = 'failed', response_json = ?, "
                    "last_error = ?, updated_at = ? WHERE id = ? AND status = 'pending'",
                    (
                        json.dumps(response, sort_keys=True, separators=(",", ":")),
                        response["error"],
                        now,
                        task.id,
                    ),
                )
        return len(rows)

    def counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM router_tasks GROUP BY status"
            ).fetchall()
        counts = {"pending": 0, "assigned": 0, "running": 0, "completed": 0, "failed": 0}
        counts.update({row["status"]: row["count"] for row in rows})
        return counts

    def cleanup_terminal(self, ttl_seconds: int) -> int:
        cutoff = time.time() - ttl_seconds
        with self._transaction():
            cursor = self._connection.execute(
                "DELETE FROM router_tasks WHERE status IN (?, ?) AND updated_at < ?",
                (*TERMINAL_STATUSES, cutoff),
            )
        return cursor.rowcount

    def _from_row(self, row: sqlite3.Row) -> RouterTask:
        return RouterTask(
            id=row["id"],
            request=json.loads(row["request_json"]),
            definition=row["definition"],
            solution_name=row["solution_name"],
            status=row["status"],
            backend_id=row["backend_id"],
            backend_instance_id=row["backend_instance_id"],
            attempts=row["attempts"],
            response=json.loads(row["response_json"]) if row["response_json"] else None,
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _transaction(self):
        return _Transaction(self._connection, self._lock)


class _Transaction:
    """Small lock-aware SQLite transaction context manager."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock):
        self._connection = connection
        self._lock = lock

    def __enter__(self):
        self._lock.acquire()
        self._connection.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        finally:
            self._lock.release()
