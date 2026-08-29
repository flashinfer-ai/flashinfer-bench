"""Durable multi-node router for benchmark servers."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from flashinfer_bench import __version__
from flashinfer_bench.serve.app import BatchRequest, EvaluateRequest, EvaluateResponse, TaskResponse
from flashinfer_bench.serve.router_store import (
    RouterQueueFull,
    RouterTask,
    RouterTaskConflict,
    RouterTaskStore,
)

logger = logging.getLogger(__name__)

_BACKEND_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class BackendConfig:
    """Static connection and scheduling configuration for one benchmark server."""

    id: str
    url: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not _BACKEND_ID_RE.fullmatch(self.id):
            raise ValueError(
                f"Invalid backend ID {self.id!r}; use letters, numbers, '.', '_', or '-'"
            )
        parsed = urlparse(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid backend URL: {self.url}")
        if parsed.path not in ("", "/") or parsed.params or parsed.query or parsed.fragment:
            raise ValueError(f"Backend URL must not include a path, query, or fragment: {self.url}")
        if self.weight <= 0:
            raise ValueError("Backend weight must be greater than zero")
        object.__setattr__(self, "url", self.url.rstrip("/"))


@dataclass
class _Backend:
    config: BackendConfig
    client: httpx.AsyncClient
    state: str = "starting"
    instance_id: Optional[str] = None
    dataset_id: Optional[str] = None
    accepting: bool = False
    healthy_workers: int = 0
    total_workers: int = 0
    remote_queue_size: int = 0
    active_tasks: int = 0
    consecutive_failures: int = 0
    last_success_at: Optional[float] = None
    last_error: Optional[str] = None


class BackendPool:
    """Tracks server health, compatibility, and bounded scheduling capacity."""

    def __init__(
        self,
        configs: List[BackendConfig],
        store: RouterTaskStore,
        *,
        health_interval: float = 5.0,
        failure_threshold: int = 3,
        request_timeout: float = 10.0,
        max_inflight_per_worker: int = 1,
        transport_factory: Optional[Callable[[BackendConfig], httpx.AsyncBaseTransport]] = None,
    ):
        if not configs:
            raise ValueError("At least one backend is required")
        if len({config.id for config in configs}) != len(configs):
            raise ValueError("Backend IDs must be unique")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least one")
        if max_inflight_per_worker < 1:
            raise ValueError("max_inflight_per_worker must be at least one")

        self._store = store
        self._health_interval = health_interval
        self._failure_threshold = failure_threshold
        self._max_inflight_per_worker = max_inflight_per_worker
        self._backends: Dict[str, _Backend] = {}
        for config in configs:
            transport = transport_factory(config) if transport_factory else None
            client = httpx.AsyncClient(
                base_url=config.url, timeout=request_timeout, transport=transport
            )
            self._backends[config.id] = _Backend(config=config, client=client)
        self._expected_dataset_id: Optional[str] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._closed = False
        self._tie_break = 0
        self._definition_names: Optional[set[str]] = None

    async def start(self) -> None:
        await self.probe_all()
        await self.refresh_definitions()
        self._monitor_task = asyncio.create_task(self._monitor(), name="router-health-monitor")

    async def close(self) -> None:
        self._closed = True
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            await asyncio.gather(self._monitor_task, return_exceptions=True)
        await asyncio.gather(
            *(backend.client.aclose() for backend in self._backends.values()),
            return_exceptions=True,
        )

    async def _monitor(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self._health_interval)
                await self.probe_all()
                if self._definition_names is None:
                    await self.refresh_definitions()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected failure in router health monitor")

    async def probe_all(self) -> None:
        backends = list(self._backends.values())
        results = await asyncio.gather(*(self._fetch_health(backend) for backend in backends))
        if self._expected_dataset_id is None:
            datasets = [result[0]["dataset_id"] for result in results if result[0] is not None]
            if datasets:
                counts = Counter(datasets)
                highest_count = max(counts.values())
                self._expected_dataset_id = next(
                    dataset for dataset in datasets if counts[dataset] == highest_count
                )
        for backend, (health, error) in zip(backends, results):
            if health is None:
                self._record_probe_failure(backend, str(error))
            else:
                self._apply_health(backend, health)

    async def _fetch_health(self, backend: _Backend) -> tuple[Optional[dict], Optional[Exception]]:
        try:
            response = await backend.client.get("/health")
            response.raise_for_status()
            health = response.json()
            health["instance_id"] = str(health["instance_id"])
            health["dataset_id"] = str(health["dataset_id"])
            health["healthy_workers"] = int(health["healthy_workers"])
            health["total_workers"] = int(health["total_workers"])
            health["accepting"] = bool(health["accepting"])
            return health, None
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return None, error

    def _apply_health(self, backend: _Backend, health: dict) -> None:
        instance_id = health["instance_id"]
        dataset_id = health["dataset_id"]
        healthy_workers = health["healthy_workers"]
        total_workers = health["total_workers"]
        accepting = health["accepting"]
        backend.instance_id = instance_id
        backend.dataset_id = dataset_id
        backend.accepting = accepting
        backend.healthy_workers = healthy_workers
        backend.total_workers = total_workers
        backend.remote_queue_size = int(health.get("queue_size", 0))
        backend.active_tasks = int(health.get("active_tasks", 0))
        backend.last_success_at = time.time()
        backend.last_error = None

        assert self._expected_dataset_id is not None
        if dataset_id != self._expected_dataset_id:
            backend.state = "incompatible"
            backend.consecutive_failures = 0
            backend.last_error = (
                f"Dataset {dataset_id} does not match router dataset {self._expected_dataset_id}"
            )
            requeued = self._store.requeue_backend(backend.config.id, backend.last_error)
            if requeued:
                logger.warning(
                    "Requeued %d tasks from incompatible server %s", requeued, backend.config.id
                )
            return

        requeued = self._store.requeue_other_generation(backend.config.id, instance_id)
        if requeued:
            logger.warning(
                "Server %s has a new process generation; requeued %d tasks",
                backend.config.id,
                requeued,
            )

        if not accepting or health.get("status") == "draining":
            backend.state = "draining"
            backend.consecutive_failures = 0
        elif healthy_workers > 0:
            backend.state = "healthy"
            backend.consecutive_failures = 0
        else:
            self._record_probe_failure(backend, "Server has no healthy GPU workers")

    def _record_probe_failure(self, backend: _Backend, error: str) -> None:
        backend.consecutive_failures += 1
        backend.last_error = error
        backend.accepting = False
        if backend.consecutive_failures >= self._failure_threshold:
            if backend.state != "unavailable":
                logger.warning(
                    "Server %s unavailable after %d failed probes: %s",
                    backend.config.id,
                    backend.consecutive_failures,
                    error,
                )
            backend.state = "unavailable"
            requeued = self._store.requeue_backend(
                backend.config.id, f"Server {backend.config.id} became unavailable: {error}"
            )
            if requeued:
                logger.warning("Requeued %d tasks from server %s", requeued, backend.config.id)
        else:
            backend.state = "suspect"

    def select_for_task(
        self, assignment_counts: Optional[Dict[str, int]] = None
    ) -> Optional[_Backend]:
        """Choose the least-utilized compatible server that has a free slot."""
        counts = (
            assignment_counts if assignment_counts is not None else self._store.assignment_counts()
        )
        candidates = []
        for backend in self._backends.values():
            if backend.state != "healthy" or not backend.instance_id:
                continue
            assigned = counts.get(backend.config.id, 0)
            capacity = backend.healthy_workers * self._max_inflight_per_worker
            if assigned >= capacity:
                continue
            utilization = assigned / (capacity * backend.config.weight)
            candidates.append((utilization, backend.config.id, backend))
        if not candidates:
            return None
        minimum = min(candidate[0] for candidate in candidates)
        tied = sorted(
            (candidate[2] for candidate in candidates if candidate[0] == minimum),
            key=lambda backend: backend.config.id,
        )
        selected = tied[self._tie_break % len(tied)]
        self._tie_break += 1
        return selected

    def select_for_metadata(self) -> Optional[_Backend]:
        candidates = sorted(
            (backend for backend in self._backends.values() if backend.state == "healthy"),
            key=lambda backend: backend.config.id,
        )
        if not candidates:
            return None
        selected = candidates[self._tie_break % len(candidates)]
        self._tie_break += 1
        return selected

    def get(self, backend_id: str) -> Optional[_Backend]:
        return self._backends.get(backend_id)

    async def refresh_definitions(self) -> None:
        """Cache definition names to preserve direct-server submission validation."""
        backend = self.select_for_metadata()
        if backend is None:
            return
        try:
            response = await backend.client.get("/definitions")
            response.raise_for_status()
            self._definition_names = {item["name"] for item in response.json()}
        except Exception as error:
            logger.warning("Could not refresh router definition catalog: %s", error)

    def definition_exists(self, name: str) -> Optional[bool]:
        if self._definition_names is None:
            return None
        return name in self._definition_names

    def snapshots(self) -> List[dict]:
        counts = self._store.assignment_counts()
        result = []
        for backend in sorted(self._backends.values(), key=lambda item: item.config.id):
            capacity = backend.healthy_workers * self._max_inflight_per_worker
            result.append(
                {
                    "id": backend.config.id,
                    "url": backend.config.url,
                    "state": backend.state,
                    "instance_id": backend.instance_id,
                    "dataset_id": backend.dataset_id,
                    "accepting": backend.accepting,
                    "healthy_workers": backend.healthy_workers,
                    "total_workers": backend.total_workers,
                    "capacity": capacity,
                    "assigned_tasks": counts.get(backend.config.id, 0),
                    "remote_queue_size": backend.remote_queue_size,
                    "active_tasks": backend.active_tasks,
                    "consecutive_failures": backend.consecutive_failures,
                    "last_success_at": backend.last_success_at,
                    "last_error": backend.last_error,
                }
            )
        return result


class RouterCoordinator:
    """Dispatches durable tasks and reconciles their remote status in batches."""

    def __init__(
        self,
        store: RouterTaskStore,
        pool: BackendPool,
        *,
        dispatch_interval: float = 0.5,
        max_attempts: int = 5,
        result_ttl_seconds: int = 86_400,
        batch_size: int = 100,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if dispatch_interval <= 0:
            raise ValueError("dispatch_interval must be greater than zero")
        if result_ttl_seconds < 1:
            raise ValueError("result_ttl_seconds must be at least one")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        self._store = store
        self._pool = pool
        self._dispatch_interval = dispatch_interval
        self._max_attempts = max_attempts
        self._result_ttl_seconds = result_ttl_seconds
        self._batch_size = batch_size
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        self._tick_lock = asyncio.Lock()
        self._last_cleanup = 0.0
        self._wake = asyncio.Event()

    async def start(self) -> None:
        await self._pool.start()
        await self.tick()
        self._task = asyncio.create_task(self._run(), name="router-task-coordinator")

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._pool.close()

    async def _run(self) -> None:
        while not self._closed:
            try:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self._dispatch_interval)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected failure in router task coordinator")

    def wake(self) -> None:
        """Request a background pass without delaying the caller on server I/O."""
        self._wake.set()

    async def tick(self) -> None:
        async with self._tick_lock:
            self._store.fail_exhausted(self._max_attempts)
            await self._reconcile()
            await self._dispatch()
            if time.monotonic() - self._last_cleanup >= 60:
                self._store.cleanup_terminal(self._result_ttl_seconds)
                self._last_cleanup = time.monotonic()

    async def _dispatch(self) -> None:
        submissions = []
        assignment_counts = self._store.assignment_counts()
        for task in self._store.list_pending(self._batch_size):
            backend = self._pool.select_for_task(assignment_counts)
            if backend is None or backend.instance_id is None:
                break
            if not self._store.assign(task.id, backend.config.id, backend.instance_id):
                continue
            assignment_counts[backend.config.id] = assignment_counts.get(backend.config.id, 0) + 1
            assigned = self._store.get_task(task.id)
            assert assigned is not None
            submissions.append(self._submit(assigned, backend))
        if submissions:
            await asyncio.gather(*submissions)

    async def _submit(self, task: RouterTask, backend: _Backend) -> None:
        payload = dict(task.request)
        payload["task_id"] = task.id
        try:
            response = await backend.client.post("/evaluate", json=payload)
        except (httpx.HTTPError, asyncio.TimeoutError) as error:
            logger.warning(
                "Dispatch response for task %s from server %s was lost: %s",
                task.id,
                backend.config.id,
                error,
            )
            return

        if response.status_code == 200:
            try:
                remote_task_id = response.json()["task_id"]
            except Exception:
                logger.warning("Server %s returned an invalid evaluate response", backend.config.id)
                return
            if remote_task_id != task.id:
                self._store.fail_assigned(
                    task, f"Server {backend.config.id} returned mismatched task ID {remote_task_id}"
                )
            return

        detail = _response_error(response)
        if response.status_code in (400, 409, 422):
            self._store.fail_assigned(task, detail)
        elif response.status_code == 503:
            self._store.requeue(task, f"Server {backend.config.id} rejected dispatch: {detail}")
            backend.state = "draining"
            backend.accepting = False

    async def _reconcile(self) -> None:
        requests = []
        backend_ids = {
            task.backend_id for task in self._store.list_assigned() if task.backend_id is not None
        }
        for backend_id in sorted(backend_ids):
            backend = self._pool.get(backend_id)
            if backend is None or backend.state in (
                "starting",
                "suspect",
                "unavailable",
                "incompatible",
            ):
                continue
            tasks = self._store.list_assigned(backend_id)
            for offset in range(0, len(tasks), self._batch_size):
                requests.append(
                    self._reconcile_batch(backend, tasks[offset : offset + self._batch_size])
                )
        if requests:
            await asyncio.gather(*requests)

    async def _reconcile_batch(self, backend: _Backend, tasks: List[RouterTask]) -> None:
        try:
            response = await backend.client.post(
                "/tasks/batch", json={"task_ids": [task.id for task in tasks], "timeout": 0}
            )
        except (httpx.HTTPError, asyncio.TimeoutError):
            return
        if response.status_code == 404:
            await asyncio.gather(*(self._reconcile_one(backend, task) for task in tasks))
            return
        if response.status_code != 200:
            return
        try:
            responses = {item["task_id"]: item for item in response.json()}
        except Exception:
            logger.warning("Server %s returned an invalid batch task response", backend.config.id)
            return
        for task in tasks:
            item = responses.get(task.id)
            if item is not None:
                self._record_remote_response(task, item)

    async def _reconcile_one(self, backend: _Backend, task: RouterTask) -> None:
        try:
            response = await backend.client.get(f"/tasks/{task.id}")
        except (httpx.HTTPError, asyncio.TimeoutError):
            return
        if response.status_code == 404:
            self._store.requeue(task, f"Server {backend.config.id} no longer has task {task.id}")
        elif response.status_code == 200:
            try:
                self._record_remote_response(task, response.json())
            except Exception:
                logger.warning("Server %s returned an invalid task response", backend.config.id)

    def _record_remote_response(self, task: RouterTask, response: dict) -> None:
        status = response.get("status")
        if status in ("completed", "failed"):
            self._store.finish(task, response)
        elif status in ("pending", "running"):
            self._store.mark_remote_status(task, status)


def create_router_app(
    backends: List[BackendConfig],
    *,
    state_db: Path | str,
    max_active_tasks: int = 10_000,
    health_interval: float = 5.0,
    failure_threshold: int = 3,
    request_timeout: float = 10.0,
    max_inflight_per_worker: int = 1,
    dispatch_interval: float = 0.5,
    max_attempts: int = 5,
    result_ttl_seconds: int = 86_400,
    transport_factory: Optional[Callable[[BackendConfig], httpx.AsyncBaseTransport]] = None,
) -> FastAPI:
    """Build an isolated router application and its durable coordinator."""
    store = RouterTaskStore(state_db, max_active_tasks=max_active_tasks)
    pool = BackendPool(
        backends,
        store,
        health_interval=health_interval,
        failure_threshold=failure_threshold,
        request_timeout=request_timeout,
        max_inflight_per_worker=max_inflight_per_worker,
        transport_factory=transport_factory,
    )
    coordinator = RouterCoordinator(
        store,
        pool,
        dispatch_interval=dispatch_interval,
        max_attempts=max_attempts,
        result_ttl_seconds=result_ttl_seconds,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await coordinator.start()
        try:
            yield
        finally:
            await coordinator.close()
            store.close()

    router_app = FastAPI(title="FlashInfer-Bench Router", version=__version__, lifespan=lifespan)
    router_app.state.task_store = store
    router_app.state.backend_pool = pool
    router_app.state.coordinator = coordinator

    @router_app.get("/")
    async def root():
        return {
            "name": "FlashInfer-Bench Router",
            "version": __version__,
            "docs": "/docs",
            "endpoints": [
                {"method": "GET", "path": "/live", "description": "Router process liveness"},
                {"method": "GET", "path": "/health", "description": "Router readiness"},
                {"method": "GET", "path": "/backends", "description": "Server pool status"},
                {"method": "POST", "path": "/evaluate", "description": "Submit evaluation"},
                {"method": "GET", "path": "/tasks/{task_id}", "description": "Get task"},
                {"method": "POST", "path": "/tasks/batch", "description": "Batch get tasks"},
            ],
        }

    @router_app.get("/live")
    async def live():
        return {"status": "ok"}

    @router_app.get("/health")
    async def health():
        snapshots = pool.snapshots()
        healthy = sum(item["state"] == "healthy" for item in snapshots)
        body = {
            "status": "ok" if healthy else "unavailable",
            "healthy_backends": healthy,
            "total_backends": len(snapshots),
            "tasks": store.counts(),
        }
        return JSONResponse(content=body, status_code=200 if healthy else 503)

    @router_app.get("/backends")
    async def list_backends():
        return pool.snapshots()

    @router_app.post("/evaluate", response_model=EvaluateResponse)
    async def evaluate(request: EvaluateRequest):
        if pool.definition_exists(request.solution.definition) is False:
            raise HTTPException(400, detail=f"Definition not found: {request.solution.definition}")
        normalized = request.solution.with_unique_name()
        persisted_request = request.model_dump(mode="json", exclude={"task_id"})
        try:
            task = store.create_task(
                persisted_request,
                definition=request.solution.definition,
                solution_name=normalized.name,
                task_id=request.task_id,
            )
        except RouterQueueFull as error:
            raise HTTPException(503, detail=str(error)) from error
        except RouterTaskConflict as error:
            raise HTTPException(409, detail=str(error)) from error
        coordinator.wake()
        return EvaluateResponse(task_id=task.id, normalized_solution_name=normalized.name)

    @router_app.post("/tasks/batch", response_model=List[TaskResponse])
    async def batch_get_tasks(request: BatchRequest):
        tasks = [store.get_task(task_id) for task_id in request.task_ids]
        missing = next(
            (task_id for task_id, task in zip(request.task_ids, tasks) if task is None), None
        )
        if missing is not None:
            raise HTTPException(404, detail=f"Task not found: {missing}")

        if request.timeout > 0:
            deadline = time.monotonic() + request.timeout
            while any(
                task is not None and task.status not in ("completed", "failed") for task in tasks
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(0.1, remaining))
                tasks = [store.get_task(task_id) for task_id in request.task_ids]

        return [_task_response(task) for task in tasks if task is not None]

    @router_app.get("/tasks/{task_id}", response_model=TaskResponse)
    async def get_task(task_id: str, timeout: float = Query(default=0, ge=0, le=3600)):
        results = await batch_get_tasks(BatchRequest(task_ids=[task_id], timeout=timeout))
        return results[0]

    async def proxy_get(path: str) -> Response:
        backend = pool.select_for_metadata()
        if backend is None:
            raise HTTPException(503, detail="No healthy benchmark server is available")
        try:
            response = await backend.client.get(path)
        except httpx.HTTPError as error:
            raise HTTPException(503, detail=f"Benchmark server request failed: {error}") from error
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json").split(";")[0],
        )

    @router_app.get("/definitions")
    async def list_definitions():
        return await proxy_get("/definitions")

    @router_app.get("/definitions/{name}")
    async def get_definition(name: str):
        return await proxy_get(f"/definitions/{name}")

    @router_app.get("/definitions/{name}/workloads")
    async def list_workloads(name: str):
        return await proxy_get(f"/definitions/{name}/workloads")

    @router_app.get("/workloads/{uuid}")
    async def get_workload(uuid: str):
        return await proxy_get(f"/workloads/{uuid}")

    return router_app


def parse_backend(value: str) -> BackendConfig:
    """Parse ``NAME=URL`` from the command line."""
    if "=" not in value:
        raise ValueError("Backend must use NAME=URL format")
    backend_id, url = value.split("=", 1)
    return BackendConfig(id=backend_id, url=url)


def _task_response(task: RouterTask) -> TaskResponse:
    if task.response is not None:
        return TaskResponse(**task.response)
    public_status = "running" if task.status == "running" else "pending"
    return TaskResponse(
        task_id=task.id,
        status=public_status,
        definition=task.definition,
        solution=task.solution_name,
        traces=None,
        error=None,
    )


def _response_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("detail", payload))
    except Exception:
        return f"HTTP {response.status_code}: {response.text[:500]}"
