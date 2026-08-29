"""FastAPI application for the benchmark server."""

import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from flashinfer_bench import __version__
from flashinfer_bench.data import Solution
from flashinfer_bench.serve.scheduler import Scheduler, SchedulerDrainingError
from flashinfer_bench.serve.task_store import TaskConflictError

logger = logging.getLogger(__name__)

_scheduler: Optional[Scheduler] = None


def _get_scheduler() -> Scheduler:
    if _scheduler is None:
        raise RuntimeError("Server not initialized. Call init_app() first.")
    return _scheduler


# ── Request / Response models ──


class EvaluateRequest(BaseModel):
    solution: Solution
    workload_uuids: Optional[List[str]] = None
    task_id: Optional[str] = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"
    )


class EvaluateResponse(BaseModel):
    task_id: str
    normalized_solution_name: str


class TaskResponse(BaseModel):
    task_id: str
    status: str
    definition: str
    solution: str
    traces: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class DefinitionInfo(BaseModel):
    name: str
    description: Optional[str] = None


class BatchRequest(BaseModel):
    task_ids: List[str]
    timeout: float = 0


class WorkerInfo(BaseModel):
    device: str
    healthy: bool


class HealthResponse(BaseModel):
    status: str
    instance_id: str
    dataset_id: str
    started_at: float
    accepting: bool
    workers: List[WorkerInfo]
    healthy_workers: int
    total_workers: int
    queue_size: int
    active_tasks: int


# ── App & routes ──


@asynccontextmanager
async def _lifespan(app):
    yield
    if _scheduler is not None:
        _scheduler.shutdown()


app = FastAPI(title="FlashInfer-Bench Server", version=__version__, lifespan=_lifespan)


def init_app(scheduler: Scheduler) -> FastAPI:
    """Inject the scheduler into the module-level app."""
    global _scheduler
    _scheduler = scheduler
    return app


@app.get("/")
async def root():
    """Root endpoint returning server info and available endpoints."""
    return {
        "name": "FlashInfer-Bench Server",
        "version": __version__,
        "docs": "/docs",
        "endpoints": [
            {"method": "GET", "path": "/", "description": "Server info and endpoint discovery"},
            {"method": "GET", "path": "/docs", "description": "Interactive Swagger UI"},
            {"method": "GET", "path": "/health", "description": "Server health and worker status"},
            {"method": "POST", "path": "/drain", "description": "Stop accepting new tasks"},
            {"method": "GET", "path": "/definitions", "description": "List all loaded definitions"},
            {
                "method": "GET",
                "path": "/definitions/{name}",
                "description": "Get a definition by name",
            },
            {
                "method": "GET",
                "path": "/definitions/{name}/workloads",
                "description": "List workloads for a definition",
            },
            {"method": "GET", "path": "/workloads/{uuid}", "description": "Get a workload by UUID"},
            {
                "method": "POST",
                "path": "/evaluate",
                "description": "Submit a solution for evaluation",
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}",
                "description": "Get task status and results",
            },
            {"method": "POST", "path": "/tasks/batch", "description": "Batch get multiple tasks"},
        ],
    }


@app.get("/definitions", response_model=List[DefinitionInfo])
async def list_definitions():
    sched = _get_scheduler()
    result = []
    for name, defn in sched.trace_set.definitions.items():
        result.append(DefinitionInfo(name=name, description=defn.description))
    return result


@app.get("/definitions/{name}")
async def get_definition(name: str):
    sched = _get_scheduler()
    defn = sched.trace_set.definitions.get(name)
    if defn is None:
        raise HTTPException(404, detail=f"Definition not found: {name}")
    return defn.model_dump(mode="json")


@app.get("/definitions/{name}/workloads")
async def list_workloads(name: str):
    sched = _get_scheduler()
    if name not in sched.trace_set.definitions:
        raise HTTPException(404, detail=f"Definition not found: {name}")
    traces = sched.trace_set.workloads.get(name, [])
    return [t.workload.model_dump(mode="json") for t in traces]


@app.get("/workloads/{uuid}")
async def get_workload(uuid: str):
    sched = _get_scheduler()
    for traces in sched.trace_set.workloads.values():
        for t in traces:
            if t.workload.uuid == uuid:
                return t.workload.model_dump(mode="json")
    raise HTTPException(404, detail=f"Workload not found: {uuid}")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    sched = _get_scheduler()
    if req.solution.definition not in sched.trace_set.definitions:
        raise HTTPException(400, detail=f"Definition not found: {req.solution.definition}")
    renamed = req.solution.with_unique_name()
    try:
        task_id = sched.submit(renamed, req.workload_uuids, task_id=req.task_id)
    except SchedulerDrainingError as e:
        raise HTTPException(503, detail=str(e)) from e
    except TaskConflictError as e:
        raise HTTPException(409, detail=str(e)) from e
    return EvaluateResponse(task_id=task_id, normalized_solution_name=renamed.name)


@app.post("/tasks/batch", response_model=List[TaskResponse])
async def batch_get_tasks(req: BatchRequest):
    sched = _get_scheduler()
    for task_id in req.task_ids:
        if sched.task_store.get_task(task_id) is None:
            raise HTTPException(404, detail=f"Task not found: {task_id}")

    if req.timeout > 0:
        await asyncio.to_thread(sched.task_store.wait_for_all, req.task_ids, req.timeout)

    results = []
    for tid in req.task_ids:
        task = sched.task_store.get_task(tid)
        traces_data = [t.model_dump(mode="json") for t in task.traces] if task.traces else None
        results.append(
            TaskResponse(
                task_id=task.id,
                status=task.status,
                definition=task.definition_name,
                solution=task.solution.name,
                traces=traces_data,
                error=task.error,
            )
        )
    return results


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, timeout: float = Query(default=0, ge=0, le=3600)):
    results = await batch_get_tasks(BatchRequest(task_ids=[task_id], timeout=timeout))
    return results[0]


@app.get("/health", response_model=HealthResponse)
async def health():
    sched = _get_scheduler()
    workers = [WorkerInfo(device=w.device, healthy=w.is_healthy) for w in sched.workers]
    healthy_workers = sum(worker.healthy for worker in workers)
    if not sched.accepting:
        status = "draining"
    elif workers and healthy_workers == len(workers):
        status = "ok"
    elif healthy_workers:
        status = "degraded"
    else:
        status = "unavailable"
    return HealthResponse(
        status=status,
        instance_id=sched.instance_id,
        dataset_id=sched.dataset_id,
        started_at=sched.started_at,
        accepting=sched.accepting,
        workers=workers,
        healthy_workers=healthy_workers,
        total_workers=len(workers),
        queue_size=sched.queue_size,
        active_tasks=sched.active_tasks,
    )


@app.post("/drain")
async def drain():
    """Stop accepting tasks and let already admitted work finish."""
    sched = _get_scheduler()
    sched.drain()
    return {
        "status": "draining",
        "active_tasks": sched.active_tasks,
        "queue_size": sched.queue_size,
    }


@app.post("/shutdown")
async def shutdown():
    """Gracefully shut down the server."""
    os.kill(os.getpid(), signal.SIGINT)
    return {"status": "shutting_down"}
