"""FastAPI application for the benchmark server."""

import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from flashinfer_bench import __version__
from flashinfer_bench.data import Solution
from flashinfer_bench.serve.scheduler import Scheduler

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


class EvaluateResponse(BaseModel):
    task_id: str
    normalized_solution_name: str


class RunResult(BaseModel):
    definition: str
    workload: Dict[str, Any]
    solution: str
    log: str


class TaskResponse(BaseModel):
    task_id: str
    kind: str
    status: str
    definition: str
    solution: str
    traces: Optional[List[Dict[str, Any]]] = None
    logs: Optional[List[RunResult]] = None
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
    workers: List[WorkerInfo]
    queue_size: int


class ProfileRequest(BaseModel):
    solution: Solution
    workload_uuids: Optional[List[str]] = None
    # NCU configuration (mirrors flashinfer_bench_run_ncu)
    set: str = "detailed"
    sections: Optional[List[str]] = None
    kernel_name: Optional[str] = None
    page: str = "details"
    ncu_path: str = "ncu"
    timeout: int = 60
    max_lines: Optional[int] = None


class SanitizeRequest(BaseModel):
    solution: Solution
    workload_uuids: Optional[List[str]] = None
    # Sanitizer configuration (mirrors flashinfer_bench_run_sanitizer)
    sanitizer_types: Optional[List[str]] = None
    sanitizer_path: str = "compute-sanitizer"
    timeout: int = 300
    max_lines: Optional[int] = None
    print_limit: Optional[int] = None


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
                "description": "Submit a solution for evaluation (returns task_id; poll /tasks/{task_id})",
            },
            {
                "method": "POST",
                "path": "/profile",
                "description": "Submit a solution for NCU profiling (returns task_id; poll /tasks/{task_id})",
            },
            {
                "method": "POST",
                "path": "/sanitize",
                "description": "Submit a solution for compute-sanitizer checks (returns task_id; poll /tasks/{task_id})",
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
    task_id = sched.submit_evaluate(renamed, req.workload_uuids)
    return EvaluateResponse(task_id=task_id, normalized_solution_name=renamed.name)


@app.post("/profile", response_model=EvaluateResponse)
async def profile(req: ProfileRequest):
    sched = _get_scheduler()
    if req.solution.definition not in sched.trace_set.definitions:
        raise HTTPException(400, detail=f"Definition not found: {req.solution.definition}")
    renamed = req.solution.with_unique_name()
    task_id = sched.submit_profile(
        renamed,
        req.workload_uuids,
        ncu_set=req.set,
        ncu_sections=req.sections,
        ncu_kernel_name=req.kernel_name,
        ncu_page=req.page,
        ncu_path=req.ncu_path,
        ncu_timeout=req.timeout,
        ncu_max_lines=req.max_lines,
    )
    return EvaluateResponse(task_id=task_id, normalized_solution_name=renamed.name)


@app.post("/sanitize", response_model=EvaluateResponse)
async def sanitize(req: SanitizeRequest):
    sched = _get_scheduler()
    if req.solution.definition not in sched.trace_set.definitions:
        raise HTTPException(400, detail=f"Definition not found: {req.solution.definition}")
    renamed = req.solution.with_unique_name()
    task_id = sched.submit_sanitize(
        renamed,
        req.workload_uuids,
        sanitizer_types=req.sanitizer_types,
        sanitizer_path=req.sanitizer_path,
        sanitizer_timeout=req.timeout,
        sanitizer_max_lines=req.max_lines,
        sanitizer_print_limit=req.print_limit,
    )
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
        logs_data = (
            [
                RunResult(
                    definition=lg.definition, workload=lg.workload, solution=lg.solution, log=lg.log
                )
                for lg in task.logs
            ]
            if task.logs
            else None
        )
        results.append(
            TaskResponse(
                task_id=task.id,
                kind=task.kind.value,
                status=task.status,
                definition=task.definition_name,
                solution=task.solution.name,
                traces=traces_data,
                logs=logs_data,
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
    return HealthResponse(status="ok", workers=workers, queue_size=sched.queue_size)


@app.post("/shutdown")
async def shutdown():
    """Gracefully shut down the server."""
    os.kill(os.getpid(), signal.SIGINT)
    return {"status": "shutting_down"}
