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
from flashinfer_bench.agents.ncu import flashinfer_bench_run_ncu
from flashinfer_bench.agents.sanitizer import flashinfer_bench_run_sanitizer
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


class RunResult(BaseModel):
    definition: str
    workload: Dict[str, Any]
    solution: str
    log: str


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
                "description": "Submit a solution for evaluation",
            },
            {
                "method": "POST",
                "path": "/profile",
                "description": "Run NCU profiling on a solution (synchronous; returns per-workload logs)",
            },
            {
                "method": "POST",
                "path": "/sanitize",
                "description": "Run compute-sanitizer on a solution (synchronous; returns per-workload logs)",
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
    task_id = sched.submit(renamed, req.workload_uuids)
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


def _resolve_workloads(sched: Scheduler, definition: str, workload_uuids: Optional[List[str]]):
    """Return the list of Workload objects for a (definition, uuid-filter) pair."""
    traces = sched.trace_set.workloads.get(definition, [])
    if workload_uuids:
        uuid_set = set(workload_uuids)
        traces = [t for t in traces if t.workload.uuid in uuid_set]
    return [t.workload for t in traces]


def _pick_device(sched: Scheduler) -> str:
    if not sched.workers:
        raise HTTPException(503, detail="No workers available")
    return sched.workers[0].device


@app.post("/profile", response_model=List[RunResult])
async def profile(req: ProfileRequest):
    sched = _get_scheduler()
    if req.solution.definition not in sched.trace_set.definitions:
        raise HTTPException(400, detail=f"Definition not found: {req.solution.definition}")

    workloads = _resolve_workloads(sched, req.solution.definition, req.workload_uuids)
    if not workloads:
        raise HTTPException(
            400, detail=f"No workloads found for definition: {req.solution.definition}"
        )

    device = _pick_device(sched)
    trace_set_path = str(sched.trace_set.root) if sched.trace_set.root else None

    results: List[RunResult] = []
    for wl in workloads:
        log = await asyncio.to_thread(
            flashinfer_bench_run_ncu,
            req.solution,
            wl,
            device=device,
            trace_set_path=trace_set_path,
            set=req.set,
            sections=req.sections,
            kernel_name=req.kernel_name,
            page=req.page,
            ncu_path=req.ncu_path,
            timeout=req.timeout,
            max_lines=req.max_lines,
        )
        results.append(
            RunResult(
                definition=req.solution.definition,
                workload=wl.model_dump(mode="json"),
                solution=req.solution.name,
                log=log,
            )
        )
    return results


@app.post("/sanitize", response_model=List[RunResult])
async def sanitize(req: SanitizeRequest):
    sched = _get_scheduler()
    if req.solution.definition not in sched.trace_set.definitions:
        raise HTTPException(400, detail=f"Definition not found: {req.solution.definition}")

    workloads = _resolve_workloads(sched, req.solution.definition, req.workload_uuids)
    if not workloads:
        raise HTTPException(
            400, detail=f"No workloads found for definition: {req.solution.definition}"
        )

    device = _pick_device(sched)
    trace_set_path = str(sched.trace_set.root) if sched.trace_set.root else None

    results: List[RunResult] = []
    for wl in workloads:
        log = await asyncio.to_thread(
            flashinfer_bench_run_sanitizer,
            req.solution,
            wl,
            device=device,
            trace_set_path=trace_set_path,
            sanitizer_types=req.sanitizer_types,  # type: ignore[arg-type]
            sanitizer_path=req.sanitizer_path,
            timeout=req.timeout,
            max_lines=req.max_lines,
            print_limit=req.print_limit,
        )
        results.append(
            RunResult(
                definition=req.solution.definition,
                workload=wl.model_dump(mode="json"),
                solution=req.solution.name,
                log=log,
            )
        )
    return results


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
