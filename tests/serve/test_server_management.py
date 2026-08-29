"""CPU-only tests for server health, draining, and idempotent submission."""

import queue
import threading

import httpx
import pytest

from flashinfer_bench.bench import BenchmarkConfig
from flashinfer_bench.data import TraceSet
from flashinfer_bench.serve.app import app, init_app
from flashinfer_bench.serve.scheduler import Scheduler, _GPUWorkerThread
from flashinfer_bench.serve.task_store import TaskStore
from tests.serve.conftest import solution_correct


@pytest.mark.asyncio
async def test_health_drain_and_idempotent_task_id(test_trace_set):
    scheduler = Scheduler(
        trace_set=test_trace_set,
        config=BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1),
        devices=[],
    )
    init_app(scheduler)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://server"
        ) as client:
            solution = solution_correct("test_scale").model_dump(mode="json")
            payload = {"solution": solution, "task_id": "stable-task-id"}
            first = await client.post("/evaluate", json=payload)
            second = await client.post("/evaluate", json=payload)
            assert first.status_code == 200
            assert second.status_code == 200
            assert first.json()["task_id"] == second.json()["task_id"]
            assert scheduler.queue_size == 1

            conflicting = solution_correct("test_scale").model_copy(
                update={"name": "different-name"}
            )
            conflict = await client.post(
                "/evaluate",
                json={"solution": conflicting.model_dump(mode="json"), "task_id": "stable-task-id"},
            )
            assert conflict.status_code == 409

            drained = await client.post("/drain")
            assert drained.status_code == 200
            rejected = await client.post(
                "/evaluate", json={"solution": solution, "task_id": "new-task-id"}
            )
            assert rejected.status_code == 503

            health = await client.get("/health")
            body = health.json()
            assert body["status"] == "draining"
            assert body["accepting"] is False
            assert body["instance_id"]
            assert body["dataset_id"]
            assert body["healthy_workers"] == 0
    finally:
        scheduler.shutdown()


def test_idle_worker_recovers_after_restart_failure(monkeypatch):
    second_worker_started = threading.Event()

    class FakePersistentWorker:
        instances = 0

        def __init__(self, device):
            self.device = device
            self.number = FakePersistentWorker.instances
            FakePersistentWorker.instances += 1
            if self.number == 1:
                second_worker_started.set()

        def is_healthy(self):
            return self.number > 0

        def restart(self):
            return False

        def close(self):
            pass

    monkeypatch.setattr(
        "flashinfer_bench.serve.scheduler.PersistentSubprocessWorker", FakePersistentWorker
    )
    shutdown = threading.Event()
    worker = _GPUWorkerThread(
        device="cuda:0",
        task_queue=queue.Queue(),
        task_store=TaskStore(),
        trace_set=TraceSet(),
        config=BenchmarkConfig(),
        shutdown_event=shutdown,
    )
    worker.start()
    try:
        assert second_worker_started.wait(timeout=3)
        with worker._worker_lock:
            pass
        assert FakePersistentWorker.instances == 2
        assert worker.is_healthy is True
    finally:
        shutdown.set()
        worker.join(timeout=3)
        worker.close()
