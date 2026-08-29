"""Integration tests for routing and server-failure recovery without a GPU."""

from __future__ import annotations

from typing import Dict

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from flashinfer_bench.data import Solution
from flashinfer_bench.serve.router import BackendConfig, create_router_app
from tests.serve.conftest import solution_correct


class FakeBenchmarkServer:
    """Small in-process implementation of the remote server contract."""

    def __init__(self, server_id: str, dataset_id: str = "shared-dataset"):
        self.server_id = server_id
        self.instance_id = f"{server_id}-generation-1"
        self.dataset_id = dataset_id
        self.accepting = True
        self.health_status_code = 200
        self.tasks: Dict[str, dict] = {}
        self.submission_count = 0
        self.app = FastAPI()
        self._add_routes()

    def _add_routes(self) -> None:
        @self.app.get("/health")
        async def health():
            body = {
                "status": "ok" if self.accepting else "draining",
                "instance_id": self.instance_id,
                "dataset_id": self.dataset_id,
                "accepting": self.accepting,
                "healthy_workers": 1,
                "total_workers": 1,
                "queue_size": 0,
                "active_tasks": len(self.tasks),
            }
            return JSONResponse(body, status_code=self.health_status_code)

        @self.app.post("/evaluate")
        async def evaluate(payload: dict):
            if not self.accepting:
                return JSONResponse({"detail": "draining"}, status_code=503)
            task_id = payload["task_id"]
            existing = self.tasks.get(task_id)
            if existing is None:
                solution = Solution.model_validate(payload["solution"]).with_unique_name()
                existing = {
                    "task_id": task_id,
                    "status": "pending",
                    "definition": solution.definition,
                    "solution": solution.name,
                    "traces": None,
                    "error": None,
                }
                self.tasks[task_id] = existing
                self.submission_count += 1
            return {"task_id": task_id, "normalized_solution_name": existing["solution"]}

        @self.app.post("/tasks/batch")
        async def batch(payload: dict):
            if any(task_id not in self.tasks for task_id in payload["task_ids"]):
                return JSONResponse({"detail": "missing"}, status_code=404)
            return [self.tasks[task_id] for task_id in payload["task_ids"]]

        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            if task_id not in self.tasks:
                return JSONResponse({"detail": "missing"}, status_code=404)
            return self.tasks[task_id]

        @self.app.get("/definitions")
        async def definitions():
            return [{"name": "test_scale", "description": "fake"}]

        @self.app.get("/definitions/{name}")
        async def definition(name: str):
            return {"name": name}

        @self.app.get("/definitions/{name}/workloads")
        async def workloads(name: str):
            return [{"uuid": f"{name}-workload"}]

        @self.app.get("/workloads/{uuid}")
        async def workload(uuid: str):
            return {"uuid": uuid}

    def complete(self, task_id: str) -> None:
        self.tasks[task_id].update(status="completed", traces=[])

    def restart(self) -> None:
        generation = int(self.instance_id.rsplit("-", 1)[1]) + 1
        self.instance_id = f"{self.server_id}-generation-{generation}"
        self.tasks.clear()


def _make_router(tmp_path, servers, **kwargs):
    configs = [BackendConfig(id=server_id, url=f"http://{server_id}") for server_id in servers]

    def transport_factory(config):
        return httpx.ASGITransport(app=servers[config.id].app)

    return create_router_app(
        configs,
        state_db=tmp_path / "router.db",
        health_interval=3600,
        dispatch_interval=3600,
        transport_factory=transport_factory,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_routes_with_capacity_bound_and_batches_results(tmp_path):
    servers = {"node-a": FakeBenchmarkServer("node-a"), "node-b": FakeBenchmarkServer("node-b")}
    app = _make_router(tmp_path, servers)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            solution = solution_correct("test_scale").model_dump(mode="json")
            task_ids = []
            for _ in range(3):
                response = await client.post("/evaluate", json={"solution": solution})
                assert response.status_code == 200
                task_ids.append(response.json()["task_id"])
            await app.state.coordinator.tick()

            assert sum(len(server.tasks) for server in servers.values()) == 2
            assert app.state.task_store.counts()["pending"] == 1
            assert {len(server.tasks) for server in servers.values()} == {1}

            first_server = next(
                server for server in servers.values() if task_ids[0] in server.tasks
            )
            first_server.complete(task_ids[0])
            await app.state.coordinator.tick()

            assert sum(server.submission_count for server in servers.values()) == 3
            first = await client.get(f"/tasks/{task_ids[0]}")
            assert first.json()["status"] == "completed"

            definitions = await client.get("/definitions")
            assert definitions.status_code == 200
            assert definitions.json()[0]["name"] == "test_scale"
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["healthy_backends"] == 2


@pytest.mark.asyncio
async def test_server_restart_replays_same_logical_task(tmp_path):
    server = FakeBenchmarkServer("node-a")
    app = _make_router(tmp_path, {"node-a": server})

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/evaluate",
                json={"solution": solution_correct("test_scale").model_dump(mode="json")},
            )
            task_id = response.json()["task_id"]
            await app.state.coordinator.tick()
            original = app.state.task_store.get_task(task_id)
            assert original.attempts == 1

            server.restart()
            await app.state.backend_pool.probe_all()
            requeued = app.state.task_store.get_task(task_id)
            assert requeued.status == "pending"

            await app.state.coordinator.tick()
            replayed = app.state.task_store.get_task(task_id)
            assert replayed.attempts == 2
            assert replayed.backend_instance_id == server.instance_id
            assert task_id in server.tasks

            server.complete(task_id)
            await app.state.coordinator.tick()
            result = await client.get(f"/tasks/{task_id}")
            assert result.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_consecutive_failures_trigger_requeue_without_new_task_id(tmp_path):
    server = FakeBenchmarkServer("node-a")
    app = _make_router(tmp_path, {"node-a": server}, failure_threshold=2)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/evaluate",
                json={"solution": solution_correct("test_scale").model_dump(mode="json")},
            )
            task_id = response.json()["task_id"]
            await app.state.coordinator.tick()

            server.health_status_code = 503
            await app.state.backend_pool.probe_all()
            assert app.state.task_store.get_task(task_id).status == "assigned"
            assert app.state.backend_pool.snapshots()[0]["state"] == "suspect"

            await app.state.backend_pool.probe_all()
            assert app.state.task_store.get_task(task_id).status == "pending"
            health = await client.get("/health")
            assert health.status_code == 503

            server.health_status_code = 200
            await app.state.backend_pool.probe_all()
            await app.state.coordinator.tick()
            replayed = app.state.task_store.get_task(task_id)
            assert replayed.attempts == 2
            assert list(server.tasks) == [task_id]
            assert server.submission_count == 1


@pytest.mark.asyncio
async def test_mismatched_dataset_is_never_selected(tmp_path):
    servers = {
        "node-a": FakeBenchmarkServer("node-a", dataset_id="dataset-a"),
        "node-b": FakeBenchmarkServer("node-b", dataset_id="dataset-b"),
    }
    app = _make_router(tmp_path, servers)

    async with app.router.lifespan_context(app):
        snapshots = app.state.backend_pool.snapshots()
        assert {snapshot["state"] for snapshot in snapshots} == {"healthy", "incompatible"}
        selected = app.state.backend_pool.select_for_task()
        assert selected is not None
        assert selected.state == "healthy"


@pytest.mark.asyncio
async def test_client_task_id_is_idempotent_and_conflicts_are_rejected(tmp_path):
    server = FakeBenchmarkServer("node-a")
    app = _make_router(tmp_path, {"node-a": server})

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            solution = solution_correct("test_scale").model_dump(mode="json")
            payload = {"solution": solution, "task_id": "client-stable-id"}
            first = await client.post("/evaluate", json=payload)
            second = await client.post("/evaluate", json=payload)
            assert first.status_code == 200
            assert second.status_code == 200
            assert first.json()["task_id"] == "client-stable-id"
            assert second.json()["task_id"] == "client-stable-id"
            assert app.state.task_store.counts()["pending"] == 1

            changed = solution_correct("test_scale").model_copy(update={"name": "changed"})
            conflict = await client.post(
                "/evaluate",
                json={"solution": changed.model_dump(mode="json"), "task_id": "client-stable-id"},
            )
            assert conflict.status_code == 409

            invalid = solution_correct("missing-definition")
            invalid_response = await client.post(
                "/evaluate", json={"solution": invalid.model_dump(mode="json")}
            )
            assert invalid_response.status_code == 400


@pytest.mark.asyncio
async def test_pending_task_dispatches_after_router_restart(tmp_path):
    server = FakeBenchmarkServer("node-a")
    server.health_status_code = 503
    state_db = tmp_path / "router.db"
    app = _make_router(tmp_path, {"node-a": server})

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/evaluate",
                json={"solution": solution_correct("test_scale").model_dump(mode="json")},
            )
            task_id = response.json()["task_id"]
            assert app.state.task_store.get_task(task_id).status == "pending"

    server.health_status_code = 200
    restarted = _make_router(tmp_path, {"node-a": server})
    async with restarted.router.lifespan_context(restarted):
        restored = restarted.state.task_store.get_task(task_id)
        assert restored.attempts == 1
        assert restored.status == "assigned"
        assert task_id in server.tasks
        assert state_db.exists()


@pytest.mark.asyncio
async def test_draining_server_is_removed_and_queue_limit_applies(tmp_path):
    server = FakeBenchmarkServer("node-a")
    app = _make_router(tmp_path, {"node-a": server}, max_active_tasks=1)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            server.accepting = False
            await app.state.backend_pool.probe_all()
            assert app.state.backend_pool.snapshots()[0]["state"] == "draining"

            solution = solution_correct("test_scale").model_dump(mode="json")
            accepted = await client.post("/evaluate", json={"solution": solution})
            rejected = await client.post("/evaluate", json={"solution": solution})
            assert accepted.status_code == 200
            assert rejected.status_code == 503
            await app.state.coordinator.tick()
            assert server.tasks == {}
            assert app.state.task_store.counts()["pending"] == 1
