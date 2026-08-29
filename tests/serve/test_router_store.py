"""Tests for durable router task state."""

import pytest

from flashinfer_bench.serve.router_store import RouterQueueFull, RouterTaskConflict, RouterTaskStore


def _request(name: str = "solution") -> dict:
    return {"solution": {"name": name}, "workload_uuids": None}


def _terminal_response(task_id: str, solution: str = "solution_hash") -> dict:
    return {
        "task_id": task_id,
        "status": "completed",
        "definition": "definition",
        "solution": solution,
        "traces": [],
        "error": None,
    }


def test_task_state_survives_store_restart(tmp_path):
    path = tmp_path / "router.db"
    store = RouterTaskStore(path)
    task = store.create_task(_request(), "definition", "solution_hash")
    assert store.assign(task.id, "node-a", "generation-1") is True

    assigned = store.get_task(task.id)
    assert assigned is not None
    assert assigned.status == "assigned"
    assert assigned.attempts == 1
    assert store.finish(assigned, _terminal_response(task.id)) is True
    store.close()

    reopened = RouterTaskStore(path)
    completed = reopened.get_task(task.id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.response == _terminal_response(task.id)
    assert reopened.counts()["completed"] == 1
    reopened.close()


def test_generation_change_only_requeues_old_assignments(tmp_path):
    store = RouterTaskStore(tmp_path / "router.db")
    old = store.create_task(_request("old"), "definition", "old_hash")
    current = store.create_task(_request("current"), "definition", "current_hash")
    store.assign(old.id, "node-a", "generation-1")
    store.assign(current.id, "node-a", "generation-2")

    assert store.requeue_other_generation("node-a", "generation-2") == 1
    assert store.get_task(old.id).status == "pending"
    assert store.get_task(current.id).status == "assigned"
    store.close()


def test_router_admission_limit_counts_only_active_tasks(tmp_path):
    store = RouterTaskStore(tmp_path / "router.db", max_active_tasks=1)
    first = store.create_task(_request("first"), "definition", "first_hash")
    with pytest.raises(RouterQueueFull):
        store.create_task(_request("second"), "definition", "second_hash")

    store.assign(first.id, "node-a", "generation-1")
    assigned = store.get_task(first.id)
    store.finish(assigned, _terminal_response(first.id, "first_hash"))
    second = store.create_task(_request("second"), "definition", "second_hash")
    assert second.status == "pending"
    store.close()


def test_stale_server_response_cannot_finish_reassigned_task(tmp_path):
    store = RouterTaskStore(tmp_path / "router.db")
    task = store.create_task(_request(), "definition", "solution_hash")
    store.assign(task.id, "node-a", "generation-1")
    stale = store.get_task(task.id)
    store.requeue(stale, "node lost")
    store.assign(task.id, "node-b", "generation-2")

    assert store.finish(stale, _terminal_response(task.id)) is False
    current = store.get_task(task.id)
    assert current.status == "assigned"
    assert current.backend_id == "node-b"
    store.close()


def test_client_task_id_is_idempotent(tmp_path):
    store = RouterTaskStore(tmp_path / "router.db")
    first = store.create_task(_request(), "definition", "solution_hash", task_id="client-task-id")
    repeated = store.create_task(
        _request(), "definition", "solution_hash", task_id="client-task-id"
    )
    assert repeated == first

    with pytest.raises(RouterTaskConflict):
        store.create_task(
            _request("different"), "definition", "different_hash", task_id="client-task-id"
        )
    store.close()
