"""Tests for the benchmark server API.

All tests run on real GPU with a real Scheduler and PersistentSubprocessWorker.
"""

import shutil

import pytest
import torch

from tests.serve.conftest import (
    solution_correct,
    solution_illegal_memory,
    solution_runtime_crash,
    solution_slow,
    solution_syntax_error,
    solution_wrong_dtype,
    solution_wrong_shape,
    solution_wrong_value,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(torch.cuda.device_count() == 0, reason="No CUDA devices available"),
]

DEFINITION = "test_scale"


# ── Helper ──


async def submit_and_wait(client, solution, timeout: float = 30) -> dict:
    """Submit a solution and wait for result."""
    resp = await client.post("/evaluate", json={"solution": solution.model_dump(mode="json")})
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]

    resp = await client.get(f"/tasks/{task_id}", params={"timeout": timeout})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── 1. Task-Level Failures ──


async def test_no_workloads_found(client, test_trace_set):
    """1.1 Submit with non-existent workload_uuids -> task fails."""
    sol = solution_correct(DEFINITION)
    resp = await client.post(
        "/evaluate",
        json={"solution": sol.model_dump(mode="json"), "workload_uuids": ["nonexistent_uuid"]},
    )
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    resp = await client.get(f"/tasks/{task_id}", params={"timeout": 30})
    result = resp.json()

    assert result["status"] == "failed"
    assert "No workloads found" in result["error"]


# ── 2. Evaluation-Level Failures ──


async def test_compile_error(client):
    """2.1 COMPILE_ERROR: syntax error in solution."""
    sol = solution_syntax_error(DEFINITION)
    result = await submit_and_wait(client, sol)

    assert result["status"] == "completed"
    assert result["kind"] == "evaluate"
    assert result.get("logs") is None
    assert len(result["traces"]) > 0
    assert result["traces"][0]["evaluation"]["status"] == "COMPILE_ERROR"


async def test_runtime_error_nonfatal(client):
    """2.2 RUNTIME_ERROR (non-fatal): Python exception."""
    sol = solution_runtime_crash(DEFINITION)
    result = await submit_and_wait(client, sol)

    assert result["status"] == "completed"
    assert len(result["traces"]) > 0
    assert result["traces"][0]["evaluation"]["status"] == "RUNTIME_ERROR"
    assert "intentional crash" in result["traces"][0]["evaluation"]["log"]


async def test_runtime_error_cuda_corruption(client):
    """2.3 RUNTIME_ERROR (CUDA context corruption): illegal memory access.

    After this error, worker should auto-restart and subsequent tasks succeed.
    """
    # First: submit illegal memory access solution
    sol_bad = solution_illegal_memory(DEFINITION)
    result_bad = await submit_and_wait(client, sol_bad, timeout=60)

    # Should complete (not fail at task level) with RUNTIME_ERROR or COMPILE_ERROR
    assert result_bad["status"] == "completed"
    assert len(result_bad["traces"]) > 0
    status = result_bad["traces"][0]["evaluation"]["status"]
    assert status in ("RUNTIME_ERROR", "COMPILE_ERROR")

    # Second: submit a correct solution - should succeed (worker recovered)
    sol_good = solution_correct(DEFINITION)
    result_good = await submit_and_wait(client, sol_good, timeout=60)

    assert result_good["status"] == "completed"
    assert len(result_good["traces"]) > 0
    assert result_good["traces"][0]["evaluation"]["status"] == "PASSED"


async def test_timeout(client, benchmark_config):
    """2.4 TIMEOUT: solution takes too long."""
    sol = solution_slow(DEFINITION)
    result = await submit_and_wait(client, sol, timeout=60)

    assert result["status"] == "completed"
    assert len(result["traces"]) > 0
    assert result["traces"][0]["evaluation"]["status"] == "TIMEOUT"


async def test_incorrect_shape(client):
    """2.5 INCORRECT_SHAPE: wrong output shape."""
    sol = solution_wrong_shape(DEFINITION)
    result = await submit_and_wait(client, sol)

    assert result["status"] == "completed"
    assert len(result["traces"]) > 0
    assert result["traces"][0]["evaluation"]["status"] == "INCORRECT_SHAPE"


async def test_incorrect_numerical(client):
    """2.6 INCORRECT_NUMERICAL: wrong output values."""
    sol = solution_wrong_value(DEFINITION)
    result = await submit_and_wait(client, sol)

    assert result["status"] == "completed"
    assert len(result["traces"]) > 0
    assert result["traces"][0]["evaluation"]["status"] == "INCORRECT_NUMERICAL"


async def test_incorrect_dtype(client):
    """2.7 INCORRECT_DTYPE: wrong output dtype."""
    sol = solution_wrong_dtype(DEFINITION)
    result = await submit_and_wait(client, sol)

    assert result["status"] == "completed"
    assert len(result["traces"]) > 0
    assert result["traces"][0]["evaluation"]["status"] == "INCORRECT_DTYPE"


# ── 3. Success Cases ──


async def test_all_tasks_pass(client):
    """3.1 Multiple correct solutions all pass."""
    sol1 = solution_correct(DEFINITION)
    sol2 = solution_correct(DEFINITION)
    sol3 = solution_correct(DEFINITION)

    task_ids = []
    for sol in [sol1, sol2, sol3]:
        resp = await client.post("/evaluate", json={"solution": sol.model_dump(mode="json")})
        assert resp.status_code == 200
        task_ids.append(resp.json()["task_id"])

    resp = await client.post("/tasks/batch", json={"task_ids": task_ids, "timeout": 60})
    assert resp.status_code == 200
    results = resp.json()

    assert len(results) == 3
    for r in results:
        assert r["status"] == "completed"
        assert r["traces"][0]["evaluation"]["status"] == "PASSED"


async def test_mixed_success_failure(client):
    """3.2 Mixed: 1 correct, 1 wrong shape, 1 compile error."""
    sol_ok = solution_correct(DEFINITION)
    sol_shape = solution_wrong_shape(DEFINITION)
    sol_compile = solution_syntax_error(DEFINITION)

    task_ids = []
    for sol in [sol_ok, sol_shape, sol_compile]:
        resp = await client.post("/evaluate", json={"solution": sol.model_dump(mode="json")})
        assert resp.status_code == 200
        task_ids.append(resp.json()["task_id"])

    resp = await client.post("/tasks/batch", json={"task_ids": task_ids, "timeout": 60})
    assert resp.status_code == 200
    results = resp.json()

    statuses = [r["traces"][0]["evaluation"]["status"] for r in results]
    assert "PASSED" in statuses
    assert "INCORRECT_SHAPE" in statuses
    assert "COMPILE_ERROR" in statuses


# ── 4. HTTP-Level Errors ──


async def test_invalid_definition(client):
    """4.1 Submit with non-existent definition -> HTTP 400."""
    sol = solution_correct("nonexistent_definition")
    resp = await client.post("/evaluate", json={"solution": sol.model_dump(mode="json")})
    assert resp.status_code == 400
    assert "Definition not found" in resp.json()["detail"]


# ── 5. Batch API Behavior ──


async def test_batch_timeout_before_completion(client):
    """5.1 Batch with very short timeout returns immediately with pending tasks."""
    sol = solution_correct(DEFINITION)
    resp = await client.post("/evaluate", json={"solution": sol.model_dump(mode="json")})
    task_id = resp.json()["task_id"]

    # Query with 0 timeout - should return current status
    resp = await client.post("/tasks/batch", json={"task_ids": [task_id], "timeout": 0})
    assert resp.status_code == 200
    # Status could be pending, running, or completed depending on timing


async def test_batch_empty_list(client):
    """5.3 Empty task_ids -> returns empty list."""
    resp = await client.post("/tasks/batch", json={"task_ids": [], "timeout": 0})
    assert resp.status_code == 200
    assert resp.json() == []


async def test_batch_invalid_task_id(client):
    """5.4 Non-existent task_id -> HTTP 404."""
    resp = await client.post(
        "/tasks/batch", json={"task_ids": ["nonexistent_task_id"], "timeout": 0}
    )
    assert resp.status_code == 404
    assert "Task not found" in resp.json()["detail"]


async def test_batch_partial_invalid(client):
    """5.5 Mix of valid + invalid task_ids -> HTTP 404 (fail-fast)."""
    sol = solution_correct(DEFINITION)
    resp = await client.post("/evaluate", json={"solution": sol.model_dump(mode="json")})
    valid_id = resp.json()["task_id"]

    resp = await client.post(
        "/tasks/batch", json={"task_ids": [valid_id, "invalid_id"], "timeout": 0}
    )
    assert resp.status_code == 404


async def test_batch_duplicate_task_ids(client):
    """5.6 Duplicate task_ids -> no deadlock, returns duplicates."""
    sol = solution_correct(DEFINITION)
    resp = await client.post("/evaluate", json={"solution": sol.model_dump(mode="json")})
    task_id = resp.json()["task_id"]

    resp = await client.post("/tasks/batch", json={"task_ids": [task_id, task_id], "timeout": 60})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2
    assert results[0]["task_id"] == results[1]["task_id"]


# ── 6. Single Task API ──


async def test_get_task_with_timeout(client):
    """6.1 GET with timeout waits for completion."""
    sol = solution_correct(DEFINITION)
    resp = await client.post("/evaluate", json={"solution": sol.model_dump(mode="json")})
    task_id = resp.json()["task_id"]

    resp = await client.get(f"/tasks/{task_id}", params={"timeout": 60})
    assert resp.status_code == 200
    result = resp.json()
    assert result["status"] == "completed"


async def test_get_task_without_timeout(client):
    """6.2 GET without timeout returns immediately."""
    sol = solution_correct(DEFINITION)
    resp = await client.post("/evaluate", json={"solution": sol.model_dump(mode="json")})
    task_id = resp.json()["task_id"]

    # Default timeout=0, returns current status
    resp = await client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200


async def test_get_task_consistency_with_batch(client):
    """6.3 GET and POST /tasks/batch return identical results."""
    sol = solution_correct(DEFINITION)
    resp = await client.post("/evaluate", json={"solution": sol.model_dump(mode="json")})
    task_id = resp.json()["task_id"]

    # Wait for completion first
    resp_get = await client.get(f"/tasks/{task_id}", params={"timeout": 60})
    result_get = resp_get.json()

    resp_batch = await client.post("/tasks/batch", json={"task_ids": [task_id], "timeout": 0})
    result_batch = resp_batch.json()[0]

    assert result_get["task_id"] == result_batch["task_id"]
    assert result_get["status"] == result_batch["status"]
    assert result_get["definition"] == result_batch["definition"]


# ── 7. Worker Recovery ──


async def test_worker_recovery_after_cuda_error(client):
    """7.1 Worker recovers after CUDA context corruption."""
    # This is covered by test_runtime_error_cuda_corruption
    # Submit bad solution -> worker corrupts -> restarts -> submit good -> passes
    pass  # See test_runtime_error_cuda_corruption


async def test_health_endpoint(client):
    """7.2 Health endpoint shows worker status."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    health = resp.json()

    assert health["status"] == "ok"
    assert "workers" in health
    assert len(health["workers"]) > 0
    assert "queue_size" in health


# ── 8. Definition & Workload APIs ──


async def test_list_definitions(client):
    """8.1 GET /definitions returns list."""
    resp = await client.get("/definitions")
    assert resp.status_code == 200
    defs = resp.json()

    assert isinstance(defs, list)
    assert len(defs) > 0
    assert any(d["name"] == DEFINITION for d in defs)


async def test_get_definition_details(client):
    """8.2 GET /definitions/{name} returns full definition."""
    resp = await client.get(f"/definitions/{DEFINITION}")
    assert resp.status_code == 200
    defn = resp.json()

    assert defn["name"] == DEFINITION
    assert "axes" in defn
    assert "inputs" in defn
    assert "outputs" in defn
    assert "reference" in defn


async def test_list_workloads(client):
    """8.3 GET /definitions/{name}/workloads returns workloads."""
    resp = await client.get(f"/definitions/{DEFINITION}/workloads")
    assert resp.status_code == 200
    workloads = resp.json()

    assert isinstance(workloads, list)
    assert len(workloads) > 0
    assert "uuid" in workloads[0]
    assert "axes" in workloads[0]


async def test_get_workload_by_uuid(client):
    """8.4 GET /workloads/{uuid} returns workload."""
    # First get a workload UUID
    resp = await client.get(f"/definitions/{DEFINITION}/workloads")
    workloads = resp.json()
    uuid = workloads[0]["uuid"]

    resp = await client.get(f"/workloads/{uuid}")
    assert resp.status_code == 200
    wl = resp.json()
    assert wl["uuid"] == uuid


async def test_definition_not_found(client):
    """8.5a GET /definitions/nonexistent -> 404."""
    resp = await client.get("/definitions/nonexistent")
    assert resp.status_code == 404


async def test_workload_not_found(client):
    """8.5b GET /workloads/nonexistent -> 404."""
    resp = await client.get("/workloads/nonexistent")
    assert resp.status_code == 404


# ── 9. Profile / Sanitize Task APIs ──


@pytest.mark.skipif(shutil.which("ncu") is None, reason="ncu not available")
async def test_profile_task_lifecycle(client):
    """9.1 POST /profile returns task_id; /tasks/{id} returns kind=profile + logs."""
    sol = solution_correct(DEFINITION)
    resp = await client.post("/profile", json={"solution": sol.model_dump(mode="json")})
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]

    resp = await client.get(f"/tasks/{task_id}", params={"timeout": 120})
    assert resp.status_code == 200
    result = resp.json()

    assert result["status"] == "completed"
    assert result["kind"] == "profile"
    assert result.get("traces") is None or result["traces"] == []
    assert result["logs"] is not None and len(result["logs"]) > 0
    entry = result["logs"][0]
    assert entry["definition"] == DEFINITION
    assert "log" in entry


@pytest.mark.skipif(
    shutil.which("compute-sanitizer") is None, reason="compute-sanitizer not available"
)
async def test_sanitize_task_lifecycle(client):
    """9.2 POST /sanitize returns task_id; /tasks/{id} returns kind=sanitize + logs."""
    sol = solution_correct(DEFINITION)
    resp = await client.post("/sanitize", json={"solution": sol.model_dump(mode="json")})
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]

    resp = await client.get(f"/tasks/{task_id}", params={"timeout": 300})
    assert resp.status_code == 200
    result = resp.json()

    assert result["status"] == "completed"
    assert result["kind"] == "sanitize"
    assert result.get("traces") is None or result["traces"] == []
    assert result["logs"] is not None and len(result["logs"]) > 0
    entry = result["logs"][0]
    assert entry["definition"] == DEFINITION
    assert "log" in entry


async def test_profile_invalid_definition(client):
    """9.3 /profile with unknown definition -> 400."""
    sol = solution_correct("nonexistent_definition")
    resp = await client.post("/profile", json={"solution": sol.model_dump(mode="json")})
    assert resp.status_code == 400
    assert "Definition not found" in resp.json()["detail"]


async def test_sanitize_invalid_definition(client):
    """9.4 /sanitize with unknown definition -> 400."""
    sol = solution_correct("nonexistent_definition")
    resp = await client.post("/sanitize", json={"solution": sol.model_dump(mode="json")})
    assert resp.status_code == 400
    assert "Definition not found" in resp.json()["detail"]
