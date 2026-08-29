# Benchmark Server Router Design

## Status

This document describes the first production-oriented router for the FlashInfer-Bench HTTP
evaluation service. The router owns the public endpoint, keeps a durable task queue, and assigns
work to benchmark servers. Each benchmark server owns one compute node and its local GPUs.

## Goals

- Present the existing submit-and-poll API through one stable address.
- Route work only to compatible servers with healthy GPU workers.
- Keep queued work durable across router restarts.
- Detect server process restarts and replay unfinished work on another server.
- Stop sending work to a server as soon as it begins draining for a spot interruption or planned
  shutdown.
- Bound work admitted to each server so node loss does not strand a large remote queue.
- Scale health and task polling by batching requests per server instead of polling every task
  separately.
- Preserve direct, single-server deployments and API clients.

## Terminology

- A **compute node** is one host or cloud virtual machine that provides local compute devices.
- A **graphics processing unit (GPU)** is the accelerator used to evaluate kernels.
- A **benchmark server** is the existing Hypertext Transfer Protocol (HTTP) service on one compute
  node.
- A **router** is the new public HTTP service that assigns logical tasks to benchmark servers.
- A **spot instance** is a discounted cloud virtual machine that its provider may terminate with
  short notice.
- **SQLite** is the embedded transactional database used for router state. Its **write-ahead log**
  records changes before they reach the main database file, improving crash recovery.
- An **idempotent** submission produces one logical task when the same identifier and request are
  submitted repeatedly.
- **Transport Layer Security (TLS)** protects and authenticates network connections.

## Non-goals

- Exactly-once GPU execution. A node may finish work immediately before becoming unreachable, so
  its replacement can execute the same logical task again. Results remain idempotent because a
  router-generated task identifier is reused for every attempt.
- Coordinating several active router processes. The included SQLite store supports one router
  process with many benchmark servers. A later deployment can replace the store interface with a
  shared transactional database when active-active routers are required.
- Provisioning or terminating cloud instances. An autoscaler owns compute lifecycle; the router
  owns health, admission, draining, and task recovery.

## Architecture

```text
clients
   |
   v
router process ---- SQLite write-ahead log
   |                  - request payloads
   |                  - task assignment and attempt count
   |                  - terminal responses
   |
   +---- benchmark server node-a ---- local GPU workers
   +---- benchmark server node-b ---- local GPU workers
   +---- benchmark server node-c ---- local GPU workers
```

The router normalizes a submitted solution, generates the public task identifier, and commits the
request before attempting remote dispatch. A dispatcher assigns queued tasks while each server has
free capacity. A reconciler sends one batch status request per server and persists terminal
responses.

Metadata requests such as `GET /definitions` are forwarded to a healthy compatible server. All
servers in one router pool must expose the same `dataset_id`. The identifier hashes the complete
definition and workload data, preventing accidental routing across different trace sets.

## Server Health Contract

The benchmark server health response includes:

- `instance_id`: a random identifier created for each server process. A changed identifier at the
  same address proves that the process restarted.
- `dataset_id`: a deterministic identifier for the loaded trace set.
- `status`: `ok`, `degraded`, `draining`, or `unavailable`.
- `accepting`: whether new tasks are accepted.
- healthy worker count, total worker count, queue depth, and active task count.

The router probes servers concurrently. A single failed probe does not immediately move work:
failure thresholds avoid replay during a short network pause. A changed `instance_id` immediately
requeues unfinished assignments because an in-memory server task store cannot survive that change.

Servers retry failed GPU-worker startup and restart with bounded exponential backoff. Consequently,
a temporarily unavailable GPU does not require the HTTP server process to be replaced.

## Scheduling And Backpressure

The router keeps the authoritative pending queue. By default it assigns at most one task per
healthy remote GPU worker. This keeps benchmark server queues short and limits the recovery cost of
a terminated spot node. The limit can be raised when workloads are small enough to benefit from
remote buffering.

Among servers with free slots, the router chooses the lowest utilization ratio and rotates ties.
Selection scans the configured servers once. Health and reconciliation network operations are
concurrent and batch-oriented, which is the dominant scaling property for larger pools.

A configurable maximum number of non-terminal tasks provides admission backpressure. The router
returns HTTP 503 instead of accepting an unbounded amount of work when that limit is reached.

## Task Lifecycle And Recovery

1. The router writes a normalized request as `pending` in SQLite.
2. The dispatcher selects a healthy server with a free slot and marks the task `assigned` before
   sending it. The router-generated or client-supplied public task identifier is included in the
   server request.
3. A server treats repeated submissions of the same task identifier and payload as one task.
4. The reconciler polls assigned task identifiers in batches. `completed` and `failed` responses
   are copied into SQLite and remain available even if the server later disappears.
5. The router moves a non-terminal assignment back to `pending` when:
   - the server crosses its consecutive health-failure threshold;
   - the server reports a different `instance_id`; or
   - the server returns HTTP 404 for the assigned task.
6. The next dispatch can send the same identifier to another server. An attempt limit prevents a
   permanently failing infrastructure task from cycling forever.

SQLite uses write-ahead logging and full transaction synchronization. After a router restart,
pending tasks dispatch again and assigned tasks reconcile against the recorded server generation.

## Spot Instances And Planned Maintenance

On a spot interruption notice, a node-side hook should call `POST /drain` on its benchmark server.
The call immediately rejects new tasks but lets current work finish. The router observes
`accepting=false` on the next health probe and removes the server from selection. Since remote
admission is bounded, at most a small number of tasks need replay if the termination deadline wins
the race.

For planned maintenance, operators use the same drain flow and wait until `active_tasks` and
`queue_size` are zero before stopping the node. Restarting the process creates a new `instance_id`;
the router recognizes it automatically and returns the server to the pool after a successful health
probe.

Unexpected spot loss follows the consecutive-failure path. The router requeues unfinished tasks
after the configured threshold, while completed results already copied into SQLite are unaffected.

## API And Operations

The router exposes the existing endpoints:

- `POST /evaluate`
- `GET /tasks/{task_id}`
- `POST /tasks/batch`
- definition and workload read endpoints

It also exposes:

- `GET /live`: process liveness, independent of server availability.
- `GET /health`: router readiness and aggregate queue state.
- `GET /backends`: per-server health, generation, capacity, and failure details.

The direct server adds `POST /drain`. Management endpoints should be restricted by the deployment's
network policy. They are not intended for an untrusted public network.

Example:

```bash
flashinfer-bench router \
  --backend node-a=http://10.0.0.11:8000 \
  --backend node-b=http://10.0.0.12:8000 \
  --state-db /var/lib/flashinfer-bench/router.db \
  --host 0.0.0.0 \
  --port 9000
```

## Failure Semantics

- No healthy servers: already accepted tasks stay pending; metadata proxy requests return HTTP 503.
- Router queue full: new evaluations return HTTP 503 without creating a task.
- Server returns a task-level failure: the router persists it and does not replay it.
- Router loses a dispatch response: the assignment remains associated with that server and is
  reconciled by task identifier, avoiding an immediate duplicate.
- Router crashes during dispatch: the transaction leaves the task either pending or assigned. Both
  states are recoverable on startup.
- SQLite unavailable or corrupt: the router fails closed rather than accepting work it cannot
  recover.

## Security

Solution source code is already untrusted input to the benchmark worker. The router must preserve
the same request-size controls and network isolation as a direct server deployment. SQLite files
can contain submitted source and results and therefore require restrictive filesystem permissions.
TLS authentication and authorization should normally be provided by the deployment ingress. The
server and router management endpoints should only be reachable from the control network.

## Verification Plan

- Unit-test server task idempotency, conflicts, health states, and drain admission.
- Test fair routing and capacity bounds with fake servers.
- Test consecutive health failures, process-generation changes, and HTTP 404 task recovery.
- Test durable pending and terminal task state by reopening SQLite.
- Test router API submit, polling, metadata proxying, backpressure, and restart recovery without a
  GPU.
- Run the existing server API suite to protect direct-server compatibility; GPU-specific cases may
  be skipped where CUDA is unavailable.
