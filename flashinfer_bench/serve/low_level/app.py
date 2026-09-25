"""FastAPI app for the low-level GPU server."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from flashinfer_bench.serve.low_level.errors import InvalidProgramError
from flashinfer_bench.serve.low_level.multipart import (
    build_multipart_response,
    parse_multipart_request,
)
from flashinfer_bench.serve.low_level.schema import (
    ErrorResult,
    ExecuteResult,
    HealthResponse,
    ServerExecuteResponseKind,
)
from flashinfer_bench.serve.low_level.server import LowLevelServer
from flashinfer_bench.utils import list_cuda_devices
from flashinfer_bench.version import __version__

router = APIRouter()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Own the runtime lifecycle for one FastAPI app instance.

    Parameters
    ----------
    app
        FastAPI application bound to this lifespan handler.

    Yields
    ------
    None
        Control returns to FastAPI while the app is serving requests.
    """
    server = app.state.server
    assert isinstance(server, LowLevelServer), "Low-level server is not initialized"
    await server.start()
    try:
        yield
    finally:
        await server.shutdown()
        app.state.server = None


@router.post(
    "/execute",
    responses={
        200: {"description": "Success (multipart)", "model": ExecuteResult},
        400: {"description": "Invalid program or execution failed", "model": ErrorResult},
        408: {"description": "Timeout", "model": ErrorResult},
    },
)
async def execute(request: Request):
    """Execute a program on GPU.

    Parameters
    ----------
    request
        Incoming HTTP request carrying one multipart execution payload.

    Returns
    -------
    Response
        Multipart success response or JSON error response.
    """
    server = request.app.state.server
    assert isinstance(server, LowLevelServer), "Low-level server is not initialized"
    trace_id = uuid.uuid4().hex
    request_start_time = time.perf_counter()
    print(f"TRACE request_id={trace_id} span=app.request phase=begin", flush=True)
    try:
        parse_start_time = time.perf_counter()
        print(f"TRACE request_id={trace_id} span=app.parse_request phase=begin", flush=True)
        server_execute_request = await parse_multipart_request(request, server.cache_manager)
        print(
            f"TRACE request_id={trace_id} span=app.parse_request phase=end "
            f"duration_ms={(time.perf_counter() - parse_start_time) * 1000.0:.3f}",
            flush=True,
        )
    except InvalidProgramError as error:
        result = ErrorResult(error="invalid_program", message=error.message)
        return JSONResponse(status_code=400, content=result.model_dump(exclude_none=True))

    server_execute_start_time = time.perf_counter()
    print(f"TRACE request_id={trace_id} span=app.server_execute phase=begin", flush=True)
    server_response = await server.execute_request(server_execute_request, trace_id=trace_id)
    print(
        f"TRACE request_id={trace_id} span=app.server_execute phase=end "
        f"duration_ms={(time.perf_counter() - server_execute_start_time) * 1000.0:.3f}",
        flush=True,
    )
    if server_response.kind is ServerExecuteResponseKind.OK:
        success_payload = server_response.payload
        assert isinstance(
            success_payload, ExecuteResult
        ), "Successful server response must carry ExecuteResult"
        build_response_start_time = time.perf_counter()
        print(f"TRACE request_id={trace_id} span=app.build_response phase=begin", flush=True)
        body, content_type = build_multipart_response(
            result=success_payload, binary_parts=server_response.binary_parts
        )
        print(
            f"TRACE request_id={trace_id} span=app.build_response phase=end "
            f"duration_ms={(time.perf_counter() - build_response_start_time) * 1000.0:.3f}",
            flush=True,
        )
        print(
            f"TRACE request_id={trace_id} span=app.request phase=end "
            f"duration_ms={(time.perf_counter() - request_start_time) * 1000.0:.3f}",
            flush=True,
        )
        return Response(content=body, media_type=content_type, status_code=200)
    elif server_response.kind is ServerExecuteResponseKind.TIMEOUT:
        print(
            f"TRACE request_id={trace_id} span=app.request phase=end "
            f"duration_ms={(time.perf_counter() - request_start_time) * 1000.0:.3f}",
            flush=True,
        )
        return JSONResponse(
            status_code=408, content=server_response.payload.model_dump(exclude_none=True)
        )
    print(
        f"TRACE request_id={trace_id} span=app.request phase=end "
        f"duration_ms={(time.perf_counter() - request_start_time) * 1000.0:.3f}",
        flush=True,
    )
    return JSONResponse(
        status_code=400, content=server_response.payload.model_dump(exclude_none=True)
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Server health check with per-worker status.

    Returns
    -------
    HealthResponse
        Structured health summary for all managed workers.
    """
    server = request.app.state.server
    assert isinstance(server, LowLevelServer), "Low-level server is not initialized"
    return server.health_response()


def create_app(devices: list[str], default_timeout_seconds: int = 30) -> FastAPI:
    """Create the low-level app with a concrete server instance.

    Parameters
    ----------
    devices
        CUDA device strings assigned to worker processes.
    default_timeout_seconds
        Default request timeout in seconds.

    Returns
    -------
    FastAPI
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="FlashInfer-Bench Low-Level Server", version=__version__, lifespan=_lifespan
    )
    app.state.server = LowLevelServer(
        devices=devices, default_timeout_seconds=default_timeout_seconds
    )
    app.include_router(router)
    return app


def main():
    """Run the low-level server as a standalone uvicorn process.

    Returns
    -------
    None
        This function configures logging, creates the app, and starts Uvicorn.
    """
    import argparse
    import logging

    import uvicorn

    parser = argparse.ArgumentParser(description="Low-level GPU judge server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--devices", type=str, default=None, help="Comma-separated, e.g. cuda:0,cuda:1"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Default execution timeout in seconds"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    devices = args.devices.split(",") if args.devices else list_cuda_devices()
    if not devices:
        raise RuntimeError("No CUDA devices available")

    application = create_app(devices=devices, default_timeout_seconds=args.timeout)
    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
