"""Helpers for request and response multipart payloads."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import uuid
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Tuple

from fastapi import Request
from pydantic import ValidationError

from flashinfer_bench.serve.low_level.cache import CacheEntry, CacheManager
from flashinfer_bench.serve.low_level.errors import InvalidProgramError
from flashinfer_bench.serve.low_level.schema import (
    ExecuteRequest,
    ExecuteResult,
    ServerExecuteRequest,
)

_BOUNDARY_RE = re.compile(r'boundary="?([^";]+)"?')
_NAME_RE = re.compile(r'name="([^"]+)"')


def _encode_part(boundary_bytes: bytes, name: str, content_type: str, payload: bytes) -> bytes:
    """Encode one multipart/form-data part into raw bytes.

    Parameters
    ----------
    boundary_bytes
        Multipart boundary token without the leading ``--`` marker.
    name
        Form-data part name written into ``Content-Disposition``.
    content_type
        MIME type written into the part headers.
    payload
        Raw part payload bytes.

    Returns
    -------
    bytes
        Fully encoded multipart segment including leading boundary, headers,
        payload, and trailing CRLF.
    """
    return (
        b"--"
        + boundary_bytes
        + b"\r\n"
        + f'Content-Disposition: form-data; name="{name}"\r\n'.encode("utf-8")
        + f"Content-Type: {content_type}\r\n\r\n".encode("ascii")
        + payload
        + b"\r\n"
    )


def build_multipart_response(
    result: ExecuteResult, binary_parts: Dict[str, bytes]
) -> Tuple[bytes, str]:
    """Build the low-level server success response multipart payload.

    Parameters
    ----------
    result
        Structured JSON result written into the ``result`` part.
    binary_parts
        Additional binary parts keyed by multipart part name, usually
        ``return:<key>`` payloads produced by execution.

    Returns
    -------
    tuple[bytes, str]
        Encoded multipart response body plus the matching ``Content-Type`` value.
    """
    boundary = f"flashinfer-bench-{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    chunks: list[bytes] = []
    chunks.append(
        _encode_part(
            boundary_bytes,
            "result",
            "application/json",
            json.dumps(result.model_dump()).encode("utf-8"),
        )
    )
    for part_name, payload in binary_parts.items():
        chunks.append(_encode_part(boundary_bytes, part_name, "application/octet-stream", payload))
    chunks.append(b"--" + boundary_bytes + b"--\r\n")
    body = b"".join(chunks)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _get_boundary(content_type: str | None) -> bytes:
    """Extract the multipart boundary token from one Content-Type header.

    Parameters
    ----------
    content_type
        Raw ``Content-Type`` header value from the HTTP request or response.

    Returns
    -------
    bytes
        Boundary token encoded as ASCII bytes, without the leading ``--``.

    Raises
    ------
    InvalidProgramError
        Raised when the header is missing or does not describe a multipart body.
    """
    if content_type is None:
        raise InvalidProgramError("Missing Content-Type header")
    boundary_match = _BOUNDARY_RE.search(content_type)
    if boundary_match is None:
        raise InvalidProgramError("Expected multipart/form-data body")
    return boundary_match.group(1).encode("ascii")


class _MultipartState(Enum):
    EXPECT_FIRST_BOUNDARY = auto()
    READ_HEADERS = auto()
    READ_PART_BODY = auto()
    DONE = auto()


class _SyncMultipartParser:
    """Stateful synchronous parser core for low-level execute requests."""

    def __init__(self, boundary: bytes, cache_manager: CacheManager) -> None:
        self.cache_manager = cache_manager
        self.first_boundary = b"--" + boundary + b"\r\n"
        self.next_boundary = b"\r\n--" + boundary
        self.boundary_overlap = len(self.next_boundary) + 4
        self.state = _MultipartState.EXPECT_FIRST_BOUNDARY
        self.buffer = bytearray()
        self.execute_request: ExecuteRequest | None = None
        self.blob_entries: Dict[str, CacheEntry] = {}
        self.current_part_name: str | None = None
        self.current_program = bytearray()
        self.current_blob_hash: str | None = None
        self.current_blob_hasher = None
        self.current_blob_entry: CacheEntry | None = None
        self.current_blob_file = None
        self.current_blob_staged_path: Path | None = None
        self.current_blob_size = 0

    def feed(self, chunk: bytes) -> None:
        """Feed one raw body chunk into the parser state machine."""
        if chunk:
            self.buffer.extend(chunk)
        while True:
            if self.state is _MultipartState.EXPECT_FIRST_BOUNDARY:
                if len(self.buffer) < len(self.first_boundary):
                    return
                if not self.buffer.startswith(self.first_boundary):
                    raise InvalidProgramError("Invalid multipart opening boundary")
                del self.buffer[: len(self.first_boundary)]
                self.state = _MultipartState.READ_HEADERS
                continue

            if self.state is _MultipartState.READ_HEADERS:
                header_end = self.buffer.find(b"\r\n\r\n")
                if header_end < 0:
                    return
                headers = _parse_headers(bytes(self.buffer[:header_end]))
                del self.buffer[: header_end + 4]
                self._start_part(_get_part_name(headers))
                self.state = _MultipartState.READ_PART_BODY
                continue

            if self.state is _MultipartState.READ_PART_BODY:
                boundary_index = self.buffer.find(self.next_boundary)
                if boundary_index < 0:
                    if len(self.buffer) <= self.boundary_overlap:
                        return
                    self._feed_current_part(bytes(self.buffer[: -self.boundary_overlap]))
                    del self.buffer[: -self.boundary_overlap]
                    return

                self._feed_current_part(bytes(self.buffer[:boundary_index]))
                del self.buffer[:boundary_index]
                if len(self.buffer) < len(self.next_boundary) + 2:
                    return

                boundary_suffix = bytes(
                    self.buffer[len(self.next_boundary) : len(self.next_boundary) + 2]
                )
                if boundary_suffix == b"\r\n":
                    self._finish_current_part()
                    del self.buffer[: len(self.next_boundary) + 2]
                    self.state = _MultipartState.READ_HEADERS
                    continue
                if boundary_suffix == b"--":
                    if len(self.buffer) < len(self.next_boundary) + 4:
                        return
                    if (
                        bytes(
                            self.buffer[len(self.next_boundary) + 2 : len(self.next_boundary) + 4]
                        )
                        != b"\r\n"
                    ):
                        raise InvalidProgramError("Invalid multipart closing boundary")
                    self._finish_current_part()
                    del self.buffer[: len(self.next_boundary) + 4]
                    self.state = _MultipartState.DONE
                    continue
                raise InvalidProgramError("Invalid multipart boundary suffix")

            if self.state is _MultipartState.DONE:
                if self.buffer:
                    raise InvalidProgramError("Unexpected trailing multipart data")
                return

    def finish(self) -> ServerExecuteRequest:
        """Finalize parsing after all request chunks have been consumed."""
        if self.state is not _MultipartState.DONE:
            raise InvalidProgramError("Incomplete multipart body")
        if self.execute_request is None:
            raise InvalidProgramError("Missing program part")
        return ServerExecuteRequest(
            execute_request=self.execute_request, blob_entries=self.blob_entries
        )

    def abort(self) -> None:
        """Release all request-local parser resources after a failure."""
        self._cleanup_current_part()
        for acquired_entry in self.blob_entries.values():
            try:
                self.cache_manager.release(acquired_entry)
            except Exception:
                pass

    def _start_part(self, part_name: str) -> None:
        self.current_part_name = part_name
        if part_name == "program":
            if self.execute_request is not None:
                raise InvalidProgramError("Duplicate program part")
            self.current_program = bytearray()
            return

        if not part_name.startswith("blob:"):
            raise InvalidProgramError(f"Unsupported multipart part: {part_name}")
        self.current_blob_hash = part_name.split(":", 1)[1]
        if self.current_blob_hash in self.blob_entries:
            raise InvalidProgramError(f"Duplicate blob: {self.current_blob_hash}")
        self.current_blob_hasher = hashlib.sha256()
        self.current_blob_size = 0
        self.current_blob_entry = self.cache_manager.get(self.current_blob_hash)
        if self.current_blob_entry is None:
            staged_file = tempfile.NamedTemporaryFile(
                mode="w+b", dir=self.cache_manager.cache_root, prefix=".upload-", delete=False
            )
            self.current_blob_file = staged_file
            self.current_blob_staged_path = Path(staged_file.name)

    def _reset_current_part(self) -> None:
        self.current_part_name = None
        self.current_program = bytearray()
        self.current_blob_hash = None
        self.current_blob_hasher = None
        self.current_blob_entry = None
        self.current_blob_file = None
        self.current_blob_staged_path = None
        self.current_blob_size = 0

    def _cleanup_current_part(self) -> None:
        if self.current_blob_file is not None:
            self.current_blob_file.close()
            self.current_blob_file = None
        if self.current_blob_staged_path is not None:
            self.current_blob_staged_path.unlink(missing_ok=True)
            self.current_blob_staged_path = None
        if self.current_blob_entry is not None:
            self.cache_manager.release(self.current_blob_entry)
            self.current_blob_entry = None
        self._reset_current_part()

    def _feed_current_part(self, data: bytes) -> None:
        if not data:
            return
        if self.current_part_name == "program":
            self.current_program.extend(data)
            return
        if self.current_blob_hasher is None or self.current_blob_hash is None:
            raise InvalidProgramError("Multipart parser lost blob state")
        self.current_blob_hasher.update(data)
        self.current_blob_size += len(data)
        if self.current_blob_file is not None:
            self.current_blob_file.write(data)

    def _finish_current_part(self) -> None:
        if self.current_part_name == "program":
            try:
                raw_program = json.loads(self.current_program.decode("utf-8"))
                self.execute_request = ExecuteRequest.model_validate(raw_program)
            except ValidationError as error:
                raise InvalidProgramError(str(error)) from error
            except Exception as error:
                raise InvalidProgramError(f"Invalid program JSON: {error}") from error
            self._reset_current_part()
            return

        if self.current_blob_hash is None or self.current_blob_hasher is None:
            raise InvalidProgramError("Multipart parser lost blob state")
        if self.current_blob_hasher.hexdigest() != self.current_blob_hash:
            raise InvalidProgramError(f"Blob hash mismatch: {self.current_blob_hash}")

        if self.current_blob_file is not None:
            self.current_blob_file.close()
            self.current_blob_file = None
            if self.current_blob_staged_path is None:
                raise InvalidProgramError("Multipart parser lost staged blob path")
            cache_entry = self.cache_manager.set_staged(
                self.current_blob_hash, self.current_blob_staged_path, self.current_blob_size
            )
            self.current_blob_staged_path = None
        else:
            if self.current_blob_entry is None:
                raise InvalidProgramError("Multipart parser lost cached blob entry")
            cache_entry = self.current_blob_entry
            self.current_blob_entry = None
        self.blob_entries[self.current_blob_hash] = cache_entry
        self._reset_current_part()


async def parse_multipart_request(
    request: Request, cache_manager: CacheManager
) -> ServerExecuteRequest:
    """Stream one execute request into a validated program plus cached blobs.

    This parser is intentionally narrow and only supports the request protocol
    used by the low-level server:

    - one ``program`` JSON part
    - zero or more ``blob:<sha256>`` binary parts

    The async wrapper owns HTTP request streaming, while the actual multipart
    state machine lives in a private synchronous parser core.

    Parameters
    ----------
    request
        Incoming FastAPI request carrying one multipart execute payload.
    cache_manager
        Server-owned blob cache used to resolve or commit uploaded blob parts.

    Returns
    -------
    ServerExecuteRequest
        Validated execute request plus already-acquired cache entries for every
        uploaded blob referenced by the request.

    Raises
    ------
    InvalidProgramError
        Raised when the multipart envelope, program JSON, blob naming, or blob
        hash validation fails.
    """
    parser = _SyncMultipartParser(
        boundary=_get_boundary(request.headers.get("content-type")), cache_manager=cache_manager
    )
    try:
        async for chunk in request.stream():
            parser.feed(chunk)
        return parser.finish()
    except BaseException:
        parser.abort()
        raise


def _parse_headers(header_block: bytes) -> dict[str, str]:
    """Parse one multipart header block into a lower-case mapping.

    Parameters
    ----------
    header_block
        Raw bytes between one part boundary and the blank line that terminates
        the part headers.

    Returns
    -------
    dict[str, str]
        Lower-case header name to header value mapping.

    Raises
    ------
    InvalidProgramError
        Raised when one header line is malformed.
    """
    headers: dict[str, str] = {}
    for header_line in header_block.decode("utf-8", errors="replace").split("\r\n"):
        if not header_line:
            continue
        if ":" not in header_line:
            raise InvalidProgramError(f"Invalid multipart header line: {header_line!r}")
        header_name, header_value = header_line.split(":", 1)
        headers[header_name.strip().lower()] = header_value.strip()
    return headers


def _get_part_name(headers: dict[str, str]) -> str:
    """Return the form-data part name from parsed multipart headers.

    Parameters
    ----------
    headers
        Parsed lower-case header mapping for one multipart part.

    Returns
    -------
    str
        Part name extracted from ``Content-Disposition``.

    Raises
    ------
    InvalidProgramError
        Raised when the part does not provide a valid form-data name.
    """
    disposition = headers.get("content-disposition")
    if disposition is None:
        raise InvalidProgramError("Missing Content-Disposition header")
    name_match = _NAME_RE.search(disposition)
    name = name_match.group(1) if name_match is not None else None
    if name is None:
        raise InvalidProgramError("Missing multipart part name")
    return name


def _parse_multipart_parts(content_type: str | None, body: bytes) -> list[tuple[str, bytes]]:
    """Parse one complete multipart body into named payload parts.

    This helper is intentionally non-streaming and is only used on the response
    side, where the body is already fully materialized in memory.

    Parameters
    ----------
    content_type
        Raw ``Content-Type`` header containing the multipart boundary.
    body
        Full multipart payload bytes.

    Returns
    -------
    list[tuple[str, bytes]]
        Ordered ``(part_name, payload_bytes)`` pairs.

    Raises
    ------
    InvalidProgramError
        Raised when the body is not a valid multipart payload for this narrow
        parser.
    """
    boundary = _get_boundary(content_type)
    delimiter = b"--" + boundary
    if not body.startswith(delimiter):
        raise InvalidProgramError("Invalid multipart opening boundary")

    parsed_parts: list[tuple[str, bytes]] = []
    for segment in body.split(delimiter):
        if not segment or segment in (b"--", b"--\r\n", b"\r\n"):
            continue
        if segment.startswith(b"--"):
            break
        if segment.startswith(b"\r\n"):
            segment = segment[2:]
        if segment.endswith(b"\r\n"):
            segment = segment[:-2]

        header_end = segment.find(b"\r\n\r\n")
        if header_end < 0:
            raise InvalidProgramError("Invalid multipart part: missing header separator")
        headers = _parse_headers(segment[:header_end])
        payload = segment[header_end + 4 :]
        parsed_parts.append((_get_part_name(headers), payload))
    return parsed_parts


def parse_multipart_response(
    content_type: str, body: bytes
) -> Tuple[ExecuteResult, Dict[str, bytes]]:
    """Parse one multipart response produced by ``build_multipart_response``.

    Parameters
    ----------
    content_type
        Raw response ``Content-Type`` header containing the multipart boundary.
    body
        Full multipart response body bytes.

    Returns
    -------
    tuple[ExecuteResult, dict[str, bytes]]
        Parsed JSON result object plus a mapping of binary part name to payload.

    Raises
    ------
    ValueError
        Raised when the response does not include the required ``result`` part.
    """
    result: ExecuteResult | None = None
    binary_parts: Dict[str, bytes] = {}

    for part_name, payload in _parse_multipart_parts(content_type, body):
        if part_name == "result":
            result = ExecuteResult.model_validate(json.loads(payload.decode("utf-8")))
        else:
            binary_parts[part_name] = payload

    if result is None:
        raise ValueError("Missing 'result' part in multipart response")
    return result, binary_parts


__all__ = ["build_multipart_response", "parse_multipart_request", "parse_multipart_response"]
