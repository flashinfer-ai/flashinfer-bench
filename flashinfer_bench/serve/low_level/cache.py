"""Server-side persistent blob cache backed by shared-memory files."""

from __future__ import annotations

import mmap
import os
import threading
from collections import OrderedDict
from pathlib import Path

from pydantic import BaseModel


class ShmUtils:
    """Thin helpers for shared-memory blob file operations."""

    @staticmethod
    def write(cache_root: str | Path, blob_hash: str, blob: bytes) -> str:
        """Write one blob into the cache root and return its file path."""
        cache_root_path = Path(cache_root)
        cache_root_path.mkdir(parents=True, exist_ok=True)
        blob_path = cache_root_path / blob_hash
        if not blob_path.exists():
            blob_path.write_bytes(blob)
        return str(blob_path)

    @staticmethod
    def read(file_path: str | Path) -> memoryview:
        """Map one cached blob file and return a memoryview."""
        with open(file_path, "rb") as blob_file:
            blob_map = mmap.mmap(blob_file.fileno(), length=0, access=mmap.ACCESS_READ)
        return memoryview(blob_map)

    @staticmethod
    def list(cache_root: str | Path) -> list[str]:
        """List all cached blob file paths under one cache root."""
        cache_root_path = Path(cache_root)
        if not cache_root_path.exists():
            return []
        return [str(path) for path in cache_root_path.iterdir() if path.is_file()]

    @staticmethod
    def delete(file_path: str | Path) -> None:
        """Delete one cached blob file if it exists."""
        Path(file_path).unlink(missing_ok=True)


class CacheEntry(BaseModel):
    """One cached blob object stored as a file on tmpfs."""

    blob_hash: str
    """Content hash identifying the blob."""
    path: str
    """Absolute file-system path of the cached blob object."""
    size: int
    """Logical blob size in bytes."""
    ref_count: int = 0
    """Number of in-flight requests currently using this blob."""


class CacheManager:
    """Manage persistent cached blob files for the low-level server.

    The public API includes:

    - ``get``, ``set``, and ``set_staged`` all return an already-acquired
      ``CacheEntry``.
    - Every ``CacheEntry`` returned to the caller must eventually be passed
      to ``release`` exactly once.
    - ``clear`` wipes the cache directory, but only when no entry is active.
    """

    def __init__(
        self,
        cache_root: str | Path = "/dev/shm/flashinfer-bench-cache",
        cache_capacity_bytes: int = 64 * 1024 * 1024 * 1024,
    ) -> None:
        """Initialize the cache manager.

        Parameters
        ----------
        cache_root
            Root directory storing blob files.
        cache_capacity_bytes
            Best-effort capacity limit for in-memory indexed cache objects.

        Returns
        -------
        None
            This constructor prepares directories and internal state in place.
        """
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_capacity_bytes = cache_capacity_bytes
        # OrderedDict doubles as the LRU queue: oldest first, newest last.
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._indexed_bytes = 0
        self._lock = threading.Lock()

    def get(self, blob_hash: str) -> CacheEntry | None:
        """Return one cached entry and acquire a reference on it.

        Parameters
        ----------
        blob_hash
            Content hash identifying the requested blob.

        Returns
        -------
        CacheEntry | None
            Cached blob entry already acquired by the caller, or ``None``
            when the blob does not exist.
        """
        with self._lock:
            entry = self._entries.get(blob_hash)
            if entry is None:
                entry = self._find_disk_entry(blob_hash)
            if entry is None:
                return None
            entry.ref_count += 1
            self._entries.move_to_end(entry.blob_hash)
            return entry

    def set(self, blob_hash: str, blob: bytes) -> CacheEntry:
        """Commit one blob into the cache and acquire a reference.

        Parameters
        ----------
        blob_hash
            Content hash identifying the blob.
        blob
            Raw blob payload bytes.

        Returns
        -------
        CacheEntry
            Cached blob entry already acquired by the caller.
        """
        blob_size = len(blob)
        with self._lock:
            existing = self._entries.get(blob_hash)
            if existing is None:
                existing = self._find_disk_entry(blob_hash)
            if existing is not None:
                existing.ref_count += 1
                self._entries.move_to_end(existing.blob_hash)
                return existing

            blob_path = ShmUtils.write(self.cache_root, blob_hash, blob)
            entry = CacheEntry(blob_hash=blob_hash, path=blob_path, size=blob_size, ref_count=0)
            self._entries[blob_hash] = entry
            self._indexed_bytes += blob_size
            entry.ref_count += 1
            self._entries.move_to_end(entry.blob_hash)
            self._evict_entries()
            return entry

    def set_staged(self, blob_hash: str, staged_path: str | Path, size: int) -> CacheEntry:
        """Commit one staged blob file into the cache and acquire a reference.

        Parameters
        ----------
        blob_hash
            Content hash identifying the blob.
        staged_path
            Temporary file path holding the already-written blob bytes.
        size
            Logical blob size in bytes.

        Returns
        -------
        CacheEntry
            Cached blob entry already acquired by the caller.
        """
        staged_path = Path(staged_path)
        with self._lock:
            existing = self._entries.get(blob_hash)
            if existing is None:
                existing = self._find_disk_entry(blob_hash)
            if existing is not None:
                staged_path.unlink(missing_ok=True)
                existing.ref_count += 1
                self._entries.move_to_end(existing.blob_hash)
                return existing

            blob_path = self.cache_root / blob_hash
            os.replace(staged_path, blob_path)
            entry = CacheEntry(blob_hash=blob_hash, path=str(blob_path), size=size, ref_count=0)
            self._entries[blob_hash] = entry
            self._indexed_bytes += size
            entry.ref_count += 1
            self._entries.move_to_end(entry.blob_hash)
            self._evict_entries()
            return entry

    def release(self, entry: CacheEntry) -> None:
        """Release one previously acquired cache entry.

        Parameters
        ----------
        entry
            Cache entry returned by a prior ``get`` or ``set`` call.

        Returns
        -------
        None
            This method decrements the entry's active reference count in place.
        """
        with self._lock:
            tracked = self._entries.get(entry.blob_hash)
            if tracked is None or tracked is not entry:
                raise RuntimeError(f"Cannot release unknown blob: {entry.blob_hash}")
            if tracked.ref_count <= 0:
                raise RuntimeError(f"Blob {entry.blob_hash} has no active references")
            tracked.ref_count -= 1

    def clear(self) -> None:
        """Remove every cached object when no blob is actively referenced.

        Returns
        -------
        None
            This method clears cache files and in-memory state in place.
        """
        with self._lock:
            if any(entry.ref_count > 0 for entry in self._entries.values()):
                raise RuntimeError("Cannot clear cache while requests are in flight")

            for file_path in ShmUtils.list(self.cache_root):
                ShmUtils.delete(file_path)

            self._entries.clear()
            self._indexed_bytes = 0

    def _find_disk_entry(self, blob_hash: str) -> CacheEntry | None:
        """Find one on-disk cache entry and add it to the in-memory index.

        The caller must already hold ``self._lock``.

        Parameters
        ----------
        blob_hash
            Content hash identifying the requested blob.

        Returns
        -------
        CacheEntry | None
            Cached blob entry already acquired by the caller, or ``None``
            when the blob does not exist.
        """
        blob_path = self.cache_root / blob_hash
        if not blob_path.exists():
            return None

        entry = CacheEntry(
            blob_hash=blob_hash, path=str(blob_path), size=blob_path.stat().st_size, ref_count=0
        )
        self._entries[blob_hash] = entry
        self._indexed_bytes += entry.size
        return entry

    def _evict_entries(self) -> None:
        """Evict inactive indexed entries until the cache fits the capacity.

        Returns
        -------
        None
            This method mutates cache files and in-memory state in place. The
            caller must already hold ``self._lock``. Entries with active
            references are skipped.
        """
        if self._indexed_bytes <= self.cache_capacity_bytes:
            return

        for blob_hash, entry in list(self._entries.items()):
            if self._indexed_bytes <= self.cache_capacity_bytes:
                return
            if entry.ref_count > 0:
                continue

            ShmUtils.delete(entry.path)
            self._indexed_bytes -= entry.size
            self._entries.pop(blob_hash, None)


__all__ = ["CacheEntry", "CacheManager", "ShmUtils"]
