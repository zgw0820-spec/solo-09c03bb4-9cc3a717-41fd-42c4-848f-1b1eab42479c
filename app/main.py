"""FastAPI service exposing path-scoped restores from read-only tar.gz backups."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .recycle import (
    BLOCK_SIZE,
    RecycleBin,
    RecycleError,
    RecycleNotFound,
)
from .restore import (
    InvalidArchive,
    InvalidSelection,
    LinkError,
    MalformedArchive,
    NotADirectory,
    PathNotFound,
    RestoreError,
    restore_directory,
)

logger = logging.getLogger("restore")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

ARCHIVE_DIR = os.environ.get("RESTORE_ARCHIVE_DIR", "/backups")
OUTPUT_DIR = os.environ.get("RESTORE_OUTPUT_DIR", "/restored")
MAX_CONCURRENT = max(1, int(os.environ.get("RESTORE_MAX_CONCURRENT", "2")))
STALE_STAGING_MAX_AGE = 24 * 60 * 60  # seconds

# Recycle bin: capacity is metered in 512-byte blocks actually occupied by
# regular files (hard links counted once per tree).  Retention starts at
# move-in time and is persisted, so restarts never reset the clock.
# The bin lives under <output>/.recycle by default (override with RECYCLE_DIR);
# it must share a filesystem with the output directory so moves are atomic.
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"invalid {name}={raw!r}: expected an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"invalid {name}={raw!r}: expected a number") from exc


RECYCLE_CAPACITY_BLOCKS = max(0, _env_int("RECYCLE_CAPACITY_BLOCKS", 512 * 1024))
RECYCLE_RETENTION_SECONDS = max(1.0, _env_float("RECYCLE_RETENTION_SECONDS", 7 * 24 * 3600.0))

_restore_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


def build_recycle_bin(
    output_dir: str = OUTPUT_DIR,
    capacity_blocks: int = RECYCLE_CAPACITY_BLOCKS,
    retention_seconds: float = RECYCLE_RETENTION_SECONDS,
) -> RecycleBin:
    """Construct a recycle bin under ``<output_dir>/.recycle``."""

    recycle_dir = os.environ.get("RECYCLE_DIR")
    if recycle_dir is None:
        recycle_dir = os.path.join(output_dir, ".recycle")
    return RecycleBin(
        recycle_dir,
        capacity_blocks,
        retention_seconds,
        restore_root=output_dir,
    )


recycle_bin = build_recycle_bin()


class ArchiveInfo(BaseModel):
    name: str
    size_bytes: int
    modified_at: float


class RestoreRequest(BaseModel):
    archive: str = Field(..., description="Archive file name, as returned by GET /api/archives")
    path: str = Field(
        ...,
        description="Directory to restore, relative to the archive root (e.g. 'data/2024')",
        examples=["data/2024"],
    )


class RestoreResponse(BaseModel):
    requested_path: str
    restore_id: str
    restored_path: str
    directories: int
    files: int
    hardlinks: int
    bytes_written: int
    bytes_extracted: int
    archive_payload_bytes: int
    skipped_unsafe: int
    skipped_unsupported: int


class RecycleMoveRequest(BaseModel):
    restore_id: str = Field(
        ...,
        description="Restore id returned by POST /api/restores (the <restore-id> directory under the output root)",
        examples=["3f0c2b9e7a6d4f1a8b2c3d4e5f607182"],
    )


def _archive_path(name: str) -> str:
    """Resolve an archive name to a path inside the configured backup mount."""

    if not name or name != os.path.basename(name) or name in (".", ".."):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid archive name")
    path = os.path.join(ARCHIVE_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"archive {name!r} not found")
    return path


def _cleanup_stale_staging(now: float | None = None) -> None:
    """Remove staging directories left behind by a killed process.

    Normal failures clean up after themselves; this only catches hard kills
    where the process never got to run its ``finally`` block.
    """

    import time

    staging = os.path.join(OUTPUT_DIR, ".staging")
    if not os.path.isdir(staging):
        return
    now = time.time() if now is None else now
    for entry in os.scandir(staging):
        try:
            if now - entry.stat().st_mtime > STALE_STAGING_MAX_AGE:
                shutil.rmtree(entry.path, ignore_errors=True)
                logger.warning("removed stale staging directory %s", entry.path)
        except OSError:
            continue


@asynccontextmanager
async def lifespan(_: FastAPI):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _cleanup_stale_staging()
    recycle_bin.startup()
    logger.info(
        "archive dir=%s output dir=%s recycle dir=%s capacity=%d blocks (%d bytes) retention=%.0fs",
        ARCHIVE_DIR,
        OUTPUT_DIR,
        recycle_bin.base_dir,
        RECYCLE_CAPACITY_BLOCKS,
        RECYCLE_CAPACITY_BLOCKS * BLOCK_SIZE,
        RECYCLE_RETENTION_SECONDS,
    )
    yield


app = FastAPI(
    title="Archive Subdirectory Restore",
    version="1.1.0",
    description=(
        "Restore a single directory (including hard links that reach outside it) "
        "from a read-only tar.gz backup, without unpacking the whole archive. "
        "Published restore trees can be moved into a capacity-bounded recycle bin, "
        "withdrawn before their retention expires, and cleaned up afterwards."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/archives", response_model=list[ArchiveInfo], tags=["archives"])
def list_archives() -> list[ArchiveInfo]:
    try:
        entries = list(os.scandir(ARCHIVE_DIR))
    except FileNotFoundError:
        return []
    archives = []
    for entry in entries:
        if not entry.is_file():
            continue
        if not (entry.name.endswith(".tar.gz") or entry.name.endswith(".tgz")):
            continue
        stat = entry.stat()
        archives.append(
            ArchiveInfo(name=entry.name, size_bytes=stat.st_size, modified_at=stat.st_mtime)
        )
    return sorted(archives, key=lambda a: a.name)


@app.post(
    "/api/restores",
    response_model=RestoreResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["restore"],
    summary="Restore one subdirectory",
)
def create_restore(request: RestoreRequest) -> RestoreResponse:
    archive_path = _archive_path(request.archive)
    if not _restore_slots.acquire(blocking=False):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many concurrent restores, retry shortly",
        )
    try:
        result = restore_directory(archive_path, request.path, OUTPUT_DIR)
    except InvalidSelection as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except NotADirectory as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except PathNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except LinkError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except MalformedArchive as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except InvalidArchive as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except RestoreError as exc:
        logger.exception("restore failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
    except OSError as exc:
        logger.exception("restore failed with OS error")
        raise HTTPException(status.HTTP_507_INSUFFICIENT_STORAGE, str(exc)) from exc
    finally:
        _restore_slots.release()

    logger.info(
        "restored %s from %s -> %s (%d files, %d links, %d bytes)",
        request.path,
        request.archive,
        result.restored_path,
        result.files,
        result.hardlinks,
        result.bytes_written,
    )
    return RestoreResponse(**asdict(result))


# ---------------------------------------------------------------------------
# Recycle bin
# ---------------------------------------------------------------------------


def _raise_recycle_error(exc: RecycleError) -> None:
    raise HTTPException(exc.status_code, str(exc)) from exc


@app.get(
    "/api/recycle",
    tags=["recycle"],
    summary="Recycle bin capacity and all held items",
)
def recycle_status() -> dict:
    return recycle_bin.status()


@app.get(
    "/api/recycle/items",
    tags=["recycle"],
    summary="List recycle-bin items",
)
def recycle_list() -> list[dict]:
    return recycle_bin.list_items()


@app.get(
    "/api/recycle/items/{item_id}",
    tags=["recycle"],
    summary="Inspect one recycle-bin item",
)
def recycle_get(item_id: str) -> dict:
    try:
        return recycle_bin.get_item(item_id)
    except RecycleNotFound as exc:
        _raise_recycle_error(exc)


@app.post(
    "/api/recycle/items",
    status_code=status.HTTP_201_CREATED,
    tags=["recycle"],
    summary="Move a published restore tree into the recycle bin",
    description=(
        "Moves `<output>/<restore-id>` into the bin in one atomic rename: the "
        "original path disappears immediately. Charged by disk blocks of regular "
        "files (hard links counted once). If the bin has no headroom the move is "
        "refused with 507 and the original path is kept."
    ),
)
def recycle_move_in(request: RecycleMoveRequest) -> dict:
    try:
        return recycle_bin.move_in(OUTPUT_DIR, request.restore_id)
    except RecycleError as exc:
        _raise_recycle_error(exc)


@app.post(
    "/api/recycle/items/{item_id}/withdraw",
    tags=["recycle"],
    summary="Withdraw an item before it expires",
    description=(
        "Moves the whole tree back to its original path. Only allowed before "
        "expiry; expired items can only be cleaned up."
    ),
)
def recycle_withdraw(item_id: str) -> dict:
    try:
        return recycle_bin.withdraw(item_id)
    except RecycleError as exc:
        _raise_recycle_error(exc)


@app.post(
    "/api/recycle/cleanup",
    tags=["recycle"],
    summary="Clean up expired items (and retry interrupted deletions)",
    description=(
        "Deletes every expired item, plus any item left in the `deleting` state "
        "by an interrupted cleanup. Deletion is incremental: an interrupted pass "
        "keeps the item and its quota reserved, and calling again finishes it. "
        "The quota is released only once the tree is fully gone."
    ),
)
def recycle_cleanup() -> dict:
    try:
        return recycle_bin.cleanup_expired()
    except RecycleError as exc:
        _raise_recycle_error(exc)


@app.post(
    "/api/recycle/items/{item_id}/cleanup",
    tags=["recycle"],
    summary="Clean up one expired item, or retry its interrupted deletion",
)
def recycle_cleanup_one(item_id: str) -> dict:
    try:
        return recycle_bin.cleanup_expired(item_id=item_id)
    except RecycleError as exc:
        _raise_recycle_error(exc)
