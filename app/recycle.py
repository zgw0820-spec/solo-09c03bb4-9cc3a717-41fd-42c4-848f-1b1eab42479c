"""Recycle bin for published restore trees.

A restore tree that has been published and is no longer changing can be moved
into the recycle bin by its restore id.  The move is a same-filesystem
``rename``, so the original path disappears in one atomic step and can be made
to reappear at the exact same path by withdrawing the item before it expires.

Billing
-------
Capacity is metered in *disk blocks actually occupied by regular files*.
Every other entry type (directories, symlinks, sockets, ...) costs nothing.
Hard links pointing at the same inode inside one tree are counted once: the
tree walk keeps a ``(st_dev, st_ino)`` set and each inode contributes the
blocks of a single ``lstat``.

Retries and interruption
------------------------
* State (including the original path and the move-in timestamp) is persisted
  in a JSON file and atomically rewritten on every change, so a restart never
  resets the retention clock and never loses an item.
* Cleanup deletes trees incrementally with the global lock released.  An
  interrupted cleanup leaves the item in the ``deleting`` state with its quota
  still reserved; retrying the cleanup continues where it stopped, and the
  quota is released only when the tree is gone for good.  Other items remain
  withdrawable while a slow delete runs.
"""

from __future__ import annotations

import json
import logging
import os
import stat as _stat_module
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger("recycle")

# POSIX defines st_blocks units as 512-byte blocks regardless of the filesystem
# block size; Linux always reports st_blocks in this unit.
BLOCK_SIZE = 512

STATE_FILENAME = "state.json"
ITEMS_DIRNAME = "items"
STATE_VERSION = 1

# Item lifecycle states.
STATE_ACTIVE = "active"
STATE_DELETING = "deleting"


class RecycleError(Exception):
    """Base class for recycle-bin failures carrying an HTTP status code."""

    status_code = 400


class RecycleNotFound(RecycleError):
    status_code = 404


class RecycleConflict(RecycleError):
    status_code = 409


class RecycleCapacityExceeded(RecycleError):
    status_code = 507


@dataclass
class RecycleItem:
    """One tree held in the recycle bin."""

    item_id: str
    restore_id: str
    original_path: str
    blocks: int
    moved_at: float
    expires_at: float
    state: str = STATE_ACTIVE
    bytes_used: int = 0

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "RecycleItem":
        return cls(
            item_id=raw["item_id"],
            restore_id=raw["restore_id"],
            original_path=raw["original_path"],
            blocks=int(raw["blocks"]),
            moved_at=float(raw["moved_at"]),
            expires_at=float(raw["expires_at"]),
            state=raw.get("state", STATE_ACTIVE),
            bytes_used=int(raw.get("bytes_used", 0)),
        )


def _is_reg(mode: int) -> bool:
    return _stat_module.S_ISREG(mode)


def measure_tree_blocks(tree_root: str) -> tuple[int, int]:
    """Return ``(blocks, logical_bytes)`` occupied by regular files in a tree.

    Each ``(st_dev, st_ino)`` is counted once, so hard links inside the tree
    share a single charge.  Symlinks are never followed (``lstat``) and only
    regular files contribute blocks.
    """

    seen: set[tuple[int, int]] = set()
    blocks = 0
    logical_bytes = 0
    for dirpath, dirnames, filenames in os.walk(tree_root, followlinks=False):
        # Deterministic order keeps logs and failures reproducible.
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except FileNotFoundError:
                # A racing unlink; the file no longer occupies anything.
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            if not _is_reg(st.st_mode):
                continue
            blocks += int(getattr(st, "st_blocks", 0))
            logical_bytes += st.st_size
    return blocks, logical_bytes


def _remove_tree_incrementally(root: str) -> None:
    """Delete ``root`` bottom-up, tolerating entries that have vanished.

    Unlike :func:`shutil.rmtree` this walks the tree itself so an error halfway
    through leaves a smaller, still-valid tree that a later retry can remove.
    Symlinked directories are removed as links (never descended into).
    """

    if not os.path.lexists(root):
        return

    # First pass: unlink every leaf entry and remove emptied directories.
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        for name in sorted(dirnames):
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                continue
            try:
                os.rmdir(path)
            except FileNotFoundError:
                pass
            except OSError:
                # Entries were re-created or a file remains; a later cleanup
                # pass walks this directory again.
                pass
    try:
        os.rmdir(root)
    except FileNotFoundError:
        pass


class RecycleBin:
    """Filesystem-backed, thread-safe recycle bin."""

    def __init__(
        self,
        base_dir: str,
        capacity_blocks: int,
        retention_seconds: float,
        restore_root: Optional[str] = None,
        *,
        clock=time.time,
    ) -> None:
        if capacity_blocks < 0:
            raise ValueError("capacity_blocks must be >= 0")
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be > 0")
        self.base_dir = os.path.abspath(base_dir)
        self.items_dir = os.path.join(self.base_dir, ITEMS_DIRNAME)
        self.state_path = os.path.join(self.base_dir, STATE_FILENAME)
        self.restore_root = os.path.abspath(restore_root) if restore_root else None
        self.capacity_blocks = capacity_blocks
        self.retention_seconds = float(retention_seconds)
        self._clock = clock
        self._lock = threading.RLock()
        # Items currently being deleted inside this process.
        self._deleting: set[str] = set()
        self._items: dict[str, RecycleItem] = {}
        self._next_serial = 1

    # ------------------------------------------------------------------ setup

    def startup(self) -> None:
        """Load persisted state and reconcile it with the filesystem."""

        os.makedirs(self.items_dir, exist_ok=True)
        self._load_state()
        with self._lock:
            self._reconcile()
            self._persist_locked()

    def _load_state(self) -> None:
        raw: dict
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            raw = {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("recycle state %s unreadable (%s); starting empty", self.state_path, exc)
            backup = f"{self.state_path}.corrupt-{uuid.uuid4().hex}"
            try:
                os.rename(self.state_path, backup)
                logger.warning("unreadable state moved aside to %s", backup)
            except OSError:
                pass
            raw = {}
        items: dict[str, RecycleItem] = {}
        for entry in raw.get("items", []):
            try:
                item = RecycleItem.from_dict(entry)
            except (KeyError, TypeError, ValueError):
                logger.warning("skipping malformed state entry: %r", entry)
                continue
            items[item.item_id] = item
        self._items = items
        self._next_serial = int(raw.get("next_serial", 1))

    def _persist_locked(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "next_serial": self._next_serial,
            "items": [item.to_dict() for item in sorted(self._items.values(), key=lambda i: i.moved_at)],
        }
        tmp = f"{self.state_path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    def _reconcile(self) -> None:
        """Repair state after a hard crash between a state write and a rename."""

        on_disk: set[str] = set()
        with os.scandir(self.items_dir) as scan:
            for entry in scan:
                if entry.is_dir(follow_symlinks=False):
                    on_disk.add(entry.name)
        known = set(self._items)

        # Orphaned directories with no state record: the rename happened but the
        # state write did not. Charge and track them as fresh, expired-soon-safe
        # entries so they are never silently lost.
        for name in sorted(on_disk - known):
            path = os.path.join(self.items_dir, name)
            blocks, logical = measure_tree_blocks(path)
            now = self._clock()
            # The true original path lived in the lost state file. If a restore
            # root is known, reconstruct the conventional published location.
            if self.restore_root is not None:
                original_path = os.path.join(self.restore_root, name)
            else:
                original_path = os.path.join(os.path.dirname(self.items_dir), name)
            item = RecycleItem(
                item_id=name,
                restore_id=name,
                original_path=original_path,
                blocks=blocks,
                moved_at=now,
                expires_at=now + self.retention_seconds,
                bytes_used=logical,
            )
            self._items[name] = item
            logger.warning("adopted orphaned recycle directory %s (%d blocks)", name, blocks)

        # State records without a directory: finish whatever operation was in
        # flight when the process died.
        for name in sorted(known - on_disk):
            item = self._items.pop(name)
            logger.warning(
                "discarding state for missing recycle directory %s (was %s)",
                name,
                item.state,
            )

    # ------------------------------------------------------------------ views

    def _used_blocks_locked(self) -> int:
        return sum(item.blocks for item in self._items.values())

    def status(self) -> dict:
        with self._lock:
            now = self._clock()
            items = sorted(self._items.values(), key=lambda i: i.moved_at)
            used = self._used_blocks_locked()
            return {
                "capacity_blocks": self.capacity_blocks,
                "used_blocks": used,
                "available_blocks": max(0, self.capacity_blocks - used),
                "block_size": BLOCK_SIZE,
                "capacity_bytes": self.capacity_blocks * BLOCK_SIZE,
                "used_bytes": sum(item.bytes_used for item in items),
                "retention_seconds": self.retention_seconds,
                "item_count": len(items),
                "deleting_count": sum(1 for i in items if i.state == STATE_DELETING),
                "items": [self._public_item_locked(item, now) for item in items],
            }

    def list_items(self) -> list[dict]:
        return self.status()["items"]

    def get_item(self, item_id: str) -> dict:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                raise RecycleNotFound(f"recycle item {item_id!r} not found")
            return self._public_item_locked(item, self._clock())

    def _public_item_locked(self, item: RecycleItem, now: float) -> dict:
        data = item.to_dict()
        data["expired"] = item.is_expired(now)
        data["stored_path"] = os.path.join(self.items_dir, item.item_id)
        return data

    # ------------------------------------------------------------------ move

    def move_in(self, restore_root: str, restore_id: str) -> dict:
        """Move the published tree ``<restore_root>/<restore_id>`` into the bin."""

        if not restore_id or restore_id != os.path.basename(restore_id) or restore_id in (".", ".."):
            raise RecycleError("invalid restore id")
        source = os.path.join(os.path.abspath(restore_root), restore_id)
        if os.path.islink(source) or not os.path.isdir(source):
            raise RecycleNotFound(f"published restore {restore_id!r} not found")

        # The move must be an atomic rename, which requires one filesystem.
        try:
            if os.stat(source).st_dev != os.stat(self.items_dir).st_dev:
                raise RecycleError(
                    "recycle bin and the restore output must be on the same "
                    f"filesystem (different devices: {source!r} vs {self.items_dir!r})"
                )
        except RecycleError:
            raise
        except OSError as exc:
            raise RecycleError(f"cannot stat recycle locations: {exc}") from exc

        # Measure before taking the lock so a huge tree does not block other
        # operations; the capacity decision itself is made under the lock.
        blocks, logical = measure_tree_blocks(source)

        with self._lock:
            for item in self._items.values():
                if item.restore_id == restore_id:
                    raise RecycleConflict(
                        f"restore {restore_id!r} is already in the recycle bin "
                        f"as item {item.item_id!r}"
                    )
            if self._used_blocks_locked() + blocks > self.capacity_blocks:
                raise RecycleCapacityExceeded(
                    f"recycle bin capacity exceeded: need {blocks} blocks, "
                    f"{self.capacity_blocks - self._used_blocks_locked()} available "
                    f"of {self.capacity_blocks}"
                )

            item_id = f"r{self._next_serial:08d}-{uuid.uuid4().hex[:12]}"
            self._next_serial += 1
            now = self._clock()
            item = RecycleItem(
                item_id=item_id,
                restore_id=restore_id,
                original_path=source,
                blocks=blocks,
                moved_at=now,
                expires_at=now + self.retention_seconds,
                bytes_used=logical,
            )
            target = os.path.join(self.items_dir, item_id)

            # Re-check under the lock: a concurrent move of the same restore id
            # (which measured before this critical section) may have won.
            if os.path.islink(source) or not os.path.isdir(source):
                raise RecycleNotFound(f"published restore {restore_id!r} not found")

            # Persist the record first. If we crash before the rename, startup
            # reconciliation drops the record (no directory exists). If we crash
            # after the rename, both the directory and its record exist.
            self._items[item_id] = item
            self._persist_locked()
            try:
                os.rename(source, target)
            except OSError as exc:
                self._items.pop(item_id, None)
                self._persist_locked()
                raise RecycleError(f"could not move restore into recycle bin: {exc}") from exc

            logger.info(
                "moved restore %s into recycle bin as %s (%d blocks, expires_at=%.0f)",
                restore_id,
                item_id,
                blocks,
                item.expires_at,
            )
            return self._public_item_locked(item, now)

    # --------------------------------------------------------------- withdraw

    def withdraw(self, item_id: str) -> dict:
        """Move an unexpired item back to its original path."""

        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                raise RecycleNotFound(f"recycle item {item_id!r} not found")
            if item.state == STATE_DELETING:
                raise RecycleConflict(
                    f"item {item_id!r} is being deleted and cannot be withdrawn"
                )
            now = self._clock()
            if item.is_expired(now):
                raise RecycleConflict(
                    f"item {item_id!r} expired at {item.expires_at:.0f}; "
                    "only cleanup is allowed"
                )
            if os.path.lexists(item.original_path):
                raise RecycleConflict(
                    f"original path {item.original_path!r} already exists; "
                    "refusing to overwrite it"
                )
            stored = os.path.join(self.items_dir, item_id)
            if not os.path.isdir(stored):
                raise RecycleNotFound(f"stored directory for item {item_id!r} is missing")

            parent = os.path.dirname(item.original_path)
            created_parent = False
            if not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
                created_parent = True

            try:
                os.rename(stored, item.original_path)
            except OSError as exc:
                if created_parent:
                    try:
                        os.rmdir(parent)
                    except OSError:
                        pass
                raise RecycleError(f"could not withdraw item: {exc}") from exc

            self._items.pop(item_id, None)
            self._persist_locked()
            logger.info("withdrew item %s back to %s", item_id, item.original_path)
            return {
                "item_id": item_id,
                "restore_id": item.restore_id,
                "restored_path": item.original_path,
                "blocks_returned": item.blocks,
            }

    # ---------------------------------------------------------------- cleanup

    def cleanup_expired(self, *, item_id: Optional[str] = None) -> dict:
        """Delete expired items (or one specific item), retrying partial deletes.

        Returns a report.  The heavy deletion happens without the global lock,
        so other items stay withdrawable while a large tree is being removed.
        """

        with self._lock:
            now = self._clock()
            targets: list[RecycleItem] = []
            if item_id is not None:
                item = self._items.get(item_id)
                if item is None:
                    raise RecycleNotFound(f"recycle item {item_id!r} not found")
                if item.state != STATE_DELETING and not item.is_expired(now):
                    raise RecycleConflict(
                        f"item {item_id!r} has not expired yet; "
                        f"expires at {item.expires_at:.0f}"
                    )
                targets = [item]
            else:
                targets = [
                    item
                    for item in self._items.values()
                    if item.state == STATE_DELETING or item.is_expired(now)
                ]

            claimed: list[RecycleItem] = []
            for item in targets:
                if item.item_id in self._deleting:
                    continue
                self._deleting.add(item.item_id)
                if item.state != STATE_DELETING:
                    item.state = STATE_DELETING
                    self._persist_locked()
                claimed.append(item)

        deleted: list[str] = []
        failed: list[dict] = []
        try:
            for item in claimed:
                ok, detail = self._delete_one(item)
                if ok:
                    deleted.append(item.item_id)
                else:
                    failed.append({"item_id": item.item_id, "error": detail})
        finally:
            with self._lock:
                for item in claimed:
                    self._deleting.discard(item.item_id)

        return {
            "deleted": deleted,
            "failed": failed,
            "skipped_busy": sorted(
                {i.item_id for i in targets} - set(deleted) - {f["item_id"] for f in failed}
            ),
        }

    def _delete_one(self, item: RecycleItem) -> tuple[bool, str]:
        stored = os.path.join(self.items_dir, item.item_id)
        try:
            _remove_tree_incrementally(stored)
        except OSError as exc:
            logger.warning("cleanup of %s interrupted: %s", item.item_id, exc)
            return False, str(exc)

        if os.path.lexists(stored):
            logger.warning("cleanup of %s incomplete, quota remains reserved", item.item_id)
            return False, "directory still present after deletion pass"

        # Gone for good: only now release the quota and drop the record.
        with self._lock:
            if item.item_id in self._items:
                self._items.pop(item.item_id)
                self._persist_locked()
        logger.info("cleaned up item %s, released %d blocks", item.item_id, item.blocks)
        return True, ""
