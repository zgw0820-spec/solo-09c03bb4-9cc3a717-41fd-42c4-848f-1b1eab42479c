"""Unit tests for the recycle bin."""

from __future__ import annotations

import json
import os
import threading

import pytest

from app.recycle import (
    BLOCK_SIZE,
    STATE_ACTIVE,
    STATE_DELETING,
    RecycleBin,
    RecycleCapacityExceeded,
    RecycleConflict,
    RecycleError,
    RecycleNotFound,
    measure_tree_blocks,
)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def paths(tmp_path):
    published = tmp_path / "restored"
    published.mkdir()
    bin_dir = tmp_path / "recycle"
    return published, bin_dir


@pytest.fixture
def clock():
    return FakeClock()


def _make_bin(bin_dir, clock, capacity_blocks=10_000, retention=3600.0):
    return RecycleBin(str(bin_dir), capacity_blocks, retention, clock=clock)


def _make_restore(published, restore_id, files=None, hardlinks=None):
    """Create a published restore tree; files maps rel-path -> data bytes."""

    root = published / restore_id
    root.mkdir()
    for rel, data in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for link_rel, target_rel in hardlinks or []:
        target = root / target_rel
        link = root / link_rel
        link.parent.mkdir(parents=True, exist_ok=True)
        os.link(target, link)
    return root


# --------------------------------------------------------------------- billing


def test_blocks_match_lstat_and_hardlinks_counted_once(paths):
    published, bin_dir = paths
    data = b"x" * 3000  # spans several 512-byte blocks
    root = _make_restore(
        published,
        "rid-1",
        files={"a.bin": data, "nested/b.bin": b"y"},
        hardlinks=[("a-link.bin", "a.bin"), ("nested/a-again.bin", "a.bin")],
    )
    blocks, logical = measure_tree_blocks(str(root))

    expected_one = os.lstat(str(root / "a.bin")).st_blocks
    expected_b = os.lstat(str(root / "nested/b.bin")).st_blocks
    # Three names share one inode -> charged once, plus the small file.
    assert blocks == expected_one + expected_b
    # Logical bytes likewise count the inode once.
    assert logical == 3000 + 1


def test_directories_and_symlinks_are_free(paths):
    published, bin_dir = paths
    root = _make_restore(published, "rid-1", files={"f": b"hello"})
    (root / "emptydir").mkdir()
    os.symlink("f", root / "link-to-f")
    blocks, _ = measure_tree_blocks(str(root))
    assert blocks == os.lstat(str(root / "f")).st_blocks


def test_two_separate_trees_are_billed_independently(paths):
    published, bin_dir = paths
    r1 = _make_restore(published, "rid-1", files={"f": b"a" * 2000})
    r2 = _make_restore(published, "rid-2", files={"f": b"b" * 2000})
    b1, _ = measure_tree_blocks(str(r1))
    b2, _ = measure_tree_blocks(str(r2))
    assert b1 == b2 and b1 > 0


# --------------------------------------------------------------------- move in


def test_move_in_removes_original_and_persists_item(paths, clock):
    published, bin_dir = paths
    root = _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()

    item = bin_.move_in(str(published), "rid-1")

    assert not root.exists()
    assert os.path.isdir(os.path.join(str(bin_dir), "items", item["item_id"]))
    assert item["restore_id"] == "rid-1"
    assert item["state"] == STATE_ACTIVE
    assert item["blocks"] > 0
    assert item["expires_at"] == clock.now + 3600.0
    assert item["original_path"] == str(root)
    status = bin_.status()
    assert status["used_blocks"] == item["blocks"]
    assert status["available_blocks"] == status["capacity_blocks"] - item["blocks"]

    # State is on disk.
    with open(bin_dir / "state.json", encoding="utf-8") as fh:
        raw = json.load(fh)
    assert [i["item_id"] for i in raw["items"]] == [item["item_id"]]


def test_move_in_unknown_restore_raises_not_found(paths, clock):
    published, bin_dir = paths
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    with pytest.raises(RecycleNotFound):
        bin_.move_in(str(published), "ghost")


def test_move_in_rejects_invalid_restore_id(paths, clock):
    published, bin_dir = paths
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    for bad in ("", ".", "..", "a/b", "../x"):
        with pytest.raises(Exception):
            bin_.move_in(str(published), bad)


def test_move_in_capacity_exceeded_keeps_original(paths, clock):
    published, bin_dir = paths
    root = _make_restore(published, "rid-1", files={"a": b"z" * 4096})
    needed, _ = measure_tree_blocks(str(root))
    bin_ = _make_bin(bin_dir, clock, capacity_blocks=max(1, needed - 1))
    bin_.startup()

    with pytest.raises(RecycleCapacityExceeded):
        bin_.move_in(str(published), "rid-1")
    # Original path survives a rejected move and nothing is stored.
    assert root.is_dir()
    assert bin_.status()["used_blocks"] == 0
    assert os.listdir(bin_dir / "items") == []


def test_concurrent_move_ins_never_exceed_capacity(paths, clock):
    published, bin_dir = paths
    # Ten identical small trees; capacity only fits a few.
    for i in range(10):
        _make_restore(published, f"rid-{i}", files={"a": b"q" * 2000})
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    one_blocks, _ = measure_tree_blocks(str(published / "rid-0"))
    bin_.capacity_blocks = one_blocks * 3  # exactly three fit

    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(i):
        try:
            bin_.move_in(str(published), f"rid-{i}")
            result = "ok"
        except RecycleCapacityExceeded:
            result = "full"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    accepted = outcomes.count("ok")
    assert accepted == 3
    assert outcomes.count("full") == 7
    # Capacity is never overshot and the quota matches the stored directories.
    assert bin_.status()["used_blocks"] <= bin_.capacity_blocks
    assert bin_.status()["used_blocks"] == one_blocks * 3
    # Rejected restores are still present at their original paths.
    remaining = sorted(os.listdir(published))
    assert len(remaining) == 7


def test_concurrent_moves_of_same_restore_only_one_succeeds(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()

    outcomes: list[str] = []
    lock = threading.Lock()

    def worker():
        try:
            bin_.move_in(str(published), "rid-1")
            result = "ok"
        except RecycleError:
            result = "rejected"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("ok") == 1
    assert outcomes.count("rejected") == 5
    assert bin_.status()["item_count"] == 1
    # No orphaned state records or directories.
    assert len(os.listdir(bin_dir / "items")) == 1


def test_same_restore_cannot_be_moved_twice(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    bin_.move_in(str(published), "rid-1")

    # A new restore reusing the same id must be rejected as a duplicate while
    # the first item is still held, rather than silently billed a second time.
    _make_restore(published, "rid-1", files={"a": b"other"})
    with pytest.raises(RecycleConflict, match="already in the recycle bin"):
        bin_.move_in(str(published), "rid-1")
    # The second tree stays at its original path.
    assert (published / "rid-1" / "a").read_bytes() == b"other"


# ------------------------------------------------------------------- withdraw


def test_withdraw_returns_tree_to_original_path(paths, clock):
    published, bin_dir = paths
    root = _make_restore(published, "rid-1", files={"a": b"data", "n/b": b"x"})
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")
    assert not root.exists()

    result = bin_.withdraw(item["item_id"])

    assert result["restored_path"] == str(root)
    assert root.is_dir()
    assert (root / "a").read_bytes() == b"data"
    assert (root / "n" / "b").read_bytes() == b"x"
    # Hard links preserved through the rename (same inode relationships).
    assert not os.path.exists(os.path.join(str(bin_dir), "items", item["item_id"]))
    assert bin_.status()["used_blocks"] == 0
    with pytest.raises(RecycleNotFound):
        bin_.get_item(item["item_id"])


def test_withdraw_unknown_item_raises(paths, clock):
    published, bin_dir = paths
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    with pytest.raises(RecycleNotFound):
        bin_.withdraw("nope")


def test_expired_item_cannot_be_withdrawn(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock, retention=60.0)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")

    clock.advance(61)
    assert bin_.get_item(item["item_id"])["expired"] is True
    with pytest.raises(RecycleConflict, match="expired"):
        bin_.withdraw(item["item_id"])
    # Still stored, still billed.
    assert bin_.status()["used_blocks"] > 0


def test_withdraw_refuses_when_original_path_reappeared(paths, clock):
    published, bin_dir = paths
    root = _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")

    # Something else now occupies the original path.
    root.mkdir()
    with pytest.raises(RecycleConflict):
        bin_.withdraw(item["item_id"])
    # Item stays in the bin.
    assert bin_.get_item(item["item_id"])["item_id"] == item["item_id"]


def test_re_move_after_withdraw_is_allowed(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    first = bin_.move_in(str(published), "rid-1")
    bin_.withdraw(first["item_id"])
    second = bin_.move_in(str(published), "rid-1")
    assert second["item_id"] != first["item_id"]


# ------------------------------------------------------------------ retention


def test_retention_clock_survives_restart(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock, retention=100.0)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")
    original_expiry = item["expires_at"]

    clock.advance(90)
    # Simulate a process restart: a brand-new bin reads the same state file.
    restarted = _make_bin(bin_dir, clock, retention=100.0)
    restarted.startup()
    loaded = restarted.get_item(item["item_id"])
    assert loaded["moved_at"] == item["moved_at"]
    assert loaded["expires_at"] == original_expiry
    assert loaded["expired"] is False

    clock.advance(11)
    assert restarted.get_item(item["item_id"])["expired"] is True


def test_changing_retention_config_does_not_reshuffle_existing_items(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock, retention=100.0)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")

    restarted = _make_bin(bin_dir, clock, retention=9999.0)
    restarted.startup()
    # Persisted expiry wins over the new config.
    assert restarted.get_item(item["item_id"])["expires_at"] == item["expires_at"]


# -------------------------------------------------------------------- cleanup


def test_cleanup_deletes_expired_and_releases_quota_only_when_gone(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data" * 100})
    bin_ = _make_bin(bin_dir, clock, retention=60.0)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")
    used = bin_.status()["used_blocks"]
    assert used > 0

    # Not expired yet: nothing to do.
    report = bin_.cleanup_expired()
    assert report["deleted"] == []
    assert bin_.status()["used_blocks"] == used

    clock.advance(61)
    report = bin_.cleanup_expired()
    assert report["deleted"] == [item["item_id"]]
    assert bin_.status()["used_blocks"] == 0
    assert not os.path.exists(os.path.join(str(bin_dir), "items", item["item_id"]))
    with pytest.raises(RecycleNotFound):
        bin_.get_item(item["item_id"])


def test_cleanup_one_requires_expiry(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock, retention=60.0)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")
    with pytest.raises(RecycleConflict):
        bin_.cleanup_expired(item_id=item["item_id"])


def test_interrupted_cleanup_keeps_quota_and_can_retry(paths, clock, monkeypatch):
    published, bin_dir = paths
    root = _make_restore(
        published,
        "rid-1",
        files={"a": b"a" * 100, "b": b"b" * 100, "n/c": b"c" * 100},
    )
    bin_ = _make_bin(bin_dir, clock, retention=60.0)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")
    used = bin_.status()["used_blocks"]
    clock.advance(61)

    import app.recycle as recycle_mod

    real_unlink = os.unlink
    calls = {"n": 0}

    def flaky_unlink(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated interruption")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(recycle_mod.os, "unlink", flaky_unlink)
    report = bin_.cleanup_expired()
    assert report["failed"] and report["failed"][0]["item_id"] == item["item_id"]

    # Item is marked deleting, directory still present, quota still reserved.
    stored = os.path.join(str(bin_dir), "items", item["item_id"])
    assert os.path.isdir(stored)
    assert bin_.get_item(item["item_id"])["state"] == STATE_DELETING
    assert bin_.status()["used_blocks"] == used
    assert bin_.status()["deleting_count"] == 1

    # Restart with the failure gone: the deleting state is persisted, so the
    # item is picked up even by a fresh process, and quota released at the end.
    monkeypatch.undo()
    restarted = _make_bin(bin_dir, clock, retention=60.0)
    restarted.startup()
    assert restarted.get_item(item["item_id"])["state"] == STATE_DELETING
    report = restarted.cleanup_expired()
    assert report["deleted"] == [item["item_id"]]
    assert not os.path.lexists(stored)
    assert restarted.status()["used_blocks"] == 0


def test_slow_cleanup_does_not_block_other_withdrawals(paths, clock, monkeypatch):
    published, bin_dir = paths
    _make_restore(published, "rid-big", files={"big": b"x" * 100})
    _make_restore(published, "rid-keep", files={"keep": b"y" * 100})
    bin_ = _make_bin(bin_dir, clock, retention=60.0)
    bin_.startup()
    big = bin_.move_in(str(published), "rid-big")
    clock.advance(61)
    # The keep item enters the bin after the first item has already expired, so
    # the cleanup run below only claims the big one.
    keep = bin_.move_in(str(published), "rid-keep")
    assert bin_.get_item(big["item_id"])["expired"] is True
    assert bin_.get_item(keep["item_id"])["expired"] is False

    import app.recycle as recycle_mod

    entered = threading.Event()
    release = threading.Event()
    real_rmdir = os.rmdir

    def blocking_rmdir(path, *args, **kwargs):
        if str(path).endswith(big["item_id"]):
            entered.set()
            release.wait(timeout=5)
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(recycle_mod.os, "rmdir", blocking_rmdir)
    t = threading.Thread(target=bin_.cleanup_expired)
    t.start()
    assert entered.wait(timeout=5)

    # While the big tree is being deleted, the other item stays fully usable.
    withdrawn = bin_.withdraw(keep["item_id"])
    assert os.path.isfile(os.path.join(withdrawn["restored_path"], "keep"))

    release.set()
    t.join(timeout=5)
    assert not t.is_alive()


def test_cleanup_skips_items_already_being_deleted(paths, clock, monkeypatch):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock, retention=60.0)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")
    clock.advance(61)

    import app.recycle as recycle_mod

    entered = threading.Event()
    release = threading.Event()
    real_rmdir = os.rmdir

    def blocking_rmdir(path, *args, **kwargs):
        if str(path).endswith(item["item_id"]):
            entered.set()
            release.wait(timeout=5)
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(recycle_mod.os, "rmdir", blocking_rmdir)
    t = threading.Thread(target=bin_.cleanup_expired)
    t.start()
    assert entered.wait(timeout=5)

    second = bin_.cleanup_expired()
    assert item["item_id"] in second["skipped_busy"]
    release.set()
    t.join(timeout=5)


# ----------------------------------------------------------------- restart / FS


def test_orphaned_directory_without_state_is_adopted(paths, clock):
    published, bin_dir = paths
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    # Simulate: rename happened, state write did not.
    orphan_dir = bin_dir / "items" / "orphan123"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "f").write_bytes(b"orphan" * 500)

    restarted = _make_bin(bin_dir, clock)
    restarted.startup()
    item = restarted.get_item("orphan123")
    assert item["blocks"] > 0
    assert restarted.status()["used_blocks"] == item["blocks"]


def test_state_record_without_directory_is_dropped(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"data"})
    bin_ = _make_bin(bin_dir, clock)
    bin_.startup()
    item = bin_.move_in(str(published), "rid-1")
    # Simulate: directory vanished externally after the state was written.
    import shutil

    shutil.rmtree(bin_dir / "items" / item["item_id"])

    restarted = _make_bin(bin_dir, clock)
    restarted.startup()
    with pytest.raises(RecycleNotFound):
        restarted.get_item(item["item_id"])
    assert restarted.status()["used_blocks"] == 0


def test_zero_capacity_rejects_everything(paths, clock):
    published, bin_dir = paths
    _make_restore(published, "rid-1", files={"a": b"x"})
    bin_ = _make_bin(bin_dir, clock, capacity_blocks=0)
    bin_.startup()
    with pytest.raises(RecycleCapacityExceeded):
        bin_.move_in(str(published), "rid-1")


def test_block_size_constant_is_512(paths):
    assert BLOCK_SIZE == 512
