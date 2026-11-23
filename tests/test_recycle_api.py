"""API-level tests for the recycle-bin endpoints."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.main as main

    archives = tmp_path / "backups"
    restored = tmp_path / "restored"
    archives.mkdir()
    monkeypatch.setattr(main, "ARCHIVE_DIR", str(archives))
    monkeypatch.setattr(main, "OUTPUT_DIR", str(restored))
    monkeypatch.delenv("RECYCLE_DIR", raising=False)
    monkeypatch.setattr(main, "RECYCLE_CAPACITY_BLOCKS", 10_000)
    monkeypatch.setattr(main, "RECYCLE_RETENTION_SECONDS", 3600.0)
    monkeypatch.setattr(main, "recycle_bin", main.build_recycle_bin(str(restored), 10_000, 3600.0))
    with TestClient(main.app) as c:
        yield c, archives, restored, main


def _make_archive(directory, name="backup.tar.gz"):
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import make_sample_archive

    path = os.path.join(str(directory), name)
    make_sample_archive.build(path)
    return name


def _restore(client, archives, path="data/2024"):
    name = _make_archive(archives)
    response = client.post("/api/restores", json={"archive": name, "path": path})
    assert response.status_code == 201, response.text
    return response.json()


def test_recycle_status_endpoint(client):
    c, _a, _r, main = client
    body = c.get("/api/recycle").json()
    assert body["capacity_blocks"] == 10_000
    assert body["used_blocks"] == 0
    assert body["block_size"] == 512
    assert body["items"] == []


def test_full_move_withdraw_cleanup_cycle(client):
    c, archives, restored, main = client
    restored_body = _restore(c, archives)
    rid = restored_body["restore_id"]
    # The recycle bin moves the whole <output>/<restore-id> directory, which
    # contains the published basename directory.
    published_root = os.path.join(str(restored), rid)
    original = restored_body["restored_path"]
    assert os.path.isdir(original)

    # Move into the bin: original path disappears immediately.
    move = c.post("/api/recycle/items", json={"restore_id": rid})
    assert move.status_code == 201, move.text
    item = move.json()
    item_id = item["item_id"]
    assert item["restore_id"] == rid
    assert item["blocks"] > 0
    assert not os.path.exists(published_root)
    assert os.path.isdir(os.path.join(str(restored), ".recycle", "items", item_id))

    listing = c.get("/api/recycle/items").json()
    assert [i["item_id"] for i in listing] == [item_id]
    assert c.get(f"/api/recycle/items/{item_id}").json()["item_id"] == item_id

    # Withdraw before expiry: the whole tree returns to its original path.
    withdrawn = c.post(f"/api/recycle/items/{item_id}/withdraw")
    assert withdrawn.status_code == 200, withdrawn.text
    assert withdrawn.json()["restored_path"] == published_root
    assert os.path.isfile(os.path.join(original, "inside.txt"))
    assert c.get("/api/recycle").json()["used_blocks"] == 0


def test_move_unknown_restore_is_404(client):
    c, _a, _r, _m = client
    response = c.post("/api/recycle/items", json={"restore_id": "deadbeef"})
    assert response.status_code == 404


def test_capacity_exceeded_returns_507_and_keeps_path(client):
    c, archives, restored, main = client
    body = _restore(c, archives)
    rid = body["restore_id"]
    original = body["restored_path"]

    bin_ = main.recycle_bin
    bin_.capacity_blocks = 1  # nothing fits

    response = c.post("/api/recycle/items", json={"restore_id": rid})
    assert response.status_code == 507
    assert "capacity" in response.json()["detail"].lower()
    assert os.path.isdir(original)
    assert c.get("/api/recycle").json()["used_blocks"] == 0


def test_expired_item_rejects_withdraw_but_allows_cleanup(client):
    c, archives, restored, main = client
    body = _restore(c, archives)
    item = c.post("/api/recycle/items", json={"restore_id": body["restore_id"]}).json()
    item_id = item["item_id"]

    # Expiry is fixed at move-in time (restarts never recompute it); simulate
    # the clock reaching it by rewinding the stored timestamp.
    bin_ = main.recycle_bin
    bin_._items[item_id].expires_at = bin_._clock() - 1
    assert c.get(f"/api/recycle/items/{item_id}").json()["expired"] is True

    forbidden = c.post(f"/api/recycle/items/{item_id}/withdraw")
    assert forbidden.status_code == 409
    assert "expired" in forbidden.json()["detail"].lower()

    report = c.post("/api/recycle/cleanup")
    assert report.status_code == 200
    assert item_id in report.json()["deleted"]
    assert c.get("/api/recycle").json()["used_blocks"] == 0
    assert c.get(f"/api/recycle/items/{item_id}").status_code == 404


def test_cleanup_ignores_unexpired_items(client):
    c, archives, _r, _m = client
    body = _restore(c, archives)
    c.post("/api/recycle/items", json={"restore_id": body["restore_id"]})
    report = c.post("/api/recycle/cleanup").json()
    assert report["deleted"] == []
    assert c.get("/api/recycle").json()["item_count"] == 1


def test_cleanup_unknown_item_is_404(client):
    c, _a, _r, _m = client
    assert c.post("/api/recycle/items/nope/cleanup").status_code == 404


def test_interrupted_cleanup_is_retryable_over_http(client, monkeypatch):
    c, archives, _r, main = client
    body = _restore(c, archives)
    item = c.post("/api/recycle/items", json={"restore_id": body["restore_id"]}).json()
    item_id = item["item_id"]
    main.recycle_bin._items[item_id].expires_at = main.recycle_bin._clock() - 1

    import app.recycle as recycle_mod

    real_unlink = os.unlink
    calls = {"n": 0}

    def flaky(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("boom")
        return real_unlink(path, *args, **kwargs)

    recycle_mod.os.unlink = flaky
    first = c.post("/api/recycle/cleanup")
    assert first.status_code == 200
    assert first.json()["failed"][0]["item_id"] == item_id
    # Quota remains reserved while the directory is partly there.
    assert c.get("/api/recycle").json()["deleting_count"] == 1

    recycle_mod.os.unlink = real_unlink
    second = c.post("/api/recycle/cleanup")
    assert second.status_code == 200
    assert item_id in second.json()["deleted"]
    assert c.get("/api/recycle").json()["used_blocks"] == 0


def test_restore_response_contains_restore_id(client):
    c, archives, _r, _m = client
    body = _restore(c, archives)
    assert len(body["restore_id"]) == 32
    # restored_path is <output>/<restore-id>/<basename>.
    assert os.path.basename(os.path.dirname(body["restored_path"])) == body["restore_id"]
