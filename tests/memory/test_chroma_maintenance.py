"""Smart-RAG Phase 2 (P4 loadability) — chroma_maintenance.

Deterministic guardrail for the ChromaDB on-disk store:

  * ``store_size_bytes``            — os.walk bytes-budget measure
  * ``list_on_disk_segment_dirs``   — UUID-named segment dirs on disk
  * ``list_segment_ids``            — ids from the ``segments`` table
  * ``find_orphan_segment_dirs``    — on-disk dirs with no ``segments`` row
  * ``reclaim_orphan_segment_dirs`` — rmtree the orphans
  * ``vacuum_store``                — ``chroma vacuum`` CLI wrapper (injectable)
  * ``enforce_size_budget``         — boot-time decision fn

ROOT motivating this module (proven 2026-07-27, memory
``project_smart_rag_phase2_vacuum_orphan_hnsw_20260727``): ``chroma vacuum``
reclaims the SQLite bytes of deleted rows but NEVER removes the orphaned HNSW
segment directory left behind by ``delete_collection`` — those dirs leak
permanently. Orphan-GC is the missing half.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import time

import pytest

_HAS_CHROMA = importlib.util.find_spec("chromadb") is not None
_chroma_skip = pytest.mark.skipif(
    not _HAS_CHROMA, reason="chromadb not installed in this venv"
)

# Valid-shaped chroma segment UUIDs (dir names) used across tests.
_LIVE_UUID = "d96da37c-5120-4d52-80da-c7c45f1b84a7"
_ORPHAN_UUID = "3477b420-cfba-4da5-928f-09703f728ca9"
_ORPHAN_UUID2 = "889549a6-ffef-4a3d-a154-680a4145b172"


# ─── fixtures / helpers ──────────────────────────────────────────────────────

def _make_segments_db(db_dir: str, segment_ids: list[str]) -> str:
    """Fabricate a minimal chroma.sqlite3 with a ``segments`` table."""
    os.makedirs(db_dir, exist_ok=True)
    path = os.path.join(db_dir, "chroma.sqlite3")
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE segments (id TEXT PRIMARY KEY, scope TEXT, "
            "collection TEXT)"
        )
        for i, sid in enumerate(segment_ids):
            conn.execute(
                "INSERT INTO segments (id, scope, collection) VALUES (?,?,?)",
                (sid, "VECTOR", f"col{i}"),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def _make_seg_dir(db_dir: str, name: str, *, nbytes: int = 1024,
                  age_seconds: float = 0.0) -> str:
    """Create a UUID-named segment dir with a payload file, optionally aged."""
    d = os.path.join(db_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "data_level0.bin"), "wb") as fh:
        fh.write(b"\0" * nbytes)
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(d, (old, old))
    return d


# ─── store_size_bytes ────────────────────────────────────────────────────────

def test_store_size_bytes_sums_all_files(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)
    with open(os.path.join(db, "chroma.sqlite3"), "wb") as fh:
        fh.write(b"x" * 5000)
    _make_seg_dir(db, _LIVE_UUID, nbytes=2000)

    assert cm.store_size_bytes(db) == 7000


# ─── list_on_disk_segment_dirs ───────────────────────────────────────────────

def test_list_on_disk_segment_dirs_only_uuid_subdirs(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)
    _make_seg_dir(db, _LIVE_UUID)
    _make_seg_dir(db, _ORPHAN_UUID)
    # Non-uuid subdir + a plain file must be ignored.
    os.makedirs(os.path.join(db, "not-a-segment"))
    with open(os.path.join(db, "chroma.sqlite3"), "wb") as fh:
        fh.write(b"x")

    got = cm.list_on_disk_segment_dirs(db)
    assert got == {_LIVE_UUID, _ORPHAN_UUID}


# ─── list_segment_ids ────────────────────────────────────────────────────────

def test_list_segment_ids_reads_segments_table(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    _make_segments_db(db, [_LIVE_UUID, _ORPHAN_UUID])
    assert cm.list_segment_ids(db) == {_LIVE_UUID, _ORPHAN_UUID}


# ─── find_orphan_segment_dirs ────────────────────────────────────────────────

def test_find_orphan_flags_dir_with_no_segments_row(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    _make_segments_db(db, [_LIVE_UUID])           # only LIVE is registered
    _make_seg_dir(db, _LIVE_UUID, age_seconds=7200)
    _make_seg_dir(db, _ORPHAN_UUID, age_seconds=7200)  # leaked, aged 2h

    orphans = cm.find_orphan_segment_dirs(db, min_age_seconds=3600)
    assert orphans == [_ORPHAN_UUID]


def test_find_orphan_fail_safe_when_segments_table_unreadable(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)
    # No chroma.sqlite3 at all — must NEVER treat every dir as orphan.
    _make_seg_dir(db, _ORPHAN_UUID, age_seconds=7200)

    assert cm.find_orphan_segment_dirs(db, min_age_seconds=3600) == []


def test_find_orphan_fail_safe_when_segments_table_missing(tmp_path):
    """Schema drift (renamed/absent ``segments`` table) must fail safe → []."""
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)
    conn = sqlite3.connect(os.path.join(db, "chroma.sqlite3"))
    conn.execute("CREATE TABLE collections (id TEXT)")  # no `segments` table
    conn.commit()
    conn.close()
    _make_seg_dir(db, _ORPHAN_UUID, age_seconds=7200)

    assert cm.find_orphan_segment_dirs(db, min_age_seconds=3600) == []


def test_find_orphan_skips_recent_dirs(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    _make_segments_db(db, [_LIVE_UUID])
    # Orphan just created (age 0) — an in-flight segment could look like this,
    # so the min-age guard must skip it.
    _make_seg_dir(db, _ORPHAN_UUID, age_seconds=0)

    assert cm.find_orphan_segment_dirs(db, min_age_seconds=3600) == []
    # With the guard disabled it IS flagged.
    assert cm.find_orphan_segment_dirs(db, min_age_seconds=0) == [_ORPHAN_UUID]


# ─── reclaim_orphan_segment_dirs ─────────────────────────────────────────────

def test_reclaim_removes_orphan_keeps_live(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    _make_segments_db(db, [_LIVE_UUID])
    _make_seg_dir(db, _LIVE_UUID, nbytes=1000, age_seconds=7200)
    _make_seg_dir(db, _ORPHAN_UUID, nbytes=4096, age_seconds=7200)

    report = cm.reclaim_orphan_segment_dirs(db, min_age_seconds=3600)

    assert report["removed"] == [_ORPHAN_UUID]
    assert report["bytes_freed"] >= 4096
    assert not os.path.exists(os.path.join(db, _ORPHAN_UUID))   # gone
    assert os.path.exists(os.path.join(db, _LIVE_UUID))          # kept


def test_reclaim_dry_run_deletes_nothing(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    _make_segments_db(db, [_LIVE_UUID])
    _make_seg_dir(db, _ORPHAN_UUID, nbytes=4096, age_seconds=7200)

    report = cm.reclaim_orphan_segment_dirs(db, min_age_seconds=3600, dry_run=True)

    assert report["removed"] == [_ORPHAN_UUID]
    assert os.path.exists(os.path.join(db, _ORPHAN_UUID))   # still there


# ─── vacuum_store ────────────────────────────────────────────────────────────

def test_vacuum_store_invokes_runner_with_expected_args(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)
    captured = {}

    def fake_runner(args):
        captured["args"] = args
        return 0

    ok = cm.vacuum_store(db, runner=fake_runner, timeout=123)
    assert ok is True
    a = captured["args"]
    assert "vacuum" in a and "--force" in a
    assert "--path" in a and db in a
    assert "123" in a


def test_vacuum_store_returns_false_on_nonzero_exit(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)
    assert cm.vacuum_store(db, runner=lambda args: 1) is False


# ─── enforce_size_budget (boot gate decision) ────────────────────────────────

def test_enforce_size_budget_under_budget_is_noop(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)
    calls = []
    report = cm.enforce_size_budget(
        db,
        budget_bytes=1_000_000,
        size_fn=lambda d: 500,
        reclaim_fn=lambda d, **k: calls.append("reclaim") or {},
        vacuum_fn=lambda d, **k: calls.append("vacuum") or True,
    )
    assert report["over_budget"] is False
    assert report["action"] == "none"
    assert calls == []          # nothing reclaimed/vacuumed under budget


def test_enforce_size_budget_over_budget_reclaims_and_vacuums(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)
    calls = []
    sizes = iter([5_000_000, 1_000_000])   # before, after
    report = cm.enforce_size_budget(
        db,
        budget_bytes=2_000_000,
        size_fn=lambda d: next(sizes),
        reclaim_fn=lambda d, **k: (calls.append("reclaim"),
                                   {"removed": [_ORPHAN_UUID], "bytes_freed": 4_000_000})[1],
        vacuum_fn=lambda d, **k: (calls.append("vacuum"), True)[1],
    )
    assert report["over_budget"] is True
    assert report["action"] == "reclaim"
    assert calls == ["reclaim", "vacuum"]
    assert report["reclaim"]["removed"] == [_ORPHAN_UUID]
    assert report["vacuumed"] is True
    assert report["size_after"] == 1_000_000


# ─── cross-process destructive lock ──────────────────────────────────────────

def test_destructive_lock_is_exclusive(tmp_path):
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)
    with cm._destructive_lock(db) as outer:
        assert outer is True
        # A second acquirer (different fd) must be refused while held.
        with cm._destructive_lock(db) as inner:
            assert inner is False
    # Released → re-acquirable.
    with cm._destructive_lock(db) as again:
        assert again is True


def test_enforce_size_budget_skips_reclaim_when_lock_unavailable(tmp_path, monkeypatch):
    import contextlib as _cl
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    os.makedirs(db)

    @_cl.contextmanager
    def _held(_db):
        yield False

    monkeypatch.setattr(cm, "_destructive_lock", _held)
    calls = []
    report = cm.enforce_size_budget(
        db,
        budget_bytes=1,
        size_fn=lambda d: 999_999,
        reclaim_fn=lambda d, **k: (calls.append("r"), {})[1],
        vacuum_fn=lambda d, **k: (calls.append("v"), True)[1],
    )
    assert report["over_budget"] is True
    assert report["action"] == "skipped_locked"
    assert calls == []  # never touched the store while another proc holds lock


# ─── real-schema safety: healthy store yields ZERO orphans ───────────────────

@_chroma_skip
@pytest.mark.asyncio
async def test_healthy_real_store_has_zero_orphans(tmp_path, monkeypatch):
    """Against a REAL chroma store, a healthy (nothing deleted) store must
    never flag a live segment dir as an orphan. This is the safety invariant:
    the GC can only ever remove genuinely-leaked dirs."""
    from src.memory import vector_store as vs
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    monkeypatch.setattr(vs, "_DB_DIR", db, raising=False)
    monkeypatch.setattr(vs, "_initialized", False, raising=False)
    monkeypatch.setattr(vs, "_client", None, raising=False)
    monkeypatch.setattr(vs, "_collections", {}, raising=False)
    monkeypatch.setattr(vs, "_namespaced_collections", {}, raising=False)

    async def _fake_embed(text, is_query=False):
        return [float((ord(c) % 13) / 13.0) for c in (text + "xxxxxxxx")[:8]]

    monkeypatch.setattr(vs, "_get_embed_fn", lambda: _fake_embed)
    monkeypatch.setattr(vs, "_get_dimension_fn", lambda: (lambda: 8))
    # Boot gate off during this fixture init (we test the gate elsewhere).
    monkeypatch.setenv("KUTAI_CHROMA_SIZE_GATE", "off")

    assert await vs.init_store(persist_dir=db)
    for i in range(50):
        await vs.embed_and_store(f"row {i}", {"i": i}, "semantic")

    # Segment ids are readable and on-disk dirs are a subset (no false orphan).
    seg_ids = cm.list_segment_ids(db)
    assert seg_ids, "real store must expose segment ids"
    on_disk = cm.list_on_disk_segment_dirs(db)
    assert on_disk <= seg_ids
    # min_age 0 to remove the timing guard from the assertion.
    assert cm.find_orphan_segment_dirs(db, min_age_seconds=0) == []


# ─── boot-gate wiring into init_store ────────────────────────────────────────

@_chroma_skip
@pytest.mark.asyncio
async def test_init_store_invokes_size_gate_when_enabled(tmp_path, monkeypatch):
    from src.memory import vector_store as vs
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    for attr, val in (("_DB_DIR", db), ("_initialized", False), ("_client", None),
                      ("_collections", {}), ("_namespaced_collections", {})):
        monkeypatch.setattr(vs, attr, val, raising=False)
    monkeypatch.setattr(vs, "_get_embed_fn", lambda: (lambda t, is_query=False: [0.1] * 8))
    monkeypatch.setattr(vs, "_get_dimension_fn", lambda: (lambda: 8))
    monkeypatch.setenv("KUTAI_CHROMA_SIZE_GATE", "on")

    seen = {}

    def _spy(db_dir, budget_bytes=cm.DEFAULT_BUDGET_BYTES, **kw):
        seen["db_dir"] = db_dir
        seen["budget"] = budget_bytes
        return {"size_bytes": 1, "budget_bytes": budget_bytes,
                "over_budget": False, "action": "none"}

    monkeypatch.setattr(cm, "enforce_size_budget", _spy)

    assert await vs.init_store(persist_dir=db)
    assert seen["db_dir"] == db


@_chroma_skip
@pytest.mark.asyncio
async def test_init_store_gate_is_fail_open(tmp_path, monkeypatch):
    """A gate error must NEVER prevent the store from opening."""
    from src.memory import vector_store as vs
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    for attr, val in (("_DB_DIR", db), ("_initialized", False), ("_client", None),
                      ("_collections", {}), ("_namespaced_collections", {})):
        monkeypatch.setattr(vs, attr, val, raising=False)
    monkeypatch.setattr(vs, "_get_embed_fn", lambda: (lambda t, is_query=False: [0.1] * 8))
    monkeypatch.setattr(vs, "_get_dimension_fn", lambda: (lambda: 8))
    monkeypatch.setenv("KUTAI_CHROMA_SIZE_GATE", "on")

    def _boom(*a, **k):
        raise RuntimeError("gate blew up")

    monkeypatch.setattr(cm, "enforce_size_budget", _boom)

    assert await vs.init_store(persist_dir=db)  # opened despite gate error


@_chroma_skip
@pytest.mark.asyncio
async def test_init_store_gate_off_skips_enforcement(tmp_path, monkeypatch):
    from src.memory import vector_store as vs
    from src.memory import chroma_maintenance as cm

    db = str(tmp_path / "chroma")
    for attr, val in (("_DB_DIR", db), ("_initialized", False), ("_client", None),
                      ("_collections", {}), ("_namespaced_collections", {})):
        monkeypatch.setattr(vs, attr, val, raising=False)
    monkeypatch.setattr(vs, "_get_embed_fn", lambda: (lambda t, is_query=False: [0.1] * 8))
    monkeypatch.setattr(vs, "_get_dimension_fn", lambda: (lambda: 8))
    monkeypatch.setenv("KUTAI_CHROMA_SIZE_GATE", "off")

    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return {"over_budget": False, "action": "none"}

    monkeypatch.setattr(cm, "enforce_size_budget", _spy)

    assert await vs.init_store(persist_dir=db)
    assert called["n"] == 0  # gate disabled → never enforced
