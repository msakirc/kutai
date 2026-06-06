"""Task 14 — end-to-end image pipeline through the public Beckman API.

Drives the WHOLE image lane via manual pump cycles (no await_inline):

    beckman.enqueue(spec)        — write an image task
    next_task()                  — admit it, attach preselected_pick (image scorer)
    husam.run(task)              — paintress.generate via a fake provider
    on_task_finished(tid, result) — terminal write + result persisted

Isolated via the _fresh_db tmp-DB fixture (copied from
tests/test_beckman_next_task.py, including cron-seed + paused_patterns
resets) so it NEVER touches the live kutai.db.
"""
import json
import os

import pytest

import src.infra.db as _db_mod
from src.infra.db import init_db, get_task, get_db
import general_beckman  # noqa: F401  (ensures package import side-effects)
from general_beckman import enqueue, next_task, on_task_finished
import husam


@pytest.fixture(autouse=True)
async def _close_db_conn():
    """Close the aiosqlite connection after each test. Its non-daemon background
    thread otherwise blocks interpreter exit and hangs pytest teardown for hours."""
    yield
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def _fresh_db(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation.

    Faithful copy of tests/test_beckman_next_task.py::_fresh — resets the
    db.py singleton connection, the cron_seed `_seeded` flag, and the
    paused_patterns module state, then pushes all seeded scheduled_tasks
    next_run into the future so cron doesn't fire on first next_task().
    """
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    # Reset cron_seed so each test gets a clean seeder state.
    import general_beckman.cron_seed as cs_mod
    monkeypatch.setattr(cs_mod, "_seeded", False)
    # Reset paused_patterns module state.
    from general_beckman import paused_patterns as _pp
    _pp._patterns.clear()
    await init_db()
    # Push all seeded scheduled_tasks next_run into the future so cron doesn't
    # fire on first next_task() call and insert unexpected tasks.
    conn = await get_db()
    await conn.execute(
        "UPDATE scheduled_tasks SET next_run = datetime('now', '+1 hour')"
    )
    await conn.commit()


def _fake_pollinations(seed=11, color=(100, 150, 200)):
    class _FakeProvider:
        name = "pollinations"

        def available(self):
            return True

        async def generate(self, spec, *, base_url=None):
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), color).save(buf, "PNG")
            return buf.getvalue(), {"seed_used": seed}

    return _FakeProvider()


@pytest.mark.asyncio
async def test_image_generation_full_pipeline(monkeypatch, tmp_path):
    await _fresh_db(tmp_path, monkeypatch)
    import paintress
    monkeypatch.setattr(paintress, "_PROVIDERS", {"pollinations": _fake_pollinations()})
    monkeypatch.delenv("HF_TOKEN", raising=False)

    spec = {
        "title": "e2e", "description": "e2e image",
        "agent_type": "image", "kind": "image", "priority": 5,
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a mountain lake",
            "out_dir": str(tmp_path), "width": 64, "height": 64,
            "filename_hint": "lake",
        }},
    }
    tid = await enqueue(spec)
    assert isinstance(tid, int)

    task = await next_task()
    assert task is not None and task["id"] == tid
    # Admission must attach the image pick (pollinations rank 6.0; HF excluded
    # without HF_TOKEN). This exercises the admission↔ImageModelInfo seam.
    pick = task["preselected_pick"]
    assert pick.model.name == "pollinations/flux"

    result = await husam.run(task)
    assert result["provider"] == "pollinations"
    assert result["seed_used"] == 11
    assert os.path.isfile(result["path"]) and os.path.getsize(result["path"]) > 0

    await on_task_finished(tid, {"status": "completed",
                                 "result": json.dumps(result), **result})
    row = await get_task(tid)
    assert row["status"] == "completed"
    raw = row.get("result")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["path"] == result["path"]
