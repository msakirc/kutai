"""Task 16 — failed image provider excluded across two pump cycles.

Exercises the whole v3 failed_models propagation fix end-to-end:

  CYCLE 1: HF_TOKEN set → image scorer ranks HF (8.0) above pollinations (6.0)
           → HF picked first → flaky HF fails → on_task_finished appends
           the failed model to context.failed_models and re-pends (availability).
  CYCLE 2: failed_models forwarded into _select_for_admission → HF excluded
           → pollinations picked → success.

If cycle 2 re-picks HF, _select_for_admission isn't forwarding failed_models.
If cycle 2 next_task() returns None, the re-pend / backoff clearing is broken.
"""
import json

import pytest

import src.infra.db as _db_mod
from src.infra.db import init_db, get_task, get_db, update_task
import general_beckman  # noqa: F401
from general_beckman import enqueue, next_task, on_task_finished
import husam
from src.core.router import ModelCallFailed


@pytest.fixture(autouse=True)
async def _close_db_conn():
    """Close the aiosqlite connection after each test. Its non-daemon background
    thread otherwise blocks interpreter exit and hangs pytest teardown for hours."""
    yield
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def _fresh_db(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation (copy of
    tests/test_beckman_next_task.py::_fresh)."""
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    import general_beckman.cron_seed as cs_mod
    monkeypatch.setattr(cs_mod, "_seeded", False)
    from general_beckman import paused_patterns as _pp
    _pp._patterns.clear()
    await init_db()
    conn = await get_db()
    await conn.execute(
        "UPDATE scheduled_tasks SET next_run = datetime('now', '+1 hour')"
    )
    await conn.commit()


@pytest.mark.asyncio
async def test_failed_provider_excluded_on_retry(monkeypatch, tmp_path):
    await _fresh_db(tmp_path, monkeypatch)

    seen = {"hf": 0}

    class _FlakyHF:
        name = "huggingface"

        def available(self):
            return True

        async def generate(self, spec, *, base_url=None):
            seen["hf"] += 1
            raise RuntimeError("simulated provider down")

    class _FakePollinations:
        name = "pollinations"

        def available(self):
            return True

        async def generate(self, spec, *, base_url=None):
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (200, 100, 100)).save(buf, "PNG")
            return buf.getvalue(), {"seed_used": 2}

    import paintress
    monkeypatch.setattr(paintress, "_PROVIDERS",
                        {"pollinations": _FakePollinations(), "huggingface": _FlakyHF()})
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")

    spec = {
        "title": "retry-test", "description": "retry image",
        "agent_type": "image", "kind": "image", "priority": 5,
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "retry test",
            "out_dir": str(tmp_path), "width": 64, "height": 64,
            "filename_hint": "ret", "quality_tier": "quality",
        }},
    }
    tid = await enqueue(spec)

    # CYCLE 1: HF picked (top rank) → fails
    task1 = await next_task()
    assert task1["id"] == tid
    picked1 = task1["preselected_pick"].model.name
    assert picked1 == "huggingface/flux-schnell"
    with pytest.raises(ModelCallFailed):
        await husam.run(task1)
    await on_task_finished(tid, {"status": "failed", "error": "simulated provider down",
                                 "error_category": "availability", "model": picked1})
    row1 = await get_task(tid)
    assert row1["status"] == "pending"
    ctx1 = json.loads(row1.get("context") or "{}")
    assert picked1 in ctx1.get("failed_models", [])
    await update_task(tid, next_retry_at=None)  # clear backoff so cycle 2 admits now

    # CYCLE 2: HF excluded → pollinations → success
    task2 = await next_task()
    assert task2 is not None and task2["id"] == tid
    assert task2["preselected_pick"].model.name == "pollinations/flux"
    result = await husam.run(task2)
    assert result["provider"] == "pollinations"
    assert seen["hf"] >= 1
    await on_task_finished(tid, {"status": "completed",
                                 "result": json.dumps(result), **result})
    assert (await get_task(tid))["status"] == "completed"
