"""Z10 T4A — mission_deliverable_bundle verb tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


class _FakeMessage:
    def __init__(self, message_id: int):
        self.message_id = message_id


class _FakeBot:
    def __init__(self):
        self.sent_videos = []
        self.sent_texts = []

    async def send_video(self, **kwargs):
        self.sent_videos.append(kwargs)
        return _FakeMessage(1001)

    async def send_message(self, **kwargs):
        self.sent_texts.append(kwargs)
        return _FakeMessage(1002)


async def _init_db(tmp_path, monkeypatch):
    db_path = tmp_path / "bundle.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


def _write_fake_mp4(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"X" * 2048)


@pytest.mark.asyncio
async def test_deliverable_bundle_thread_path(tmp_path, monkeypatch):
    """When mission has telegram_thread_id, video uses message_thread_id."""
    db_mod = await _init_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status, telegram_thread_id) "
        "VALUES (?, ?, ?, ?)",
        (501, "demo mission", "active", 42),
    )
    await db.commit()

    # Provenance row so the bundle has top-N data.
    await db_mod.record_artifact_write(
        path="src/main.py", task_id=None, step_id="1.1",
        model_id="gpt-4o", mission_id=501,
    )

    # Stub workspace for video + commit lookup
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mp4 = workspace / "demo.mp4"
    _write_fake_mp4(mp4)
    monkeypatch.setattr(
        "mr_roboto.mission_deliverable_bundle._final_commit_sha",
        lambda p: "abcdef1234567890",
    )

    # Avoid calling format_mission_cost full machinery — short-circuit it.
    async def _fake_cost(mid):
        return f"Mission {mid} — cost\nTotal: $0.42"
    monkeypatch.setattr(
        "src.infra.cost_wiring.format_mission_cost", _fake_cost
    )

    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_ID", "12345")
    # Reload config so TELEGRAM_ADMIN_CHAT_ID re-reads env (cached at import).
    import importlib
    from src.app import config as _cfg
    importlib.reload(_cfg)

    from mr_roboto import mission_deliverable_bundle as bundle_mod

    bot = _FakeBot()
    res = await bundle_mod.run(
        bot=bot, mission_id=501, video_path=str(mp4),
        repo_path=str(workspace),
    )

    assert res["ok"] is True
    assert res["commit_sha"] == "abcdef1234567890"
    assert res["video_sent"] is True
    assert "src/main.py" in res["text"]
    assert "abcdef123456" in res["text"]
    assert "$0.42" in res["text"]

    # send_video used message_thread_id (thread path).
    assert bot.sent_videos
    assert bot.sent_videos[0]["message_thread_id"] == 42
    # send_message also used message_thread_id (post_to_mission_thread).
    assert bot.sent_texts
    assert bot.sent_texts[0].get("message_thread_id") == 42


@pytest.mark.asyncio
async def test_deliverable_bundle_flat_fallback(tmp_path, monkeypatch):
    """No telegram_thread_id → flat post with [Mission {id}] prefix."""
    db_mod = await _init_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status, telegram_thread_id) "
        "VALUES (?, ?, ?, ?)",
        (502, "demo mission 2", "active", None),
    )
    await db.commit()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mp4 = workspace / "demo.mp4"
    _write_fake_mp4(mp4)
    monkeypatch.setattr(
        "mr_roboto.mission_deliverable_bundle._final_commit_sha",
        lambda p: "feedface00000000",
    )

    async def _fake_cost(mid):
        return f"Mission {mid} — cost\nTotal: $0.10"
    monkeypatch.setattr(
        "src.infra.cost_wiring.format_mission_cost", _fake_cost
    )

    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_ID", "12345")
    import importlib
    from src.app import config as _cfg
    importlib.reload(_cfg)

    from mr_roboto import mission_deliverable_bundle as bundle_mod

    bot = _FakeBot()
    res = await bundle_mod.run(
        bot=bot, mission_id=502, video_path=str(mp4),
        repo_path=str(workspace),
    )

    assert res["ok"] is True
    # Flat path: no message_thread_id on send_video or send_message.
    assert bot.sent_videos
    assert "message_thread_id" not in bot.sent_videos[0]
    assert bot.sent_texts
    assert "message_thread_id" not in bot.sent_texts[0]
    # Flat send_message text was prefixed.
    assert bot.sent_texts[0]["text"].startswith("[Mission 502]")
