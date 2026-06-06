"""Unit tests for src.telemetry.pick_recorder.record_pick (Phase 4 eviction)."""

import types

import pytest

from src.telemetry import pick_recorder


class _Model:
    def __init__(self, name="qwen", provider="local", is_local=True):
        self.name = name
        self.provider = provider
        self.is_local = is_local


class _Pick:
    def __init__(self, model, score=12.5, top_summary="a>b>c"):
        self.model = model
        self.score = score
        self.top_summary = top_summary


@pytest.mark.asyncio
async def test_record_pick_maps_args(monkeypatch):
    captured = {}

    async def _fake_write(**kw):
        captured.update(kw)

    monkeypatch.setattr(pick_recorder, "_PLACEHOLDER", None, raising=False)
    import src.infra.pick_log as pl
    monkeypatch.setattr(pl, "write_pick_log_row", _fake_write)

    cat = types.SimpleNamespace(value="main_work")
    await pick_recorder.record_pick(
        pick=_Pick(_Model()),
        task="coder",
        category=cat,
        success=True,
        agent_type="coder",
        difficulty=6,
    )

    assert captured["picked_model"] == "qwen"
    assert captured["picked_score"] == 12.5
    assert captured["category"] == "main_work"
    assert captured["task_name"] == "coder"
    assert captured["success"] is True
    assert captured["provider"] == "local"
    assert captured["agent_type"] == "coder"
    assert captured["difficulty"] == 6
    assert captured["snapshot_summary"] == "a>b>c"


@pytest.mark.asyncio
async def test_record_pick_cloud_provider(monkeypatch):
    captured = {}

    async def _fake_write(**kw):
        captured.update(kw)

    import src.infra.pick_log as pl
    monkeypatch.setattr(pl, "write_pick_log_row", _fake_write)

    await pick_recorder.record_pick(
        pick=_Pick(_Model(name="gemini", provider="google", is_local=False)),
        task="",
        category="overhead",  # raw string accepted
        success=False,
        error_category="rate_limited",
    )

    assert captured["picked_model"] == "gemini"
    assert captured["provider"] == "google"
    assert captured["category"] == "overhead"
    assert captured["task_name"] == "overhead"  # falls back to category
    assert captured["error_category"] == "rate_limited"


@pytest.mark.asyncio
async def test_record_pick_swallows_errors(monkeypatch):
    async def _boom(**kw):
        raise RuntimeError("db gone")

    import src.infra.pick_log as pl
    monkeypatch.setattr(pl, "write_pick_log_row", _boom)

    # Must not raise.
    await pick_recorder.record_pick(
        pick=_Pick(_Model()),
        task="x",
        category="main_work",
        success=True,
    )
