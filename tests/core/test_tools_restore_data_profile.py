"""Regression: data Profile tool-restore must not corrupt the singleton.

A data ``Profile`` declares ``_original_allowed_tools`` as a dataclass field
(always present, default None). The OLD execute() finally guarded restoration on
``hasattr(profile, '_original_allowed_tools')`` — always True for a data Profile —
so on dispatch #1 it set ``allowed_tools = None`` (the field default), silently
wiping the tool list of the shared singleton (None = all-tools).

The fix gates restoration on a dedicated ``_tools_overridden`` sentinel set only
when a setup step actually snapshots. This test dispatches a data Profile twice
through ``coulson.execute`` (with the react seam mocked) and asserts
``allowed_tools`` stays ``["read_file"]`` after BOTH dispatches.
"""
import asyncio

from finch.profile import Profile


def test_data_profile_allowed_tools_survive_two_dispatches(monkeypatch):
    import coulson

    p = Profile(
        name="dataprof",
        description="d",
        system_prompt="SEED",
        allowed_tools=["read_file"],
        max_iterations=3,
    )

    async def fake_react(profile, task, *a, **k):
        # The worker would normally consume profile.allowed_tools here.
        return {"status": "complete", "result": "ok", "used_model": "x",
                "iterations": 1, "cost": 0.0}

    monkeypatch.setattr(coulson, "_react_run", fake_react)

    task = {"id": 1, "title": "t", "description": "summarize x",
            "context": {}, "agent_type": "dataprof"}

    # Dispatch #1
    out1 = asyncio.run(coulson.execute(p, dict(task)))
    assert isinstance(out1, dict)
    assert p.allowed_tools == ["read_file"], "tool list wiped after dispatch #1"
    assert p._original_allowed_tools is None
    assert p._tools_overridden is False

    # Dispatch #2 — singleton must remain uncorrupted
    out2 = asyncio.run(coulson.execute(p, dict(task)))
    assert isinstance(out2, dict)
    assert p.allowed_tools == ["read_file"], "tool list wiped after dispatch #2"
    assert p._original_allowed_tools is None
    assert p._tools_overridden is False


def test_data_profile_auto_strip_restores_after_dispatch(monkeypatch):
    """With an artifact_schema (auto-strip active), write tools are stripped
    during the call but allowed_tools is restored to the original after."""
    import coulson

    p = Profile(
        name="dataprof2",
        description="d",
        system_prompt="SEED",
        allowed_tools=["read_file", "write_file"],
        max_iterations=3,
    )

    seen = {}

    async def fake_react(profile, task, *a, **k):
        # During the call, write_file should have been stripped.
        seen["during"] = list(profile.allowed_tools)
        return {"status": "complete", "result": "ok", "used_model": "x",
                "iterations": 1, "cost": 0.0}

    monkeypatch.setattr(coulson, "_react_run", fake_react)

    task = {"id": 2, "title": "t", "description": "d",
            "context": {"artifact_schema": {"type": "object"}},
            "agent_type": "dataprof2"}

    asyncio.run(coulson.execute(p, task))

    assert seen["during"] == ["read_file"], "write_file not stripped during call"
    # Restored to the original after the call (sentinel-guarded).
    assert p.allowed_tools == ["read_file", "write_file"]
    assert p._tools_overridden is False
