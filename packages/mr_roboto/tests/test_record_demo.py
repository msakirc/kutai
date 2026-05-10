"""Z10 T4A — record_demo verb tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────────


def _make_workspace_with_spec(tmp_path: Path) -> Path:
    """Workspace with a Playwright spec + a test-results/<name>/video.webm."""
    ws = tmp_path / "workspace"
    (ws / "tests" / "e2e").mkdir(parents=True)
    (ws / "tests" / "e2e" / "golden_path.spec.ts").write_text(
        "test('x', () => {});", encoding="utf-8"
    )
    return ws


def _drop_webm(workspace: Path, name: str = "golden") -> Path:
    out = workspace / "test-results" / name / "video.webm"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"FAKE-WEBM-BYTES")
    return out


# ── Reversibility registry ───────────────────────────────────────────


def test_record_demo_in_reversibility_registry():
    from mr_roboto.reversibility import VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["record_demo"] == "full"
    assert VERB_REVERSIBILITY["verify_demo_artifact"] == "full"
    assert VERB_REVERSIBILITY["mission_deliverable_bundle"] == "irreversible"


# ── Mocked happy path ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_demo_happy_path(tmp_path, monkeypatch):
    """All subprocesses succeed; webm is found; ffmpeg writes mp4."""
    from mr_roboto import record_demo as rd

    ws = _make_workspace_with_spec(tmp_path)
    webm = _drop_webm(ws)

    # Point demo dir into tmp_path.
    monkeypatch.setattr(rd, "_project_root", lambda: str(tmp_path))

    # Mock all subprocess calls — succeed in order.
    calls: list[list[str]] = []

    async def _fake_run(cmd, timeout=300.0):
        calls.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            return 0, "true\n", ""
        if cmd[:2] == ["docker", "exec"]:
            return 0, "ok", ""
        if cmd[0] == "ffmpeg":
            # Produce the output file.
            out_path = cmd[-1]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # >1MB so verify_demo_artifact would pass; ffprobe mocked below.
            Path(out_path).write_bytes(b"X" * (2 * 1024 * 1024))
            return 0, "", ""
        return 0, "", ""

    monkeypatch.setattr(rd, "_run_subprocess", _fake_run)
    monkeypatch.setattr(rd, "_video_duration_seconds", lambda p: 12.5)

    res = await rd.run(
        mission_id=999,
        scenario_path="tests/e2e/golden_path.spec.ts",
        max_seconds=90,
        workspace_root=str(ws),
    )

    assert res["video_path"].endswith(os.path.join("999", "demo.mp4"))
    assert res["duration_s"] == 12.5
    assert isinstance(res["sha256"], str) and len(res["sha256"]) == 64
    assert res["source_webm"] == str(webm)

    # Container name uses mission_id; ffmpeg trims via -t.
    docker_exec = next(c for c in calls if c[:2] == ["docker", "exec"])
    assert "kutai-mission-999" in docker_exec
    ffmpeg = next(c for c in calls if c[0] == "ffmpeg")
    assert "-t" in ffmpeg and "90" in ffmpeg


@pytest.mark.asyncio
async def test_record_demo_no_webm_returns_failure(tmp_path, monkeypatch):
    """Playwright run succeeds but no .webm is produced → RuntimeError."""
    from mr_roboto import record_demo as rd

    ws = _make_workspace_with_spec(tmp_path)
    monkeypatch.setattr(rd, "_project_root", lambda: str(tmp_path))

    async def _fake_run(cmd, timeout=300.0):
        if cmd[:2] == ["docker", "inspect"]:
            return 0, "true\n", ""
        if cmd[:2] == ["docker", "exec"]:
            return 0, "", ""  # playwright "passes" but no video config
        return 0, "", ""

    monkeypatch.setattr(rd, "_run_subprocess", _fake_run)

    with pytest.raises(RuntimeError, match="no .webm produced"):
        await rd.run(
            mission_id=1000,
            scenario_path="tests/e2e/golden_path.spec.ts",
            workspace_root=str(ws),
        )


@pytest.mark.asyncio
async def test_record_demo_ffmpeg_failure(tmp_path, monkeypatch):
    """ffmpeg returns rc != 0 → RuntimeError surface."""
    from mr_roboto import record_demo as rd

    ws = _make_workspace_with_spec(tmp_path)
    _drop_webm(ws)
    monkeypatch.setattr(rd, "_project_root", lambda: str(tmp_path))

    async def _fake_run(cmd, timeout=300.0):
        if cmd[:2] == ["docker", "inspect"]:
            return 0, "true\n", ""
        if cmd[:2] == ["docker", "exec"]:
            return 0, "ok", ""
        if cmd[0] == "ffmpeg":
            return 2, "", "moov atom not found"
        return 0, "", ""

    monkeypatch.setattr(rd, "_run_subprocess", _fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg trim failed"):
        await rd.run(
            mission_id=1001,
            scenario_path="tests/e2e/golden_path.spec.ts",
            workspace_root=str(ws),
        )


@pytest.mark.asyncio
async def test_record_demo_container_not_running(tmp_path, monkeypatch):
    """docker inspect says container isn't running → fast-fail."""
    from mr_roboto import record_demo as rd

    ws = _make_workspace_with_spec(tmp_path)
    monkeypatch.setattr(rd, "_project_root", lambda: str(tmp_path))

    async def _fake_run(cmd, timeout=300.0):
        if cmd[:2] == ["docker", "inspect"]:
            return 1, "", "No such object: kutai-mission-1002"
        return 0, "", ""

    monkeypatch.setattr(rd, "_run_subprocess", _fake_run)

    with pytest.raises(RuntimeError, match="container .* not running"):
        await rd.run(
            mission_id=1002,
            scenario_path="tests/e2e/golden_path.spec.ts",
            workspace_root=str(ws),
        )


@pytest.mark.asyncio
async def test_record_demo_playwright_failure(tmp_path, monkeypatch):
    """Playwright run exits non-zero → RuntimeError with playwright hint."""
    from mr_roboto import record_demo as rd

    ws = _make_workspace_with_spec(tmp_path)
    monkeypatch.setattr(rd, "_project_root", lambda: str(tmp_path))

    async def _fake_run(cmd, timeout=300.0):
        if cmd[:2] == ["docker", "inspect"]:
            return 0, "true\n", ""
        if cmd[:2] == ["docker", "exec"]:
            return 1, "", "Error: playwright not found"
        return 0, "", ""

    monkeypatch.setattr(rd, "_run_subprocess", _fake_run)

    with pytest.raises(RuntimeError, match="playwright run failed"):
        await rd.run(
            mission_id=1003,
            scenario_path="tests/e2e/golden_path.spec.ts",
            workspace_root=str(ws),
        )
