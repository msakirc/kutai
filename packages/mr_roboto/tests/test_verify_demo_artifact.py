"""Z10 T4A — verify_demo_artifact verb tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _write_fake_mp4(path: Path, size: int = 2 * 1024 * 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"X" * size)


def test_verify_demo_artifact_ok(tmp_path, monkeypatch):
    from mr_roboto import verify_demo_artifact as vd

    mp4 = tmp_path / "data" / "missions" / "555" / "demo.mp4"
    _write_fake_mp4(mp4)
    monkeypatch.setattr(vd, "_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(vd, "_ffprobe_duration", lambda p: 10.0)

    res = vd.run(mission_id=555)
    assert res["ok"] is True
    assert res["mime"] == "video/mp4"
    assert res["duration_s"] == 10.0


def test_verify_demo_artifact_missing(tmp_path, monkeypatch):
    from mr_roboto import verify_demo_artifact as vd

    monkeypatch.setattr(vd, "_project_root", lambda: str(tmp_path))
    res = vd.run(mission_id=666)
    assert res["ok"] is False
    assert "missing" in res["reason"]


def test_verify_demo_artifact_too_small(tmp_path, monkeypatch):
    from mr_roboto import verify_demo_artifact as vd

    mp4 = tmp_path / "data" / "missions" / "777" / "demo.mp4"
    _write_fake_mp4(mp4, size=100)
    monkeypatch.setattr(vd, "_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(vd, "_ffprobe_duration", lambda p: 10.0)

    res = vd.run(mission_id=777)
    assert res["ok"] is False
    assert "too small" in res["reason"]


def test_verify_demo_artifact_too_short(tmp_path, monkeypatch):
    from mr_roboto import verify_demo_artifact as vd

    mp4 = tmp_path / "data" / "missions" / "888" / "demo.mp4"
    _write_fake_mp4(mp4)
    monkeypatch.setattr(vd, "_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(vd, "_ffprobe_duration", lambda p: 1.0)

    res = vd.run(mission_id=888)
    assert res["ok"] is False
    assert "too short" in res["reason"]


def test_verify_demo_artifact_explicit_path(tmp_path, monkeypatch):
    """video_path arg overrides default mission path."""
    from mr_roboto import verify_demo_artifact as vd

    mp4 = tmp_path / "elsewhere" / "demo.mp4"
    _write_fake_mp4(mp4)
    monkeypatch.setattr(vd, "_ffprobe_duration", lambda p: 30.0)

    res = vd.run(mission_id=999, video_path=str(mp4))
    assert res["ok"] is True
    assert res["path"] == str(mp4)
