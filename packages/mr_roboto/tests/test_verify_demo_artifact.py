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


# ── z10-wire-fixes F3 — non-video artifact acceptance ───────────────────


def test_verify_demo_artifact_accepts_gif(tmp_path, monkeypatch):
    """F3: a recorded .gif demo passes the gate (no duration check)."""
    from mr_roboto import verify_demo_artifact as vd

    gif = tmp_path / "data" / "missions" / "1100" / "demo.gif"
    gif.parent.mkdir(parents=True, exist_ok=True)
    gif.write_bytes(b"GIF89a" + b"\x00" * (2 * 1024 * 1024))
    monkeypatch.setattr(vd, "_project_root", lambda: str(tmp_path))

    res = vd.run(mission_id=1100)
    assert res["ok"] is True, res
    assert res["kind"] == "gif"
    assert "duration_s" not in res  # no ffprobe on gif


def test_verify_demo_artifact_accepts_asciinema_cast(tmp_path, monkeypatch):
    """F3: a recorded asciinema .cast demo passes — small size threshold."""
    from mr_roboto import verify_demo_artifact as vd

    cast = tmp_path / "data" / "missions" / "1200" / "demo.cast"
    cast.parent.mkdir(parents=True, exist_ok=True)
    # asciinema casts are small (typically <100 KB).
    cast.write_text('{"version": 2}\n[0.1, "o", "hello"]\n', encoding="utf-8")
    cast.write_bytes(cast.read_bytes() + b"x" * 2048)  # bump above 1 KB floor
    monkeypatch.setattr(vd, "_project_root", lambda: str(tmp_path))

    res = vd.run(mission_id=1200)
    assert res["ok"] is True, res
    assert res["kind"] == "cast"


def test_verify_demo_artifact_prefers_mp4_when_multiple_exist(tmp_path, monkeypatch):
    """F3: mp4 still wins when both mp4 and gif present (web is canonical)."""
    from mr_roboto import verify_demo_artifact as vd

    base = tmp_path / "data" / "missions" / "1300"
    base.mkdir(parents=True, exist_ok=True)
    (base / "demo.mp4").write_bytes(b"X" * (2 * 1024 * 1024))
    (base / "demo.gif").write_bytes(b"GIF89a" + b"\x00" * (2 * 1024 * 1024))
    monkeypatch.setattr(vd, "_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(vd, "_ffprobe_duration", lambda p: 10.0)

    res = vd.run(mission_id=1300)
    assert res["ok"] is True
    assert res["kind"] == "video"
    assert res["path"].endswith("demo.mp4")
