"""Disk resolver: target_artifact filename -> content under mission workspace.

`target_artifact` in reviewer findings is a bare filename
(`competitive_positioning.md`) but the file lives in a dotted subdir
(`.charter/`, `.prd/`, `.research/`, `.intake/`). The resolver does a bounded
basename match under the mission workspace (disk = canonical).
"""
from __future__ import annotations

import asyncio
import os

import mr_roboto.verify_review_verdict as vv


def _patch_ws(monkeypatch, root):
    import src.tools.workspace as ws_mod
    monkeypatch.setattr(ws_mod, "get_mission_workspace", lambda mid: root)


def test_resolves_file_in_dotted_subdir(tmp_path, monkeypatch):
    charter = tmp_path / ".charter"
    charter.mkdir()
    (charter / "product_charter.md").write_text("# Charter\nbody", encoding="utf-8")
    _patch_ws(monkeypatch, str(tmp_path))

    content = asyncio.run(vv._resolve_artifact_content(90, "product_charter.md"))
    assert content is not None
    assert "# Charter" in content


def test_resolves_root_level_file(tmp_path, monkeypatch):
    (tmp_path / "market_research_report.md").write_text("TAM data", encoding="utf-8")
    _patch_ws(monkeypatch, str(tmp_path))
    content = asyncio.run(vv._resolve_artifact_content(90, "market_research_report.md"))
    assert content == "TAM data"


def test_missing_file_returns_none(tmp_path, monkeypatch):
    _patch_ws(monkeypatch, str(tmp_path))
    content = asyncio.run(vv._resolve_artifact_content(90, "nope.md"))
    assert content is None


def test_none_target_returns_none(tmp_path, monkeypatch):
    _patch_ws(monkeypatch, str(tmp_path))
    assert asyncio.run(vv._resolve_artifact_content(90, None)) is None


def test_does_not_descend_into_code_dirs(tmp_path, monkeypatch):
    """A same-named file under a non-dot (code) dir must NOT be returned —
    the walk is pruned to the workspace root + dotted artifact dirs."""
    app = tmp_path / "backend" / "src"
    app.mkdir(parents=True)
    (app / "report.md").write_text("CODE DIR COPY", encoding="utf-8")
    research = tmp_path / ".research"
    research.mkdir()
    (research / "report.md").write_text("REAL ARTIFACT", encoding="utf-8")
    _patch_ws(monkeypatch, str(tmp_path))
    content = asyncio.run(vv._resolve_artifact_content(90, "report.md"))
    assert content == "REAL ARTIFACT"
