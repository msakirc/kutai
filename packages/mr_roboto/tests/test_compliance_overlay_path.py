"""compliance_template_present must resolve a relative overlay_path against
the mission WORKSPACE_DIR, not the process CWD.

Bug (2026-05-26): the post-hook payload passes overlay_path='mission_77/
compliance_overlay.json' (relative to the workspace). The executor opened it
against CWD (project root) → FileNotFoundError → ok=False → the mechanical
gate DLQ'd (task #166560) with a misleading 'missing=[] checked=[]' message.
find_similar_missions already resolves 'mission_<id>' via WORKSPACE_DIR; this
gate must too.
"""
from __future__ import annotations

import json

# NB: mr_roboto/__init__ re-exports the function `compliance_template_present`,
# shadowing the submodule of the same name in the package namespace — import
# the function from the submodule directly.
from mr_roboto.compliance_template_present import compliance_template_present


def test_relative_overlay_path_resolves_against_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path),
                        raising=False)
    msn = tmp_path / "mission_x"
    msn.mkdir()
    (msn / "compliance_overlay.json").write_text(
        json.dumps({"required_documents": []}), encoding="utf-8",
    )

    res = compliance_template_present(
        overlay_path="mission_x/compliance_overlay.json",
    )

    # Read must succeed (no 'could not read overlay'); empty required_documents
    # → nothing to check → ok=True.
    assert res["ok"] is True, res
    assert not res.get("error"), res


def test_absolute_overlay_path_still_read(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path / "ws"),
                        raising=False)
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"required_documents": []}), encoding="utf-8")

    res = compliance_template_present(overlay_path=str(overlay))
    assert res["ok"] is True, res
