"""Tests for compliance_blocker_check — Z1 T5A (P6) phase-6 terminal post-hook."""
from __future__ import annotations

import json
import os

import pytest

from mr_roboto.compliance_blocker_check import compliance_blocker_check


def test_no_overlay_passes_with_warning_marker(tmp_path):
    """Per spec: no overlay = pass with overlay_present=False (reviewer warns)."""
    res = compliance_blocker_check(
        mission_id=42, current_phase=6, workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["overlay_present"] is False
    assert res["pending"] == []


def test_overlay_with_no_blockers_passes(tmp_path):
    overlay = {
        "_schema_version": "1",
        "required_documents": [
            {
                "doc_type": "privacy_policy",
                "template_id": "privacy_policy",
                "blocker_for_phase": 13,
                "generated_template_path": "",
            },
        ],
    }
    (tmp_path / "compliance_overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8",
    )
    res = compliance_blocker_check(
        mission_id=42, current_phase=6, workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["overlay_present"] is True
    assert res["pending"] == []


def test_overlay_with_pending_blocker_fails(tmp_path):
    overlay = {
        "required_documents": [
            {
                "doc_type": "data_processor_compatibility",
                "template_id": "hipaa_processor_audit_v1",
                "blocker_for_phase": 4,
                "generated_template_path": "/nonexistent/path/audit.md",
            },
        ],
    }
    (tmp_path / "compliance_overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8",
    )
    res = compliance_blocker_check(
        mission_id=42, current_phase=6, workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert len(res["pending"]) == 1
    assert res["pending"][0]["doc_type"] == "data_processor_compatibility"
    assert res["pending"][0]["blocker_for_phase"] == 4


def test_overlay_with_rendered_doc_passes(tmp_path):
    rendered = tmp_path / "audit.md"
    rendered.write_text("# Rendered audit\n", encoding="utf-8")
    overlay = {
        "required_documents": [
            {
                "doc_type": "data_processor_compatibility",
                "template_id": "hipaa_processor_audit_v1",
                "blocker_for_phase": 4,
                "generated_template_path": str(rendered),
            },
        ],
    }
    (tmp_path / "compliance_overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8",
    )
    res = compliance_blocker_check(
        mission_id=42, current_phase=6, workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["pending"] == []


def test_blocker_above_current_phase_skipped(tmp_path):
    """Blocker for phase 13 doesn't fire when current_phase=6."""
    overlay = {
        "required_documents": [
            {
                "doc_type": "privacy_policy",
                "template_id": "privacy_policy",
                "blocker_for_phase": 13,
                "generated_template_path": "/nope/privacy.md",
            },
        ],
    }
    (tmp_path / "compliance_overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8",
    )
    res = compliance_blocker_check(
        mission_id=42, current_phase=6, workspace_path=str(tmp_path),
    )
    assert res["ok"] is True


def test_alt_overlay_path_compliance_subdir(tmp_path):
    """Overlay can also live at .compliance/overlay.json."""
    sub = tmp_path / ".compliance"
    sub.mkdir()
    (sub / "overlay.json").write_text(
        json.dumps({"required_documents": []}), encoding="utf-8",
    )
    res = compliance_blocker_check(
        mission_id=42, current_phase=6, workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["overlay_present"] is True


def test_founder_review_required_no_signoff_fails(tmp_path):
    """File rendered but founder_review_required=true and no signoff ⇒ blocked."""
    rendered = tmp_path / "privacy.md"
    rendered.write_text("# Privacy\n", encoding="utf-8")
    overlay = {
        "required_documents": [
            {
                "doc_type": "privacy_policy",
                "template_id": "privacy_policy",
                "blocker_for_phase": 4,
                "generated_template_path": str(rendered),
                "founder_review_required": True,
            },
        ],
    }
    (tmp_path / "compliance_overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8",
    )
    res = compliance_blocker_check(
        mission_id=42, current_phase=6, workspace_path=str(tmp_path),
        signoffs=set(),  # explicit: nothing signed yet
    )
    assert res["ok"] is False
    assert res["pending"][0]["doc_type"] == "privacy_policy"
    assert res["pending"][0]["reason"] == "founder_signoff_missing"


def test_founder_review_required_with_signoff_passes(tmp_path):
    """File rendered + signoff in set ⇒ pass."""
    rendered = tmp_path / "privacy.md"
    rendered.write_text("# Privacy\n", encoding="utf-8")
    overlay = {
        "required_documents": [
            {
                "doc_type": "privacy_policy",
                "template_id": "privacy_policy",
                "blocker_for_phase": 4,
                "generated_template_path": str(rendered),
                "founder_review_required": True,
            },
        ],
    }
    (tmp_path / "compliance_overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8",
    )
    res = compliance_blocker_check(
        mission_id=42, current_phase=6, workspace_path=str(tmp_path),
        signoffs={"privacy_policy"},
    )
    assert res["ok"] is True
    assert res["pending"] == []


def test_no_signoff_required_when_flag_absent(tmp_path):
    """founder_review_required omitted ⇒ signoff not required (legacy path)."""
    rendered = tmp_path / "tos.md"
    rendered.write_text("# TOS\n", encoding="utf-8")
    overlay = {
        "required_documents": [
            {
                "doc_type": "tos",
                "blocker_for_phase": 4,
                "generated_template_path": str(rendered),
                # no founder_review_required key
            },
        ],
    }
    (tmp_path / "compliance_overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8",
    )
    res = compliance_blocker_check(
        mission_id=42, current_phase=6, workspace_path=str(tmp_path),
        signoffs=set(),
    )
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_dispatch_through_run_passes(tmp_path):
    from mr_roboto import run
    overlay = {"required_documents": []}
    (tmp_path / "compliance_overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8",
    )
    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "compliance_blocker_check",
            "current_phase": 6,
            "workspace_path": str(tmp_path),
        },
    }
    result = await run(task)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_dispatch_through_run_fails_with_pending(tmp_path):
    from mr_roboto import run
    overlay = {
        "required_documents": [
            {
                "doc_type": "x",
                "template_id": "x",
                "blocker_for_phase": 4,
                "generated_template_path": "/nope.md",
            },
        ],
    }
    (tmp_path / "compliance_overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8",
    )
    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "compliance_blocker_check",
            "current_phase": 6,
            "workspace_path": str(tmp_path),
        },
    }
    result = await run(task)
    assert result.status == "failed"
