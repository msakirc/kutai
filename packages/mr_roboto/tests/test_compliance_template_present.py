"""Tests for compliance_template_present — Z1 T5A (P6) post-hook on 1.11a."""
from __future__ import annotations

import json
import os

import pytest

from mr_roboto.compliance_template_present import compliance_template_present


@pytest.fixture
def template_root(tmp_path):
    root = tmp_path / "compliance_templates"
    (root / "default" / "en").mkdir(parents=True)
    # privacy_policy template present
    (root / "default" / "en" / "privacy_policy.md.j2").write_text(
        "# Privacy Policy template\n", encoding="utf-8"
    )
    # gdpr_hipaa_privacy_v3 with custom name (matches plan-v3 example).
    (root / "default" / "en" / "gdpr_hipaa_privacy_v3.md.j2").write_text(
        "# GDPR/HIPAA template\n", encoding="utf-8"
    )
    return str(root)


def test_returns_ok_when_no_ids_to_check(template_root):
    res = compliance_template_present(template_ids=[], template_root=template_root)
    assert res["ok"] is True
    assert res["missing"] == []
    assert res["checked"] == []


def test_returns_ok_when_template_present(template_root):
    res = compliance_template_present(
        template_ids=["privacy_policy"], template_root=template_root,
    )
    assert res["ok"] is True
    assert res["missing"] == []
    assert "privacy_policy" in res["checked"]


def test_fails_when_template_missing(template_root):
    res = compliance_template_present(
        template_ids=["nonexistent_doc"], template_root=template_root,
    )
    assert res["ok"] is False
    assert "nonexistent_doc" in res["missing"]


def test_extracts_ids_from_overlay_obj(template_root):
    overlay = {
        "required_documents": [
            {"doc_type": "privacy_policy", "template_id": "privacy_policy"},
            {"doc_type": "dpa", "template_id": "gdpr_hipaa_privacy_v3"},
        ],
    }
    res = compliance_template_present(
        overlay_obj=overlay, template_root=template_root,
    )
    assert res["ok"] is True
    assert set(res["checked"]) == {"privacy_policy", "gdpr_hipaa_privacy_v3"}


def test_extracts_ids_from_overlay_path(template_root, tmp_path):
    overlay_path = tmp_path / "overlay.json"
    overlay = {
        "required_documents": [
            {"doc_type": "privacy_policy", "template_id": "privacy_policy"},
            {"doc_type": "audit", "template_id": "missing_template"},
        ],
    }
    overlay_path.write_text(json.dumps(overlay), encoding="utf-8")
    res = compliance_template_present(
        overlay_path=str(overlay_path), template_root=template_root,
    )
    assert res["ok"] is False
    assert res["missing"] == ["missing_template"]


def test_finds_template_recursively(tmp_path):
    """A template nested under jurisdiction/lang/ counts."""
    root = tmp_path / "compliance_templates"
    (root / "EU" / "en").mkdir(parents=True)
    (root / "EU" / "en" / "gdpr_basic.md.j2").write_text("EU template", encoding="utf-8")
    res = compliance_template_present(
        template_ids=["gdpr_basic"], template_root=str(root),
    )
    assert res["ok"] is True


def test_missing_root_means_all_templates_missing(tmp_path):
    nonexistent = str(tmp_path / "no_such_dir")
    res = compliance_template_present(
        template_ids=["privacy_policy"], template_root=nonexistent,
    )
    assert res["ok"] is False
    assert "privacy_policy" in res["missing"]


@pytest.mark.asyncio
async def test_dispatch_through_run(template_root):
    """The mr_roboto.run dispatcher routes the action correctly."""
    from mr_roboto import run, Action
    task = {
        "id": 1,
        "mission_id": 1,
        "payload": {
            "action": "compliance_template_present",
            "template_ids": ["privacy_policy"],
            "template_root": template_root,
        },
    }
    result = await run(task)
    assert isinstance(result, Action)
    assert result.status == "completed"
    assert result.result["ok"] is True


@pytest.mark.asyncio
async def test_dispatch_fails_when_template_missing(template_root):
    from mr_roboto import run
    task = {
        "id": 1,
        "mission_id": 1,
        "payload": {
            "action": "compliance_template_present",
            "template_ids": ["does_not_exist"],
            "template_root": template_root,
        },
    }
    result = await run(task)
    assert result.status == "failed"
    assert "does_not_exist" in (result.result.get("missing") or [])
