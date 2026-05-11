"""Z6 T4A — tests for mr_roboto.executors.legal_document_render."""
from __future__ import annotations

import json
import os

import pytest

from mr_roboto.executors.legal_document_render import legal_document_render


@pytest.fixture
def overlay_with_three_docs():
    return {
        "jurisdictions": [],
        "data_categories": ["account", "usage"],
        "third_parties": ["aws"],
        "fingerprint": {
            "jurisdictions": [],
            "user_classes": ["registered_user"],
            "data_categories_coarse": ["account", "usage"],
            "data_categories": ["account", "usage"],
            "third_party_processors_expected": ["aws"],
            "third_parties": ["aws"],
            "lawful_basis": ["contract"],
            "retention_period": "365 days",
            "project_name": "TestProj",
            "controller_contact": "privacy@example.com",
        },
        "required_documents": [
            {"doc_type": "privacy_policy", "template_id": "privacy_policy"},
            {"doc_type": "tos", "template_id": "tos"},
            {"doc_type": "cookie_banner", "template_id": "cookie_banner"},
        ],
    }


@pytest.mark.asyncio
async def test_renders_all_three_writer_docs(tmp_path, overlay_with_three_docs):
    res = await legal_document_render(
        mission_id=999,
        workspace_path=str(tmp_path),
        overlay_obj=overlay_with_three_docs,
    )
    assert res["ok"]
    assert len(res["rendered"]) == 3
    legal_dir = tmp_path / ".compliance" / "legal"
    assert (legal_dir / "terms_of_service.md").is_file()
    assert (legal_dir / "privacy_policy.md").is_file()
    assert (legal_dir / "cookie_policy.md").is_file()
    # Each rendered draft must include the LEGAL REVIEW marker.
    for name in ("terms_of_service", "privacy_policy", "cookie_policy"):
        content = (legal_dir / f"{name}.md").read_text(encoding="utf-8")
        assert content
        # privacy_policy template is the pre-existing one and may not carry
        # the marker; only assert for the new T4B templates.
        if name in ("terms_of_service", "cookie_policy"):
            assert "[LEGAL REVIEW REQUIRED]" in content


@pytest.mark.asyncio
async def test_overlay_missing_returns_failure(tmp_path):
    res = await legal_document_render(
        mission_id=999,
        workspace_path=str(tmp_path),
        overlay_obj=None,
    )
    assert res["ok"] is False
    assert res.get("reason") == "overlay_missing"


@pytest.mark.asyncio
async def test_overlay_loaded_from_disk(tmp_path, overlay_with_three_docs):
    overlay_path = tmp_path / "compliance_overlay.json"
    overlay_path.write_text(
        json.dumps(overlay_with_three_docs), encoding="utf-8",
    )
    res = await legal_document_render(
        mission_id=999,
        workspace_path=str(tmp_path),
        overlay_obj=None,
    )
    assert res["ok"]
    assert len(res["rendered"]) == 3


@pytest.mark.asyncio
async def test_empty_required_documents_is_ok_with_skipped(tmp_path):
    overlay = {"required_documents": []}
    res = await legal_document_render(
        mission_id=999,
        workspace_path=str(tmp_path),
        overlay_obj=overlay,
    )
    assert res["ok"]
    assert res["rendered"] == []
    assert "all" in res["skipped"]


@pytest.mark.asyncio
async def test_unknown_doc_type_is_skipped_not_failed(tmp_path):
    overlay = {
        "required_documents": [
            {"doc_type": "data_processing_record"},  # not writer-facing
            {"doc_type": "privacy_policy"},
        ],
    }
    res = await legal_document_render(
        mission_id=999,
        workspace_path=str(tmp_path),
        overlay_obj=overlay,
    )
    assert res["ok"]
    assert len(res["rendered"]) == 1
    skipped_doc_types = [s.get("doc_type") for s in res["skipped"]]
    assert "data_processing_record" in skipped_doc_types


@pytest.mark.asyncio
async def test_missing_template_handled_via_error_entry(tmp_path, monkeypatch):
    """When compliance_template_render returns ok=False the run keeps going."""
    overlay = {
        "required_documents": [
            {"doc_type": "tos"},
            {"doc_type": "privacy_policy"},
        ],
    }

    real_render = None
    from src.tools import compliance_templates as ct

    real_render = ct.compliance_template_render

    def _stub(fingerprint, doc_type, lang="en", template_root=None):
        if doc_type == "tos":
            return {"ok": False, "error": "no template"}
        return real_render(
            fingerprint, doc_type, lang=lang, template_root=template_root,
        )

    monkeypatch.setattr(ct, "compliance_template_render", _stub)

    res = await legal_document_render(
        mission_id=999,
        workspace_path=str(tmp_path),
        overlay_obj=overlay,
    )
    # privacy_policy still rendered -> ok=True
    assert res["ok"]
    error_doc_types = [e.get("doc_type") for e in res["errors"]]
    assert "tos" in error_doc_types
