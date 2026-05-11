"""Z6 T4B — verify the 7 missing default/en compliance templates exist and render.

Each template:
- has a sibling .meta.json with version + last_reviewed + doc_type fields,
- renders with a minimal fingerprint via compliance_template_render(),
- contains at least one ``[LEGAL REVIEW REQUIRED]`` marker (starter-draft contract),
- emits well-formed Markdown (non-empty after render).
"""
from __future__ import annotations

import json
import os

import pytest

from src.tools.compliance_templates import (
    TEMPLATE_ROOT,
    compliance_template_render,
)


T4B_DOC_TYPES = [
    "cookie_banner",
    "dpa",
    "tos",
    "retention_policy",
    "age_gate",
    "accessibility_statement",
    "data_processing_record",
]


MINIMAL_FINGERPRINT = {
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
    "jurisdiction": "default",
}


@pytest.mark.parametrize("doc_type", T4B_DOC_TYPES)
def test_template_file_exists(doc_type):
    path = os.path.join(TEMPLATE_ROOT, "default", "en", f"{doc_type}.md.j2")
    assert os.path.isfile(path), f"missing template: {path}"


@pytest.mark.parametrize("doc_type", T4B_DOC_TYPES)
def test_meta_json_is_valid(doc_type):
    path = os.path.join(TEMPLATE_ROOT, "default", "en", f"{doc_type}.meta.json")
    assert os.path.isfile(path), f"missing meta: {path}"
    with open(path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta.get("version")
    assert meta.get("last_reviewed")
    assert meta.get("doc_type") == doc_type
    assert meta.get("lang") == "en"
    assert meta.get("jurisdiction") == "default"


@pytest.mark.parametrize("doc_type", T4B_DOC_TYPES)
def test_template_renders_with_minimal_fingerprint(doc_type):
    res = compliance_template_render(
        fingerprint=MINIMAL_FINGERPRINT,
        doc_type=doc_type,
        lang="en",
    )
    assert res["ok"], f"render failed: {res.get('error')}"
    assert res["rendered"]
    assert len(res["rendered"]) > 200, "rendered output is suspiciously short"


@pytest.mark.parametrize("doc_type", T4B_DOC_TYPES)
def test_template_contains_legal_review_marker(doc_type):
    res = compliance_template_render(
        fingerprint=MINIMAL_FINGERPRINT,
        doc_type=doc_type,
        lang="en",
    )
    assert res["ok"]
    assert "[LEGAL REVIEW REQUIRED]" in res["rendered"], (
        f"{doc_type} rendered with no [LEGAL REVIEW REQUIRED] marker — "
        "starter drafts MUST flag items that need counsel"
    )


@pytest.mark.parametrize("doc_type", T4B_DOC_TYPES)
def test_template_safe_with_empty_fingerprint(doc_type):
    """Defaults must guard against missing vars (default('[NOT SPECIFIED]'))."""
    res = compliance_template_render(
        fingerprint={"jurisdictions": [], "user_classes": []},
        doc_type=doc_type,
        lang="en",
    )
    # Privacy_policy template handles undefined vars via {% if %} guards.
    # Our T4B templates use Jinja `default()` filter so render must succeed.
    assert res["ok"], (
        f"{doc_type} render failed on empty fingerprint: {res.get('error')}"
    )
