"""Z6 T4C — jurisdiction-variant resolution tests."""
from __future__ import annotations

import os

import pytest

from src.tools.compliance_templates import (
    TEMPLATE_ROOT,
    compliance_template_render,
)


MINIMAL_FP = {
    "user_classes": ["registered_user"],
    "data_categories_coarse": ["account", "usage"],
    "data_categories": ["account", "usage"],
    "third_party_processors_expected": ["aws"],
    "third_parties": ["aws"],
    "lawful_basis": ["contract"],
    "retention_period": "365 days",
    "project_name": "TestProj",
    "controller_contact": "privacy@example.com",
}


def _fp(jurisdictions):
    return {**MINIMAL_FP, "jurisdictions": list(jurisdictions)}


# ─── existence ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("doc_type", [
    "privacy_policy", "dpa", "retention_policy",
])
def test_gdpr_variant_file_exists(doc_type):
    path = os.path.join(TEMPLATE_ROOT, "gdpr", "en", f"{doc_type}.md.j2")
    assert os.path.isfile(path)
    meta = os.path.join(TEMPLATE_ROOT, "gdpr", "en", f"{doc_type}.meta.json")
    assert os.path.isfile(meta)


@pytest.mark.parametrize("doc_type", ["privacy_policy", "retention_policy"])
def test_ccpa_variant_file_exists(doc_type):
    path = os.path.join(TEMPLATE_ROOT, "ccpa", "en", f"{doc_type}.md.j2")
    assert os.path.isfile(path)
    meta = os.path.join(TEMPLATE_ROOT, "ccpa", "en", f"{doc_type}.meta.json")
    assert os.path.isfile(meta)


# ─── resolver picks variant when matched ─────────────────────────────────

@pytest.mark.parametrize("jurisdictions", [["gdpr"], ["EU"], ["UK"]])
def test_gdpr_resolved_for_privacy_policy(jurisdictions):
    res = compliance_template_render(
        fingerprint=_fp(jurisdictions),
        doc_type="privacy_policy",
    )
    assert res["ok"]
    assert os.sep + "gdpr" + os.sep in res["template_path"], (
        f"expected gdpr/ variant for {jurisdictions}, got {res['template_path']}"
    )


@pytest.mark.parametrize("jurisdictions", [["ccpa"], ["California"]])
def test_ccpa_resolved_for_privacy_policy(jurisdictions):
    res = compliance_template_render(
        fingerprint=_fp(jurisdictions),
        doc_type="privacy_policy",
    )
    assert res["ok"]
    assert os.sep + "ccpa" + os.sep in res["template_path"], (
        f"expected ccpa/ variant for {jurisdictions}, got {res['template_path']}"
    )


def test_gdpr_dpa_resolved():
    res = compliance_template_render(
        fingerprint=_fp(["gdpr"]),
        doc_type="dpa",
    )
    assert res["ok"]
    assert os.sep + "gdpr" + os.sep in res["template_path"]


def test_gdpr_retention_resolved():
    res = compliance_template_render(
        fingerprint=_fp(["gdpr"]),
        doc_type="retention_policy",
    )
    assert res["ok"]
    assert os.sep + "gdpr" + os.sep in res["template_path"]


def test_ccpa_retention_resolved():
    res = compliance_template_render(
        fingerprint=_fp(["ccpa"]),
        doc_type="retention_policy",
    )
    assert res["ok"]
    assert os.sep + "ccpa" + os.sep in res["template_path"]


# ─── fallback to default for uncovered jurisdiction ──────────────────────

def test_uncovered_jurisdiction_falls_back_to_default():
    res = compliance_template_render(
        fingerprint=_fp(["Japan"]),
        doc_type="privacy_policy",
    )
    assert res["ok"]
    assert os.sep + "default" + os.sep in res["template_path"]


def test_uncovered_jurisdiction_for_uncovered_doctype_uses_default():
    # gdpr has no tos.md.j2; should fall back to default/en/tos.md.j2.
    res = compliance_template_render(
        fingerprint=_fp(["gdpr"]),
        doc_type="tos",
    )
    assert res["ok"]
    assert os.sep + "default" + os.sep in res["template_path"]


# ─── GDPR/CCPA content sanity ────────────────────────────────────────────

def test_gdpr_privacy_policy_mentions_articles():
    res = compliance_template_render(
        fingerprint=_fp(["gdpr"]),
        doc_type="privacy_policy",
    )
    assert res["ok"]
    rendered = res["rendered"]
    assert "Article 6" in rendered or "Art. 6" in rendered
    assert "Article 15" in rendered or "Art. 15" in rendered
    assert "supervisory authority" in rendered.lower()


def test_ccpa_privacy_policy_mentions_consumer_rights():
    res = compliance_template_render(
        fingerprint=_fp(["ccpa"]),
        doc_type="privacy_policy",
    )
    assert res["ok"]
    rendered = res["rendered"]
    assert "1798" in rendered  # CCPA section refs
    assert "Do Not Sell or Share" in rendered
