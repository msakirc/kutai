"""Z6 T7D — docs landed + cross-refs in place."""
from __future__ import annotations

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_founder_actions_md_exists():
    p = REPO_ROOT / "docs" / "architecture" / "founder_actions.md"
    assert p.is_file()
    body = p.read_text(encoding="utf-8")
    # Substantive content checks — not just an empty file.
    assert "founder_actions" in body
    assert "credential_paste" in body
    assert "vendor_enroll" in body
    assert "cost_ack" in body
    assert "blocked_on_founder_action" in body


def test_vendor_call_md_exists():
    p = REPO_ROOT / "docs" / "architecture" / "vendor_call.md"
    assert p.is_file()
    body = p.read_text(encoding="utf-8")
    assert "vendor_call" in body
    assert "IntegrationRegistry" in body
    assert "AGENT_ALLOWLIST" in body
    assert "audit" in body.lower()


def test_readme_points_to_v2():
    p = REPO_ROOT / "docs" / "i2p-evolution" / "00-README.md"
    assert p.is_file()
    body = p.read_text(encoding="utf-8")
    assert "06-real-world-bridge-v2.md" in body
    # v1 is referenced as superseded.
    assert "superseded" in body.lower() or "supersed" in body.lower()


def test_v2_doc_has_updates_for_each_tier():
    p = REPO_ROOT / "docs" / "i2p-evolution" / "06-real-world-bridge-v2.md"
    assert p.is_file()
    body = p.read_text(encoding="utf-8")
    for tier in ("T1 shipped", "T2 shipped", "T3 shipped", "T4 shipped",
                 "T5 shipped", "T6 shipped", "T7 shipped"):
        assert tier in body, f"missing Updates entry for {tier}"


def test_v2_doc_has_cross_references():
    p = REPO_ROOT / "docs" / "i2p-evolution" / "06-real-world-bridge-v2.md"
    body = p.read_text(encoding="utf-8")
    for zone in ("Z0", "Z1", "Z5", "Z8", "Z9", "Z10"):
        assert zone in body, f"missing cross-ref to {zone}"
