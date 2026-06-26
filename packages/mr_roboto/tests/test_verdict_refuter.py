"""Tier-2 adversarial refuter — pure spec builder + verdict parser + keep rule.

The refuter is a single batched admitted LLM child (mirrors SP6 critic-gate)
asked, per surviving unverifiable finding, to REFUTE it: confirm it is actually
true of the artifact (quote the evidence, or confirm a genuine absence) or
answer UNSUPPORTED. Default-to-UNSUPPORTED on uncertainty. We then keep only
findings the refuter supports — fail-closed against a refuter that confabulates
its OWN supporting quote.
"""
from __future__ import annotations

from mr_roboto.verdict_refuter import (
    build_refuter_spec,
    parse_refuter_output,
    refuter_keep,
)

_CANDIDATES = [
    {"target_artifact": "market_research_report.md",
     "problem": "Check 1 - figures lack source citations.",
     "content": "# Report\nTAM large, no citations.\n"},
    {"target_artifact": "product_charter.md",
     "problem": "Check 8 - contradictory pricing tiers.",
     "content": "# Charter\nno pricing here.\n"},
]


# ── spec builder ──────────────────────────────────────────────────────────────

def test_spec_is_overhead_raw_dispatch():
    spec = build_refuter_spec(_CANDIDATES)
    llm = spec["context"]["llm_call"]
    assert llm["raw_dispatch"] is True
    assert llm["call_category"] == "overhead"
    assert llm["messages"], "must carry a prompt message"


def test_spec_prompt_includes_each_finding_and_artifact():
    spec = build_refuter_spec(_CANDIDATES)
    prompt = spec["context"]["llm_call"]["messages"][0]["content"]
    assert "Check 1" in prompt
    assert "Check 8" in prompt
    assert "TAM large" in prompt          # artifact content injected
    assert "UNSUPPORTED" in prompt         # the refute instruction
    assert "index" in prompt.lower()       # indexed so we can map verdicts back


# ── verdict parser (fail-closed for WHOLE-output garbage) ────────────────────

def test_parse_good_json():
    raw = ('{"verdicts": [{"index": 0, "status": "SUPPORTED", "quote": "no citations"},'
           ' {"index": 1, "status": "UNSUPPORTED"}]}')
    out = parse_refuter_output(raw, 2)
    assert out[0]["status"] == "supported"
    assert out[1]["status"] == "unsupported"


def test_parse_fenced_json():
    raw = '```json\n{"verdicts": [{"index": 0, "status": "UNSUPPORTED"}]}\n```'
    out = parse_refuter_output(raw, 1)
    assert out[0]["status"] == "unsupported"


def test_parse_garbage_returns_none():
    """Whole-output unparseable → None → caller keeps ALL candidates (an outage
    must not silently disable the halt)."""
    assert parse_refuter_output("the model rambled with no json", 2) is None
    assert parse_refuter_output("", 2) is None


# ── per-candidate keep rule (fail-closed against confabulated support) ────────

def test_unsupported_dropped():
    assert refuter_keep({"status": "unsupported"}, "any content") is False


def test_supported_with_present_quote_kept():
    assert refuter_keep({"status": "supported", "quote": "no citations"},
                        "# Report\nTAM large, no citations.\n") is True


def test_supported_with_absent_quote_dropped():
    """Refuter claims SUPPORTED but its quote is not in the artifact — the
    support is fabricated → drop (fail-closed against the refuter)."""
    assert refuter_keep({"status": "supported", "quote": "completely free forever"},
                        "# Report\nTAM large.\n") is False


def test_supported_absence_claim_no_quote_kept():
    """An absence finding ('lacks X') has no quote to give — a SUPPORTED with no
    quote is the refuter confirming the absence → keep."""
    assert refuter_keep({"status": "supported"}, "# Report\nTAM large.\n") is True


def test_missing_verdict_entry_kept():
    """The refuter ran but did not address this candidate → keep (don't drop on
    silence)."""
    assert refuter_keep(None, "content") is True
    assert refuter_keep({}, "content") is True
