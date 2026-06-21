"""Tests for rendering the rejection ledger into the retry prompt (T2).

Pure: ledger list in -> rendered string out. No DB, no LLM.

The block is gated on >=2 entries (a single rejection is already covered
by the existing schema checklist). Header ``## Prior attempts (do not
repeat):`` + one WHOLE line per entry; if the block would exceed ~2000
chars, OLDEST whole entries are dropped (no byte-slice — spec C1/F5).
"""
from __future__ import annotations

from coulson.context import render_rejection_ledger


def test_two_entries_render_with_header():
    ledger = [
        {"attempt": 1, "category": "quality", "reason": "schema: missing Risks", "out_hash": "a"},
        {"attempt": 2, "category": "quality", "reason": "grade: shallow analysis", "out_hash": "b"},
    ]
    out = render_rejection_ledger(ledger)
    assert out
    assert "## Prior attempts (do not repeat):" in out
    assert "schema: missing Risks" in out
    assert "grade: shallow analysis" in out
    assert "attempt 1" in out
    assert "attempt 2" in out


def test_single_entry_no_header():
    ledger = [{"attempt": 1, "category": "quality", "reason": "only one", "out_hash": "a"}]
    out = render_rejection_ledger(ledger)
    assert out == ""


def test_empty_ledger_no_header():
    assert render_rejection_ledger([]) == ""
    assert render_rejection_ledger(None) == ""


def test_each_entry_is_whole_line_not_byteslice():
    ledger = [
        {"attempt": 1, "category": "quality", "reason": "R" * 400, "out_hash": "a"},
        {"attempt": 2, "category": "quality", "reason": "S" * 400, "out_hash": "b"},
    ]
    out = render_rejection_ledger(ledger)
    # Both full reasons present uncut (no "..." byte truncation of a kept line)
    assert "R" * 400 in out
    assert "S" * 400 in out


def test_oldest_entries_dropped_when_over_budget():
    # Many large entries -> oldest WHOLE entries dropped, newest kept whole.
    ledger = [
        {"attempt": i, "category": "quality", "reason": f"attempt{i}-" + ("Z" * 300), "out_hash": str(i)}
        for i in range(1, 20)
    ]
    out = render_rejection_ledger(ledger)
    assert len(out) <= 2200  # ~2000 budget + header slack
    # newest entry kept
    assert "attempt19-" in out
    # an oldest entry dropped
    assert "attempt1-" not in out.replace("attempt19-", "").replace("attempt18-", "")
    # every line that IS present is whole (its 300 Z's intact)
    for line in out.splitlines():
        if line.startswith("- attempt"):
            assert "Z" * 300 in line


def test_renders_in_attempt_order():
    ledger = [
        {"attempt": 1, "category": "quality", "reason": "first", "out_hash": "a"},
        {"attempt": 2, "category": "quality", "reason": "second", "out_hash": "b"},
        {"attempt": 3, "category": "quality", "reason": "third", "out_hash": "c"},
    ]
    out = render_rejection_ledger(ledger)
    assert out.index("first") < out.index("second") < out.index("third")
