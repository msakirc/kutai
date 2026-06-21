"""Tests for same-step durable prior-draft read-back (T3).

On a workflow-step RETRY (worker_attempts > 0), the worker must see its
OWN prior draft WHOLE (spec C3/C1) — not the 6k-truncated _prev_output —
so a 75%-complete attempt is continued/fixed, not restarted. The draft
comes from the durable artifact store keyed by the step's OUTPUT artifact
NAMES (M2: names, not produces paths), preferring `<name>_summary` then
the bare name (mirroring _fetch_deps).

The store is mocked — no DB, no LLM.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from coulson.context import fetch_prior_draft


def _store(mapping: dict):
    s = AsyncMock()

    async def _ret(mid, name):
        return mapping.get(name)

    s.retrieve = AsyncMock(side_effect=_ret)
    return s


@pytest.mark.asyncio
async def test_full_draft_injected_whole_no_truncation():
    big = "X" * 8000
    store = _store({"competitive_positioning": big})
    out = await fetch_prior_draft(
        output_names=["competitive_positioning"],
        mission_id=1,
        store=store,
        already_injected=set(),
    )
    assert "## Your prior draft" in out
    assert big in out  # full 8k, whole
    assert "[truncated]" not in out
    assert "..." not in out


@pytest.mark.asyncio
async def test_prefers_summary_then_bare():
    store = _store({
        "art_summary": "SUMMARY-FORM",
        "art": "FULL-FORM",
    })
    out = await fetch_prior_draft(
        output_names=["art"], mission_id=1, store=store, already_injected=set(),
    )
    assert "SUMMARY-FORM" in out
    assert "FULL-FORM" not in out


@pytest.mark.asyncio
async def test_falls_back_to_bare_when_no_summary():
    store = _store({"art": "FULL-ONLY"})
    out = await fetch_prior_draft(
        output_names=["art"], mission_id=1, store=store, already_injected=set(),
    )
    assert "FULL-ONLY" in out


@pytest.mark.asyncio
async def test_no_stored_artifact_no_block():
    store = _store({})
    out = await fetch_prior_draft(
        output_names=["nothing_here"], mission_id=1, store=store, already_injected=set(),
    )
    assert out == ""


@pytest.mark.asyncio
async def test_no_names_no_block():
    store = _store({"art": "v"})
    out = await fetch_prior_draft(
        output_names=[], mission_id=1, store=store, already_injected=set(),
    )
    assert out == ""


@pytest.mark.asyncio
async def test_dedup_skips_names_already_injected_by_deps():
    store = _store({"art": "DRAFTVAL"})
    out = await fetch_prior_draft(
        output_names=["art"], mission_id=1, store=store,
        already_injected={"art"},
    )
    # art was already injected by deps -> no draft block
    assert out == ""


@pytest.mark.asyncio
async def test_no_mission_id_no_block():
    store = _store({"art": "v"})
    out = await fetch_prior_draft(
        output_names=["art"], mission_id=None, store=store, already_injected=set(),
    )
    assert out == ""


@pytest.mark.asyncio
async def test_multiple_names_all_injected_whole():
    store = _store({"a": "AAA" * 1000, "b": "BBB" * 1000})
    out = await fetch_prior_draft(
        output_names=["a", "b"], mission_id=1, store=store, already_injected=set(),
    )
    assert "AAA" * 1000 in out
    assert "BBB" * 1000 in out
