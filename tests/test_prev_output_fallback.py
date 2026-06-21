"""T4 — `_prev_output[:N]` is demoted to a no-durable-artifact FALLBACK.

The artifact-backed continuation reads the FULL prior draft via
``fetch_prior_draft`` (T3); the capped ``_prev_output`` survives ONLY for
pure-conversation output that has no durable artifact (spec §4.1). This
test pins two things:

1. Behavioural: the durable read-back returns the WHOLE draft (never a
   byte-slice), so an artifact-backed retry does not depend on the capped
   value.
2. Source-audit guard: every audited ``_prev_output[:N]`` / ``_prev[:N]``
   truncation site carries the explicit "fallback only" comment so a future
   edit cannot silently re-promote the truncated value to the live carrier.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from unittest.mock import AsyncMock

from coulson.context import fetch_prior_draft

_ROOT = Path(__file__).resolve().parents[1]
_HOOKS = _ROOT / "src" / "workflows" / "engine" / "hooks.py"
_APPLY = (
    _ROOT / "packages" / "general_beckman" / "src" / "general_beckman" / "apply.py"
)
_CONTEXT = _ROOT / "packages" / "coulson" / "src" / "coulson" / "context.py"

_FALLBACK_TAG = "fallback only"


@pytest.mark.asyncio
async def test_durable_readback_is_whole_not_truncated():
    big = "Z" * 9000  # larger than any [:6000]/[:4000] cap
    store = AsyncMock()
    store.retrieve = AsyncMock(
        side_effect=lambda mid, name: big if name == "art" else None
    )
    out = await fetch_prior_draft(
        output_names=["art"], mission_id=1, store=store, already_injected=set(),
    )
    assert big in out  # full 9k present — not capped at 6000/4000
    assert len(out) > 6000


def _truncation_sites(path: Path, patterns):
    """Yield (lineno, line) for each _prev_output/_prev truncation site."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        for pat in patterns:
            if re.search(pat, line):
                # Comment may be on the same line OR within the 2 lines above.
                window = "\n".join(lines[max(0, i - 3): i])
                yield i, line, window


def test_apply_prev_output_sites_are_marked_fallback():
    sites = list(_truncation_sites(_APPLY, [r"prev_output\[:\d+\]"]))
    assert sites, "expected _prev_output[:N] sites in apply.py"
    unmarked = [n for n, _l, w in sites if _FALLBACK_TAG not in w]
    assert not unmarked, f"apply.py _prev_output[:N] sites missing fallback comment: {unmarked}"


def test_hooks_prev_output_site_is_marked_fallback():
    sites = list(_truncation_sites(_HOOKS, [r"canonicalize_for_retry\("]))
    # the schema-fail _prev_output assignment uses canonicalize_for_retry(...)[:6000]
    marked = [w for _n, _l, w in sites if "[:6000]" in w or "_prev_output" in w]
    assert any(_FALLBACK_TAG in w for w in marked), \
        "hooks.py schema-fail _prev_output site missing fallback comment"


def test_context_prev_render_site_is_marked_fallback():
    sites = list(_truncation_sites(_CONTEXT, [r"_prev\[:\d+\]"]))
    assert sites, "expected _prev[:N] render site in context.py"
    unmarked = [n for n, _l, w in sites if _FALLBACK_TAG not in w]
    assert not unmarked, f"context.py _prev[:N] sites missing fallback comment: {unmarked}"
