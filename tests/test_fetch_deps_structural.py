"""fetch_deps must not byte-slice dependency results mid-content.

The legacy depends_on path dumped every upstream ``result`` and cut each
with ``text[:per_dep]`` (mid-line / mid-JSON), then raw-sliced the whole
joined block again with truncate_to_tokens. A model fed a result cut
mid-structure — under a header claiming the content is provided "in
full" — is misled. Contract: whole results are kept or dropped (honest
omission note); only a single oversized result is truncated, and that
truncation lands on a line boundary, clearly marked.
"""
from __future__ import annotations

import re

import pytest

from coulson.context import fetch_deps


def _patch_deps(monkeypatch, mapping: dict):
    async def _fake(depends_on):
        return {k: mapping[k] for k in depends_on if k in mapping}
    monkeypatch.setattr("dabidabi.get_completed_dependency_results", _fake)


@pytest.mark.asyncio
async def test_whole_results_kept_or_dropped_not_partially_sliced(monkeypatch):
    """Two deps, budget fits only the first whole: the second is dropped
    entirely (not partially injected) and an omission note is added."""
    dep1 = "ALPHA_LINE xxxxxxxxxx\n" * 140   # ~3000 chars, multi-line
    dep2 = "BETA_UNIQUE yyyyyyyyyy\n" * 140
    _patch_deps(monkeypatch, {
        10: {"id": 10, "title": "A", "result": dep1},
        11: {"id": 11, "title": "B", "result": dep2},
    })
    task = {"id": 1, "depends_on": [10, 11]}

    # budget ~ enough for one whole dep + intro, not two
    out = await fetch_deps(None, task, max_tokens=900)

    assert "ALPHA_LINE" in out, "first dep should be present"
    assert "BETA_UNIQUE" not in out, "second dep must be dropped whole, not partially sliced"
    assert "omitted to fit budget" in out.lower(), "honest omission note expected"


@pytest.mark.asyncio
async def test_single_oversized_dep_truncated_on_line_boundary(monkeypatch):
    """A lone dep larger than the whole budget is truncated — but on a
    line boundary, never mid-line, and clearly marked."""
    body = "".join(f"line{i:04d}_xxxxx\n" for i in range(600))  # uniform lines
    _patch_deps(monkeypatch, {10: {"id": 10, "title": "A", "result": body}})
    task = {"id": 1, "depends_on": [10]}

    out = await fetch_deps(None, task, max_tokens=300)

    assert "(truncated" in out, "truncation must be marked"
    # Every emitted dep line must be a COMPLETE line (no mid-line byte cut).
    for ln in out.split("\n"):
        if ln.startswith("line"):
            assert re.fullmatch(r"line\d{4}_xxxxx", ln), f"partial line: {ln!r}"


@pytest.mark.asyncio
async def test_small_deps_injected_in_full_no_marker(monkeypatch):
    """Deps that fit are injected whole with no truncation/omission noise."""
    _patch_deps(monkeypatch, {
        10: {"id": 10, "title": "A", "result": "short result one"},
        11: {"id": 11, "title": "B", "result": "short result two"},
    })
    task = {"id": 1, "depends_on": [10, 11]}

    out = await fetch_deps(None, task, max_tokens=4000)

    assert "short result one" in out
    assert "short result two" in out
    assert "omitted to fit budget" not in out.lower()
    assert "(truncated" not in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
