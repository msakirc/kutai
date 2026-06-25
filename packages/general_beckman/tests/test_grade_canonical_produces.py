"""Grade chain validates the CANONICAL produces artifact, not ``tasks.result``.

Root cause (task 567373, mission 90, [0.1] product_charter): schema'd
``produces`` markdown steps auto-strip ``write_file`` — so the writer NARRATES
("Wrote mission_90/.charter/product_charter.md containing all five sections…")
instead of emitting the body. ``materialize_produces`` writes the correct
artifact to disk (unconditionally), and ``verify_charter_shape`` + downstream
consumers read THAT. But the grade chain read ``source.result`` (the narration,
which has no ``##`` headers) → the schema gate reported "missing all 5 sections"
→ a false-reject the writer cannot fix → byte-identical re-emit → degenerate DLQ.

Uniform fix: for a single-``produces`` step the artifact is the materialized
disk canonical. ``resolve_produces_artifact`` PULLS that, and the grade branch
overrides the in-memory ``source['result']`` once so BOTH the deterministic
schema gate AND the LLM grader (build_grading_spec reads source.result) judge
the artifact, exactly as the rest of the engine already does.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest


_CHARTER_SCHEMA = {
    "product_charter": {
        "type": "markdown",
        "required_sections": [
            "Product Positioning", "Brand Keywords", "Core Problem",
            "Goals & Mission", "Solutions We Own",
        ],
        "_schema_version": "1",
    }
}

# The agent's raw final_answer for a write-stripped step: a prose narration.
# It NAMES the sections but carries no markdown headers — the exact 567373 shape.
_NARRATION = (
    "Wrote `mission_7/.charter/product_charter.md` (7.8KB) containing all five "
    "required sections: Product Positioning, Brand Keywords (10 bullets), Core "
    "Problem / JTBD, Goals & Mission with Desired Outcomes, and Solutions We "
    "Own (5 solutions). The document is coherent and contains no placeholders."
)

# The canonical artifact materialize_produces wrote to disk — real headers.
_CANONICAL = """## Product Positioning

For busy professionals, HabitHub is the habit tracker that bets on streaks.

## Brand Keywords

- **Streak** — the core loop.
- **Nudge** — gentle reminders.
- **Insight** — weekly review.
- **Calm** — no shame UX.
- **Momentum** — compounding wins.

## Core Problem / JTBD

People start habits and quit by week two; the job is staying consistent.

## Goals & Mission

Mission: make consistency effortless.

### Desired Outcomes
- 80% week-4 retention.
- Daily active streaks.
- Positive affect.
- Referral growth.

## Solutions We Own

### Streak Engine
What it solves: drop-off. Typical path: log → streak → nudge. Outcome for the
user: consistency. Boundaries: no social pressure. Guiding principles: calm.
"""


def _write_disk(ws_dir: str, rel_path: str, content: str) -> None:
    abs_path = os.path.join(ws_dir, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(content)


# ── Unit: the resolver ───────────────────────────────────────────────────────

def test_resolve_produces_artifact_reads_disk_over_narration(tmp_path, monkeypatch):
    import src.tools.workspace as ws
    from src.workflows.engine.hooks import resolve_produces_artifact

    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(tmp_path))
    _write_disk(str(tmp_path), "mission_7/.charter/product_charter.md", _CANONICAL)

    source = {"id": 1, "mission_id": 7, "result": _NARRATION}
    source_ctx = {"produces": ["mission_7/.charter/product_charter.md"]}

    out = resolve_produces_artifact(source, source_ctx)
    assert isinstance(out, str) and "## Product Positioning" in out
    assert out != _NARRATION


def test_resolve_returns_none_for_non_produces(tmp_path, monkeypatch):
    import src.tools.workspace as ws
    from src.workflows.engine.hooks import resolve_produces_artifact

    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(tmp_path))
    source = {"id": 1, "mission_id": 7, "result": _NARRATION}
    assert resolve_produces_artifact(source, {"produces": []}) is None
    # Multi-produces is materialized per-file; output_value can't stand in.
    assert resolve_produces_artifact(
        source, {"produces": ["a.md", "b.md"]}
    ) is None


def test_resolve_returns_none_when_disk_missing(tmp_path, monkeypatch):
    import src.tools.workspace as ws
    from src.workflows.engine.hooks import resolve_produces_artifact

    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(tmp_path))
    source = {"id": 1, "mission_id": 7, "result": _NARRATION}
    source_ctx = {"produces": ["mission_7/.charter/product_charter.md"]}
    assert resolve_produces_artifact(source, source_ctx) is None


# ── Behavioral: the grade gate judges canonical, not narration ───────────────

@pytest.mark.asyncio
async def test_grade_validates_canonical_disk_not_result_narration(tmp_path, monkeypatch):
    import general_beckman.apply as apply_mod
    import src.tools.workspace as ws

    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(tmp_path))
    _write_disk(str(tmp_path), "mission_7/.charter/product_charter.md", _CANONICAL)

    async def fake_apply(child_task, verdict):  # must NOT fire on a valid artifact
        raise AssertionError(
            f"valid on-disk charter must not short-circuit grade: {verdict.raw}"
        )

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", fake_apply)

    source = {
        "id": 567373, "mission_id": 7, "result": _NARRATION,
        "title": "product_charter", "description": "write the product charter",
    }
    source_ctx = {
        "artifact_schema": _CHARTER_SCHEMA,
        "produces": ["mission_7/.charter/product_charter.md"],
    }

    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=901)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", source, source_ctx)

    # Schema gate passed on the disk canonical → LLM grade child enqueued.
    enq.assert_awaited_once()
