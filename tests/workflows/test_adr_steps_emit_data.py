"""ADR steps must EMIT DATA, not hand-write files (Fix #4, mission 81).

The architect ADR steps used to declare a multi-produces list
(`<adr>.json` + `register.md`) and instruct the agent to `write_file` a large
escaped ADR JSON. Multi-produces disables the materializer's `output_value`
rescue (Cut #2), so the step depended entirely on the fragile big-content
write_file — which truncated / parse-failed / narration-wrapped and DLQ'd.

Healthy contract: each per-decision ADR step
  - declares a SINGLE `.json` produces (so materialize_produces rescues it
    from the returned artifact),
  - does NOT carry `register.md` (the 4.14 rollup rebuilds the register from
    the .json files; only 4.14 runs verify_adr_register),
  - instructs the agent to RETURN the ADR JSON object as final_answer, not to
    write files.
"""
from __future__ import annotations

import json
import os

import pytest

_I2P = os.path.join("src", "workflows", "i2p", "i2p_v3.json")

# Per-decision ADR emitters (NOT 4.14, the register rollup).
_ADR_STEP_IDS = ["4.1", "4.2", "4.4", "4.6", "4.8", "4.9", "4.10", "4.2a"]


def _steps():
    with open(_I2P, encoding="utf-8") as fh:
        return {s["id"]: s for s in json.load(fh)["steps"]}


@pytest.mark.parametrize("sid", _ADR_STEP_IDS)
def test_adr_step_produces_single_json(sid):
    s = _steps()[sid]
    produces = s.get("produces") or []
    assert produces, f"{sid} has no produces"
    assert all(str(p).endswith(".json") for p in produces), \
        f"{sid} produces must be .json only (no register.md sibling): {produces}"
    json_entries = [p for p in produces if str(p).endswith(".json")]
    assert len(json_entries) == 1, f"{sid} must declare exactly one .json: {produces}"
    assert not any("register.md" in str(p) for p in produces), \
        f"{sid} must NOT produce register.md (4.14 rollup owns it)"


@pytest.mark.parametrize("sid", _ADR_STEP_IDS)
def test_adr_step_instruction_emits_data_not_write_file(sid):
    s = _steps()[sid]
    instr = s.get("instruction") or ""
    # No mandate to append the register from this step.
    assert "register.md" not in instr, \
        f"{sid} instruction must not mention register.md (4.14 owns the register)"
    # No mandate to write the ADR file via write_file — the materializer writes
    # the declared produces from the returned artifact.
    assert "write_file" not in instr.lower(), \
        f"{sid} instruction must not tell the agent to write_file the ADR"


# ── end-to-end: agent returns the ADR, materializer writes it (no write_file) ──

_ADR_SCHEMA = {
    "architecture_pattern_decision": {
        "type": "object",
        "required_fields": [
            "adr_id", "title", "status", "context", "decision",
            "consequences", "options_considered", "chosen_option_id",
            "falsification_signal", "reversal_cost", "supersedes_adr_id",
        ],
        "_schema_version": "1",
    },
}

_ADR_JSON = json.dumps({
    "_schema_version": "1",
    "adr_id": "ADR-2026-06-11-001",
    "title": "Architecture Pattern Selection",
    "status": "proposed",
    "context": "ctx",
    "decision": "modular monolith",
    "consequences": "tradeoffs",
    "options_considered": [
        {"id": "OPT-A", "name": "monolith", "rationale_for": "x",
         "rationale_against": "y", "tech_maturity_score": 9,
         "novelty_benefit": "n", "reversal_cost": "low"},
        {"id": "OPT-B", "name": "microservices", "rationale_for": "x",
         "rationale_against": "y", "tech_maturity_score": 7,
         "novelty_benefit": "n", "reversal_cost": "high"},
    ],
    "chosen_option_id": "OPT-A",
    "falsification_signal": "if active SKU > 200 by week 4",
    "reversal_cost": "medium",
    "supersedes_adr_id": None,
})


@pytest.mark.asyncio
async def test_materializer_writes_adr_json_from_returned_artifact(tmp_path, monkeypatch):
    """The agent returns the ADR JSON (no write_file). With produces now a SINGLE
    .json, materialize_produces rescues output_value and writes a valid .json —
    no disk file pre-exists. Proves Fix #4 delivers the artifact end-to-end."""
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path), raising=False)
    from src.workflows.engine.hooks import materialize_produces

    rel = "mission_81/.adr/architecture_pattern_decision.json"
    ctx = {"produces": [rel], "artifact_schema": _ADR_SCHEMA}
    task = {"mission_id": 81, "agent_type": "architect"}

    out = await materialize_produces(ctx, task, {"result": _ADR_JSON}, _ADR_JSON)

    on_disk = json.loads((tmp_path / rel).read_text(encoding="utf-8"))
    assert on_disk["adr_id"] == "ADR-2026-06-11-001"
    assert on_disk["chosen_option_id"] == "OPT-A"
    assert len(on_disk["options_considered"]) == 2
    # Returned value parity with disk so the in-memory schema gate validates it.
    assert json.loads(out)["adr_id"] == "ADR-2026-06-11-001"
