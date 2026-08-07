"""Task 10 — 7.13 staging_environment runs mechanical deploy_staging.

The step was previously an `executor` (LLM) step that could only surface a
needs_clarification. With the deploy adapters + `deploy_staging` orchestrator
landed, 7.13 becomes a MECHANICAL step routed to `mr_roboto.deploy_staging`.

Opened with encoding="utf-8" — i2p_v3.json has non-cp1252 bytes; a bare open()
crashes on Windows (cp1252 default).
"""
import json
from pathlib import Path

_WF = Path(__file__).resolve().parents[2] / "src" / "workflows" / "i2p" / "i2p_v3.json"


def test_staging_env_is_mechanical_deploy_staging():
    with open(_WF, encoding="utf-8") as f:
        wf = json.load(f)
    step = next(s for s in wf["steps"] if s.get("name") == "staging_environment")
    assert step["agent"] == "mechanical"
    assert step["payload"]["action"] == "deploy_staging"
    assert step["payload"]["backend_arch"] == "nestjs_render"
