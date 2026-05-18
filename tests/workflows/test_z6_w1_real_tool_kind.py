"""Z6 W1 — every needs_real_tools step declares real_tool_kind.

Audit-driven test. T1C admission gate + T3D resolver hinge on
``real_tool_kind`` being present on every step that flips
``needs_real_tools`` to true. Without it, the admission resolver
silently no-ops and the step skips its adapter availability check.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_WF_JSON = _REPO / "src" / "workflows" / "i2p" / "i2p_v3.json"


def _load_workflow() -> dict:
    with _WF_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_all_steps(wf: dict):
    """Yield (id_or_name, step_dict) for top-level steps + template substeps."""
    for s in wf.get("steps", []):
        yield s.get("id", s.get("name", "?")), s
        # Templates can hold sub-step blueprints under `substeps` or `template_steps`.
        for sub_field in ("substeps", "template_steps"):
            for sub in s.get(sub_field, []) or []:
                key = sub.get("template_step_id") or sub.get("id") or sub.get("name", "?")
                yield key, sub
    # Templates may live at the top level under "templates".
    for tpl in wf.get("templates", []) or []:
        for sub in tpl.get("steps", []) or []:
            key = sub.get("template_step_id") or sub.get("id") or sub.get("name", "?")
            yield key, sub


def test_every_needs_real_tools_step_declares_real_tool_kind():
    wf = _load_workflow()
    missing: list[str] = []
    for sid, s in _iter_all_steps(wf):
        if s.get("needs_real_tools") and not s.get("real_tool_kind"):
            missing.append(sid)
    assert not missing, (
        f"steps with needs_real_tools=true but no real_tool_kind: {missing}"
    )


def test_w1_specific_step_tags():
    """Exact values audit prescribed."""
    wf = _load_workflow()
    by_id = {s.get("id"): s for s in wf["steps"]}

    assert by_id["7.13"]["real_tool_kind"] == "vercel|railway|fly"
    assert by_id["13.1"]["real_tool_kind"] == "vercel|railway|supabase"
    assert by_id["13.3"]["real_tool_kind"] == "sentry|datadog|new_relic"

    # feat.13 lives inside a template step's substeps.
    feat13 = None
    for _sid, s in _iter_all_steps(wf):
        if s.get("template_step_id") == "feat.13":
            feat13 = s
            break
    assert feat13 is not None, "feat.13 template substep not found"
    assert feat13["real_tool_kind"] == "vercel|railway"


def test_expander_hoists_real_tool_kind_to_task_context():
    """Expander must lift step.real_tool_kind into task.context for admission."""
    from src.workflows.engine.expander import expand_steps_to_tasks

    step = {
        "id": "fake.1",
        "name": "fake_real_tools_step",
        "agent": "executor",
        "needs_real_tools": True,
        "real_tool_kind": "vercel|railway",
        "cost_estimate_usd": 0.5,
        "input_artifacts": [],
        "output_artifacts": [],
    }
    expanded = expand_steps_to_tasks([step], mission_id="1")
    assert expanded, "expander returned no tasks"
    task = expanded[0]
    ctx = task.get("context")
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
    assert ctx.get("real_tool_kind") == "vercel|railway"
    assert ctx.get("needs_real_tools") is True


@pytest.mark.asyncio
async def test_admission_resolver_matches_registered_adapter(monkeypatch):
    """real_tool_kind 'vercel|railway' should resolve to whichever adapter is registered."""
    from src.integrations import resolver as resolver_mod
    from src.integrations.resolver import resolve_real_tool

    class _Reg:
        def __init__(self, names):
            self._names = set(names)

        def get(self, name):
            return object() if name in self._names else None

    # Only railway is registered — resolver should pick it from the pipe-list.
    import src.integrations.registry as reg_mod
    import src.security.credential_store as cs_mod
    monkeypatch.setattr(
        reg_mod, "get_integration_registry", lambda: _Reg({"railway"}),
    )

    async def _has_railway(svc):
        return {"token": "x"} if svc == "railway" else None
    monkeypatch.setattr(cs_mod, "get_credential", _has_railway)

    picked = await resolve_real_tool("vercel|railway")
    assert picked == "railway"

    # None registered → resolver returns None (admission emits founder_action).
    monkeypatch.setattr(
        reg_mod, "get_integration_registry", lambda: _Reg(set()),
    )
    picked = await resolve_real_tool("vercel|railway")
    assert picked is None
