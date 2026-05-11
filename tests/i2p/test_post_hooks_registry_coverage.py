"""Structural test: every ``post_hooks`` kind declared in i2p_v3.json must
be registered in ``POST_HOOK_REGISTRY``.

Caught Z1's silent dead-post-hook bug: 6 mechanical kinds were declared on
steps (compliance_template_present, compliance_blocker_check,
find_similar_missions, index_idea_fingerprint, surface_prior_mission_hints,
prior_art_min_coverage) but never registered, so
``determine_posthooks`` filtered them out before dispatch.

Also verifies that every registered kind that the apply layer dispatches
to mechanical (verb != ``code_reviewer``) has a corresponding handler
branch in ``_posthook_agent_and_payload``.
"""
from __future__ import annotations

import json
from pathlib import Path

from general_beckman.posthooks import POST_HOOK_REGISTRY
from general_beckman.apply import (
    _Z1_MECHANICAL_KINDS, _posthook_agent_and_payload,
)
from general_beckman.result_router import RequestPostHook


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "workflows" / "i2p" / "i2p_v3.json"
)


def _all_post_hook_kinds_in_workflow() -> set[str]:
    with open(WORKFLOW_PATH, encoding="utf-8") as fh:
        wf = json.load(fh)
    kinds: set[str] = set()
    for step in wf.get("steps", []):
        for k in step.get("post_hooks") or []:
            if isinstance(k, str) and k.strip():
                kinds.add(k.strip())
    return kinds


def test_every_workflow_post_hook_kind_is_registered():
    """Every JSON ``post_hooks`` entry must exist in POST_HOOK_REGISTRY.

    Without this, ``determine_posthooks`` silently filters the kind at
    the registry gate and the post-hook never spawns a task.
    """
    declared = _all_post_hook_kinds_in_workflow()
    registered = set(POST_HOOK_REGISTRY.keys())
    missing = declared - registered
    assert not missing, (
        f"Workflow post_hooks declared in i2p_v3.json but not in "
        f"POST_HOOK_REGISTRY: {sorted(missing)}. "
        f"Add a PostHookSpec row to posthooks.py and a dispatch branch "
        f"in apply._posthook_agent_and_payload."
    )


def test_every_z1_mechanical_kind_dispatches():
    """Every Z1 kind must produce a (mechanical, payload) tuple.

    Locks the dispatch contract — if a future refactor drops a Z1 branch
    from ``_posthook_agent_and_payload``, this test surfaces it before
    the silent ValueError ever fires at runtime.
    """
    fake_source = {"id": 1, "mission_id": 1, "title": "test"}
    fake_ctx = {
        "produces": ["mission_1/test_artifact.json"],
        "workspace_path": "/tmp/mission_1",
        "idea_summary": "stub",
    }
    for kind in _Z1_MECHANICAL_KINDS:
        a = RequestPostHook(source_task_id=1, kind=kind, source_ctx=fake_ctx)
        agent_type, payload = _posthook_agent_and_payload(
            a, fake_source, fake_ctx,
        )
        assert agent_type == "mechanical", (
            f"Z1 kind {kind!r} routed to {agent_type!r}, expected 'mechanical'"
        )
        assert payload.get("posthook_kind") == kind, (
            f"Z1 kind {kind!r} payload posthook_kind={payload.get('posthook_kind')!r}"
        )
        assert payload.get("executor") == "mechanical"
        assert payload["payload"]["action"] == kind, (
            f"Z1 kind {kind!r} payload.action={payload['payload'].get('action')!r}"
        )


def test_z1_mechanical_kinds_match_registry():
    """Z1 kinds known to apply.py must all be registered."""
    unregistered = _Z1_MECHANICAL_KINDS - set(POST_HOOK_REGISTRY.keys())
    assert not unregistered, (
        f"Z1 mechanical kinds in apply.py but not in POST_HOOK_REGISTRY: "
        f"{sorted(unregistered)}"
    )
