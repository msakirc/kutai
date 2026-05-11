"""Z6 T1C + T6B — real-world bridge admission gate.

Before a ``needs_real_tools=true`` task gets dispatched, three prerequisites
must hold:

1. A vendor adapter matching ``task.context.real_tool_kind`` is registered
   in the IntegrationRegistry (T3 wires the actual configs).
2. Credentials for that service exist in the credential vault.
3. If the task is ``reversibility='irreversible'`` with a positive
   ``cost_estimate_usd``, the founder has acknowledged the spend at least
   once for this mission+step.

When any prerequisite is missing, the gate emits one or more
``founder_action`` rows and returns ``admit=False``. Beckman's
``next_task()`` then parks the task in status ``blocked_on_founder_action``
and moves on to the next candidate. Once the founder resolves the actions,
T1E's lifecycle hook unblocks the mission and the task becomes eligible
again on the next pump cycle.

This module is intentionally separate from the legacy ``admission.py``
(pool-pressure urgency). Two concerns; two files. Future folding is
fine but not required for T1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("beckman.z6_admission")


@dataclass
class AdmissionResult:
    admit: bool
    reason: str = ""
    founder_actions_emitted: list[int] = field(default_factory=list)


def _parse_context(task: dict) -> dict:
    raw = task.get("context")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _needs_real_tools(task: dict, ctx: dict) -> bool:
    """Truthy needs_real_tools? Prefer the indexed column; fall back to ctx."""
    col = task.get("needs_real_tools")
    if col is not None:
        try:
            return bool(int(col))
        except (TypeError, ValueError):
            return bool(col)
    return bool(ctx.get("needs_real_tools"))


def _split_kinds(kind: Optional[str]) -> list[str]:
    """Pipe-separated list ('vercel|railway|supabase') → ['vercel', 'railway', ...]"""
    if not kind:
        return []
    if isinstance(kind, list):
        return [str(k).strip() for k in kind if str(k).strip()]
    return [k.strip() for k in str(kind).split("|") if k.strip()]


def _resolve_adapter(kinds: list[str]) -> Optional[str]:
    """Return the first kind whose adapter is registered, else None.

    Lazy import — registry import touches integrations/configs/ on first
    call and we don't want admission to pay for that on every pump tick
    even when no needs_real_tools task is present.
    """
    if not kinds:
        return None
    try:
        from src.integrations.registry import get_integration_registry
        reg = get_integration_registry()
        for k in kinds:
            if reg.get(k) is not None:
                return k
    except Exception as e:  # noqa: BLE001
        logger.debug("integration registry lookup failed", error=str(e))
    return None


async def _resolve_adapter_with_cred(kinds: list[str]) -> Optional[str]:
    """Z6 T3D — prefer a kind with both adapter AND credentials.

    Returns the first kind in ``kinds`` that has (a) a registered adapter
    AND (b) credentials in the vault. Falls back to ``_resolve_adapter``
    (adapter-only first-match) when no kind has credentials — admission's
    downstream credential check then emits ``credential_paste`` on that
    kind.
    """
    if not kinds:
        return None
    try:
        from src.integrations.resolver import resolve_real_tool
        match = await resolve_real_tool(kinds)
        if match is not None:
            return match
    except Exception as e:  # noqa: BLE001
        logger.debug("resolver failed; fallback to adapter-only", error=str(e))
    return _resolve_adapter(kinds)


async def _prior_cost_ack(mission_id: int, step_id: Optional[str]) -> bool:
    """True if a resolved cost_ack exists for this mission+step."""
    if not step_id:
        return False
    try:
        import src.founder_actions as fa
        rows = await fa.list_by_mission(mission_id, status_filter=["done"])
        for r in rows:
            if r.kind == "cost_ack" and r.blocking_step_id == step_id:
                return True
    except Exception as e:  # noqa: BLE001
        logger.debug("prior cost_ack lookup failed", error=str(e))
    return False


async def _has_pending_action(
    mission_id: int,
    kind: str,
    step_id: Optional[str],
) -> bool:
    """De-dup guard: don't re-emit the same kind/step pair while one is open."""
    try:
        import src.founder_actions as fa
        rows = await fa.list_by_mission(
            mission_id, status_filter=["pending", "in_progress"],
        )
        for r in rows:
            if r.kind == kind and r.blocking_step_id == step_id:
                return True
    except Exception as e:  # noqa: BLE001
        logger.debug("pending action probe failed", error=str(e))
    return False


async def check_z6_admission(
    task: dict,
    mission_id: int,
) -> AdmissionResult:
    """Gate for a single task. Emits founder_actions on missing prereqs.

    Pure read of task state + side-effects only on the founder_actions
    table. Caller is responsible for flipping the task row to
    ``status='blocked_on_founder_action'`` when ``admit=False``.
    """
    ctx = _parse_context(task)
    if not _needs_real_tools(task, ctx):
        return AdmissionResult(admit=True, reason="not needs_real_tools")

    import src.founder_actions as fa
    emitted: list[int] = []
    task_id = task.get("id")
    step_id = ctx.get("workflow_step_id") or task.get("workflow_step_id")
    reversibility = (
        task.get("reversibility") or ctx.get("reversibility")
    )
    real_tool_kind = ctx.get("real_tool_kind")
    cost_estimate = ctx.get("cost_estimate_usd")

    # 2. Resolve real_tool_kind. Missing kind → generic founder_action.
    kinds = _split_kinds(real_tool_kind)
    if not kinds:
        if not await _has_pending_action(mission_id, "generic", step_id):
            a = await fa.create(
                mission_id=mission_id,
                kind="generic",
                title=f"Declare real_tool_kind for step {step_id or '?'}",
                why=(
                    "Step is marked needs_real_tools=true but no "
                    "real_tool_kind was declared. Cannot resolve which "
                    "vendor adapter to use."
                ),
                instructions=[
                    "Edit the i2p workflow JSON for this step.",
                    "Set 'real_tool_kind' to a single vendor or pipe-list "
                    "(e.g. 'vercel' or 'vercel|railway').",
                ],
                blocking_task_id=task_id,
                blocking_step_id=step_id,
                reversibility=reversibility,
            )
            emitted.append(a.id)
        return AdmissionResult(
            admit=False,
            reason="real_tool_kind missing",
            founder_actions_emitted=emitted,
        )

    # 3. Adapter check. T3D: prefer a kind with adapter+credentials; falls
    # back to adapter-only first-match so the existing credential_paste
    # path still fires when no kind has creds yet.
    matched_kind = await _resolve_adapter_with_cred(kinds)
    if matched_kind is None:
        if not await _has_pending_action(mission_id, "vendor_enroll", step_id):
            a = await fa.create(
                mission_id=mission_id,
                kind="vendor_enroll",
                title=f"Enroll vendor: {' or '.join(kinds)}",
                why=(
                    f"Step {step_id or '?'} needs a vendor adapter for "
                    f"{', '.join(kinds)} but none is registered in "
                    f"IntegrationRegistry."
                ),
                instructions=[
                    f"Configure one of: {', '.join(kinds)}.",
                    "Drop the JSON config into src/integrations/configs/ "
                    "and restart, or contact maintainer.",
                ],
                blocking_task_id=task_id,
                blocking_step_id=step_id,
                reversibility=reversibility,
            )
            emitted.append(a.id)
        return AdmissionResult(
            admit=False,
            reason=f"no adapter for {kinds}",
            founder_actions_emitted=emitted,
        )

    # 4. Credential check on the matched kind.
    try:
        from src.security.credential_store import get_credential
        cred = await get_credential(matched_kind)
    except Exception as e:  # noqa: BLE001
        logger.warning("credential lookup failed", service=matched_kind, error=str(e))
        cred = None
    if not cred:
        if not await _has_pending_action(mission_id, "credential_paste", step_id):
            a = await fa.create(
                mission_id=mission_id,
                kind="credential_paste",
                title=f"Paste {matched_kind} credentials",
                why=(
                    f"Step {step_id or '?'} needs to call {matched_kind} "
                    f"but no credentials are stored."
                ),
                instructions=[
                    f"Open the {matched_kind} dashboard.",
                    "Generate or copy API credentials.",
                    f"Send `/credential add {matched_kind} "
                    f"{{\"token\": \"<value>\"}}` to the bot.",
                ],
                blocking_task_id=task_id,
                blocking_step_id=step_id,
                expected_output_kind="credential",
                reversibility=reversibility,
            )
            emitted.append(a.id)
        return AdmissionResult(
            admit=False,
            reason=f"no credential for {matched_kind}",
            founder_actions_emitted=emitted,
        )

    # 5. Cost ack: irreversible + cost>0 → require prior ack for this step.
    try:
        cost_float = float(cost_estimate) if cost_estimate is not None else 0.0
    except (TypeError, ValueError):
        cost_float = 0.0
    if reversibility == "irreversible" and cost_float > 0:
        if not await _prior_cost_ack(mission_id, step_id):
            if not await _has_pending_action(mission_id, "cost_ack", step_id):
                a = await fa.create(
                    mission_id=mission_id,
                    kind="cost_ack",
                    title=f"Confirm spend ${cost_float:.2f} for step {step_id or '?'}",
                    why=(
                        f"Step {step_id or '?'} is irreversible and is "
                        f"estimated to cost ${cost_float:.2f}. Founder ack "
                        f"required before dispatch."
                    ),
                    instructions=[
                        f"Confirm or block via inline buttons / "
                        f"/action_done.",
                    ],
                    blocking_task_id=task_id,
                    blocking_step_id=step_id,
                    cost_estimate_usd=cost_float,
                    reversibility=reversibility,
                    expected_output_kind="ack_only",
                )
                emitted.append(a.id)
            return AdmissionResult(
                admit=False,
                reason=f"cost_ack pending (${cost_float:.2f})",
                founder_actions_emitted=emitted,
            )

    # All prereqs satisfied.
    return AdmissionResult(admit=True, reason="prereqs ok")


__all__ = ["AdmissionResult", "check_z6_admission"]
