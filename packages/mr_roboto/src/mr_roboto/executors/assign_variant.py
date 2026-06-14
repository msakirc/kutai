"""Z9 Growth T5D/T5E — assign_variant mechanical executor.

Splits a Phase-8+ feature mission into A/B variants: creates ``control``
and ``treatment`` rows in ``experiment_variants``, wires a PostHog
feature-flag (50/50 multivariate rollout) via ``vendor_call``, and stamps
the ``assignment_rule`` so the verdict pipeline can later pick a winner.

Routed via ``mr_roboto.run`` when ``payload["action"] == "assign_variant"``.
Added to i2p as Phase-8 step ``8.0ab`` (mechanical, default-on).

Architecture contract
---------------------
**Mechanical, no LLM.** Variant assignment and the insufficient-N guard
are deterministic. The later A/B *winner* call reuses
``src/growth/verdict_stats.py`` (also no LLM). Nothing here touches the
``LLMDispatcher``.

Insufficient-N guard
--------------------
A/B is only meaningful with enough traffic. Before splitting we ask
PostHog for the product's daily-active-user count (``get_active_users``;
mock-mode safe offline). If DAU < ``MIN_DAILY_ACTIVE`` (100) we do NOT
split — the feature ships at 100% rollout, a ``growth_events`` row
``kind='ab_skipped_low_n'`` records the skip, and the hypothesis still
stands so the verdict loop measures the single-arm outcome.

Default-on / opt-out
--------------------
Every Phase-8+ feature mission runs this step. ``mission.context['use_ab']``
defaults true; ``/experiment_disable <mission_id>`` flips it false, which
this executor honours (skips the split, no DB rows, ok=True).

Pricing variant (T5E)
---------------------
When ``payload['variant_kind'] == 'pricing'`` the ``treatment`` arm's
``assignment_rule`` carries a Stripe price-id instead of a flag bucket.
The actual Stripe price object is created by the founder-gated
``/confirm pricing ...`` telegram flow — this executor only records the
variant ledger rows; it never moves real money.
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.assign_variant")

# Founder-decided floor: below this daily-active count an A/B split is
# statistically pointless — see docs/i2p-evolution/09-growth-v2.md
# "Insufficient-N guard".
MIN_DAILY_ACTIVE: int = 100


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


def _pick_payload(task: dict, ctx: dict) -> dict:
    spec = ctx.get("payload")
    if isinstance(spec, dict) and spec:
        return spec
    payload = task.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _is_num(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


async def _mission_context(mission_id: int | None) -> dict:
    """Decode ``missions.context`` JSON to a dict (best-effort)."""
    if mission_id is None:
        return {}
    try:
        from dabidabi import get_mission

        row = await get_mission(mission_id)
        if not row:
            return {}
        raw = row.get("context")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            return json.loads(raw) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("mission context decode failed", error=str(exc))
    return {}


async def _daily_active_users(task: dict) -> tuple[bool, int]:
    """Pull daily-active-user count via PostHog ``get_active_users``.

    Returns ``(ok, dau)``. Mock-mode yields a deterministic count from
    ``configs/posthog.json`` with no network hop. On any failure returns
    ``(False, 0)`` — the caller treats an unknown DAU as insufficient.
    """
    from mr_roboto.executors.vendor_call import run as vendor_call_run

    sub = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "context": {
            "post_hook": {
                "service": "posthog",
                "action": "get_active_users",
                "params": {"project_id": "default"},
            }
        },
    }
    try:
        env = await vendor_call_run(sub)
    except Exception as exc:  # noqa: BLE001
        logger.warning("posthog get_active_users raised", error=str(exc))
        return False, 0
    if not (isinstance(env, dict) and env.get("ok")):
        return False, 0
    result = env.get("result") or {}
    # Mock shape: {"result": [{"label": ..., "data": [...], "count": N}]}
    inner = result.get("result") if isinstance(result, dict) else None
    if isinstance(inner, list) and inner:
        first = inner[0]
        if isinstance(first, dict):
            if _is_num(first.get("count")):
                return True, int(float(first["count"]))
            data = first.get("data")
            if isinstance(data, list) and data and _is_num(data[-1]):
                return True, int(float(data[-1]))
    return False, 0


async def _wire_posthog_flag(
    task: dict, flag_key: str
) -> dict:
    """Create a 50/50 multivariate PostHog feature-flag via vendor_call."""
    from mr_roboto.executors.vendor_call import run as vendor_call_run

    sub = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "context": {
            "post_hook": {
                "service": "posthog",
                "action": "create_feature_flag",
                "params": {
                    "project_id": "default",
                    "key": flag_key,
                    "name": f"A/B experiment {flag_key}",
                    "active": True,
                    "filters": {
                        "multivariate": {
                            "variants": [
                                {"key": "control",
                                 "rollout_percentage": 50},
                                {"key": "treatment",
                                 "rollout_percentage": 50},
                            ]
                        }
                    },
                },
            }
        },
    }
    try:
        env = await vendor_call_run(sub)
    except Exception as exc:  # noqa: BLE001
        logger.warning("posthog create_feature_flag raised", error=str(exc))
        return {"ok": False, "flag_id": None}
    if isinstance(env, dict) and env.get("ok"):
        result = env.get("result") or {}
        flag_id = result.get("id") if isinstance(result, dict) else None
        return {"ok": True, "flag_id": flag_id}
    return {"ok": False, "flag_id": None}


def _latest_hypothesis_id(payload: dict) -> int | None:
    hid = payload.get("hypothesis_id")
    if _is_num(hid):
        return int(float(hid))
    return None


async def _resolve_hypothesis_id(
    payload: dict, mission_id: int | None
) -> int | None:
    """Resolve the hypothesis to attach variants to.

    Explicit ``payload['hypothesis_id']`` wins; otherwise the most recent
    pending hypothesis for this mission (recorded at Phase-7 step 7.0y).
    """
    hid = _latest_hypothesis_id(payload)
    if hid is not None:
        return hid
    if mission_id is None:
        return None
    try:
        from dabidabi import get_pending_hypotheses

        pending = await get_pending_hypotheses(mission_id) or []
        if pending:
            return int(pending[0].get("id") or 0) or None
    except Exception as exc:  # noqa: BLE001
        logger.debug("hypothesis resolve failed", error=str(exc))
    return None


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Assign A/B variants for one mission. Never raises."""
    from dabidabi import insert_variant
    from general_beckman import record_growth_event

    ctx = _parse_context(task)
    payload = _pick_payload(task, ctx)
    mission_id = task.get("mission_id")
    feature = str(payload.get("feature") or "feature")
    variant_kind = str(payload.get("variant_kind") or "feature").lower()

    # 1. Opt-out check — /experiment_disable flips mission.context['use_ab'].
    mission_ctx = await _mission_context(mission_id)
    use_ab = mission_ctx.get("use_ab", True)
    if use_ab is False or str(use_ab).lower() in ("0", "false", "no"):
        logger.info("assign_variant: A/B opted out for mission %s", mission_id)
        try:
            await record_growth_event(
                mission_id, "ab_skipped_disabled",
                {"feature": feature, "reason": "mission opted out via "
                 "/experiment_disable"},
            )
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "split": False, "reason": "ab_disabled",
                "mission_id": mission_id}

    # 2. Resolve the hypothesis these variants test.
    hypothesis_id = await _resolve_hypothesis_id(payload, mission_id)

    # 3. Insufficient-N guard — query DAU; below the floor ship 100%.
    dau_ok, dau = await _daily_active_users(task)
    if not dau_ok or dau < MIN_DAILY_ACTIVE:
        logger.info(
            "assign_variant: insufficient N (dau=%s) — 100%% rollout",
            dau,
        )
        try:
            await record_growth_event(
                mission_id, "ab_skipped_low_n",
                {
                    "feature": feature,
                    "hypothesis_id": hypothesis_id,
                    "daily_active": dau,
                    "min_required": MIN_DAILY_ACTIVE,
                    "dau_query_ok": dau_ok,
                    "reason": "daily-active below 100; A/B split skipped, "
                              "feature ships at 100% rollout, hypothesis "
                              "still recorded",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ab_skipped_low_n event failed", error=str(exc))
        return {
            "ok": True, "split": False, "reason": "insufficient_n",
            "daily_active": dau, "min_required": MIN_DAILY_ACTIVE,
            "hypothesis_id": hypothesis_id, "mission_id": mission_id,
        }

    # 4. Split — create control + treatment variant rows.
    flag_key = f"exp_m{mission_id}_{feature}".lower().replace(" ", "_")[:80]

    if variant_kind == "pricing":
        # T5E — treatment arm routes to a Stripe price-id. The actual
        # price object is created by the founder-gated /confirm pricing
        # flow; here we only record the ledger. control = current price.
        control_rule = json.dumps({
            "kind": "pricing", "arm": "control",
            "stripe_price_id": payload.get("control_price_id"),
        })
        treatment_rule = json.dumps({
            "kind": "pricing", "arm": "treatment",
            "stripe_price_id": payload.get("treatment_price_id"),
            "pending_founder_confirm": payload.get("treatment_price_id")
            is None,
        })
        flag_id = None
        flag_wired = False
    else:
        # T5D — posthog multivariate flag, 50/50.
        flag_res = await _wire_posthog_flag(task, flag_key)
        flag_id = flag_res.get("flag_id")
        flag_wired = bool(flag_res.get("ok"))
        control_rule = json.dumps({
            "kind": "feature", "arm": "control",
            "posthog_flag_key": flag_key, "posthog_flag_id": flag_id,
            "rollout_percentage": 50,
        })
        treatment_rule = json.dumps({
            "kind": "feature", "arm": "treatment",
            "posthog_flag_key": flag_key, "posthog_flag_id": flag_id,
            "rollout_percentage": 50,
        })

    try:
        control_id = await insert_variant(
            mission_id, hypothesis_id, "control", control_rule,
        )
        treatment_id = await insert_variant(
            mission_id, hypothesis_id, "treatment", treatment_rule,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("assign_variant: insert_variant failed", error=str(exc))
        return {"ok": False, "reason": "insert_failed", "error": str(exc)}

    try:
        await record_growth_event(
            mission_id, "ab_assigned",
            {
                "feature": feature,
                "variant_kind": variant_kind,
                "hypothesis_id": hypothesis_id,
                "control_variant_id": control_id,
                "treatment_variant_id": treatment_id,
                "posthog_flag_key": flag_key
                if variant_kind != "pricing" else None,
                "posthog_flag_id": flag_id,
                "flag_wired": flag_wired,
                "daily_active": dau,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ab_assigned event failed", error=str(exc))

    logger.info(
        "assign_variant complete mission=%s control=%s treatment=%s flag=%s",
        mission_id, control_id, treatment_id, flag_key,
    )
    return {
        "ok": True,
        "split": True,
        "variant_kind": variant_kind,
        "mission_id": mission_id,
        "hypothesis_id": hypothesis_id,
        "control_variant_id": control_id,
        "treatment_variant_id": treatment_id,
        "posthog_flag_key": flag_key if variant_kind != "pricing" else None,
        "posthog_flag_id": flag_id,
        "flag_wired": flag_wired,
        "daily_active": dau,
    }


__all__ = ["run", "MIN_DAILY_ACTIVE"]
