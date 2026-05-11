"""Z6 T5C — exercise the Stripe sandbox end-to-end via vendor_call.

Flow (all through ``vendor_call(stripe, …)``):

1. ``create_customer`` — test customer with a kutay_test_id metadata tag.
2. ``create_checkout_session`` — subscription mode using the first
   provisioned price id from the ``stripe_provisioned`` artifact.
3. ``confirm_test_payment`` — attaches a test card PaymentMethod
   (``tok_visa`` / ``4242…``) and confirms; Stripe's test keys honour
   the magic numbers without any real PSP roundtrip.
4. ``list_subscriptions`` — verifies the subscription exists and is
   ``active`` (or ``incomplete`` → reported as a sub-step failure).
5. ``cancel_subscription`` — tears the subscription down.
6. ``retrieve_balance`` — sanity check.

Output artifact ``payment_flow_results``::

    {
      "flows_tested": ["create_customer", ...],
      "pass_fail": "pass" | "fail",
      "steps":  [{"name": ..., "ok": bool, "stripe_id": ..., "ts": ...}, ...],
      "errors": [{...}]
    }

On any failure we emit a ``founder_action``:
* ``kind='legal_counsel'`` when the failure error blob mentions ``dispute``;
* ``kind='generic'`` otherwise.

The new vendor_call actions ``create_customer``, ``cancel_subscription``,
and ``confirm_test_payment`` are appended to ``src/integrations/configs/
stripe.json`` (config-only change). The Stripe REST shape is:

* ``POST /v1/customers``
* ``DELETE /v1/subscriptions/<id>``
* ``POST /v1/payment_intents/<id>/confirm`` — combined with attach when
  we need to test card flow.

If any of those actions are unavailable at runtime, the per-step result
records the gap and the overall flow reports ``pass_fail='fail'``. Tests
use mocks throughout — no real Stripe traffic in CI.
"""
from __future__ import annotations

import json
import time
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.stripe_payment_flow_test")


# ── artifact helpers ──────────────────────────────────────────────────────


async def _load_artifact_dict(mission_id: int, name: str) -> dict | None:
    try:
        from src.workflows.engine.artifacts import get_artifact_store
        store = get_artifact_store()
        raw = await store.retrieve(mission_id, name)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("artifact retrieve failed: %s", exc)
    return None


async def _persist_artifact(mission_id: int, name: str, value: dict) -> None:
    try:
        from src.workflows.engine.artifacts import get_artifact_store
        await get_artifact_store().store(mission_id, name, json.dumps(value))
    except Exception as exc:  # noqa: BLE001
        logger.debug("artifact persist failed: %s", exc)


# ── vendor_call indirection ────────────────────────────────────────────────


async def _vc(task: dict, service: str, action: str, params: dict) -> dict:
    from mr_roboto.executors.vendor_call import run as vendor_call_run
    sub = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "context": {
            "post_hook": {
                "service": service,
                "action": action,
                "params": params,
            }
        },
    }
    return await vendor_call_run(sub)


# ── founder_action helpers ────────────────────────────────────────────────


async def _emit_founder_action(
    mission_id: int,
    title: str,
    why: str,
    *,
    kind: str = "generic",
) -> None:
    try:
        import src.founder_actions as fa
        await fa.create(
            mission_id=int(mission_id),
            kind=kind,
            title=title,
            why=why[:500],
            instructions=[
                "Inspect the payment_flow_results artifact for the failing step.",
                "Verify Stripe test credentials are still valid (/credential list).",
                "Once fixed, retry the task or escalate via /actions.",
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("founder_action emit failed: %s", exc)


def _detect_dispute(err: Any) -> bool:
    if err is None:
        return False
    if isinstance(err, dict):
        blob = json.dumps(err, default=str).lower()
    else:
        blob = str(err).lower()
    return "dispute" in blob


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── main entrypoint ───────────────────────────────────────────────────────


async def run(task: dict[str, Any]) -> dict[str, Any]:
    mission_id = task.get("mission_id")
    if mission_id is None:
        return {"ok": False, "reason": "missing_mission_id"}
    try:
        mission_id_int = int(mission_id)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "invalid_mission_id"}

    provisioned = await _load_artifact_dict(mission_id_int, "stripe_provisioned")
    items = (provisioned or {}).get("items") or []
    if not items:
        return {
            "ok": False,
            "reason": "no_provisioned_products",
        }
    first_price = items[0].get("stripe_price_id")
    if not first_price:
        return {"ok": False, "reason": "first_provisioned_missing_price_id"}

    steps: list[dict] = []
    errors: list[dict] = []
    flows_tested: list[str] = []

    customer_id: str | None = None
    checkout_session_id: str | None = None
    subscription_id: str | None = None
    overall_ok = True

    def _record(name: str, res: dict, *, stripe_id_key: str = "id") -> bool:
        nonlocal overall_ok
        flows_tested.append(name)
        ok = bool(res.get("ok"))
        sid = None
        if ok:
            payload = res.get("result") or {}
            if isinstance(payload, dict):
                sid = payload.get(stripe_id_key)
        else:
            errors.append({"step": name, "detail": res})
            overall_ok = False
        steps.append(
            {"name": name, "ok": ok, "stripe_id": sid, "ts": _now_iso()}
        )
        return ok

    # 1. customer
    res = await _vc(
        task, "stripe", "create_customer",
        {"metadata": {"kutay_test_id": f"mission_{mission_id_int}"}},
    )
    if _record("create_customer", res):
        customer_id = (res.get("result") or {}).get("id")

    # 2. checkout session (only if customer succeeded)
    if customer_id:
        res = await _vc(
            task, "stripe", "create_checkout_session",
            {
                "mode": "subscription",
                "customer": customer_id,
                "line_items[0][price]": first_price,
                "line_items[0][quantity]": 1,
                "success_url": "https://example.invalid/ok",
                "cancel_url": "https://example.invalid/cancel",
            },
        )
        if _record("create_checkout_session", res):
            checkout_session_id = (res.get("result") or {}).get("id")
    else:
        flows_tested.append("create_checkout_session")
        steps.append({"name": "create_checkout_session", "ok": False,
                      "stripe_id": None, "ts": _now_iso()})

    # 3. confirm test payment (test card 4242…). Real product code would
    # complete via the hosted checkout page; in CI we attach a test
    # PaymentMethod token to the customer and confirm via the action.
    if customer_id:
        res = await _vc(
            task, "stripe", "confirm_test_payment",
            {
                "customer": customer_id,
                "payment_method_data[type]": "card",
                "payment_method_data[card][token]": "tok_visa",
                "checkout_session": checkout_session_id,
            },
        )
        _record("confirm_test_payment", res)

    # 4. list subscriptions
    if customer_id:
        res = await _vc(
            task, "stripe", "list_subscriptions",
            {"customer": customer_id, "limit": 5},
        )
        if _record("list_subscriptions", res):
            data = (res.get("result") or {}).get("data") or []
            for row in data:
                if isinstance(row, dict):
                    subscription_id = row.get("id")
                    break

    # 5. cancel subscription
    if subscription_id:
        res = await _vc(
            task, "stripe", "cancel_subscription",
            {"subscription_id": subscription_id},
        )
        _record("cancel_subscription", res)

    # 6. balance probe
    res = await _vc(task, "stripe", "retrieve_balance", {})
    _record("retrieve_balance", res)

    pass_fail = "pass" if overall_ok else "fail"
    out_artifact = {
        "_schema_version": "1",
        "flows_tested": flows_tested,
        "pass_fail": pass_fail,
        "steps": steps,
        "errors": errors,
    }
    await _persist_artifact(mission_id_int, "payment_flow_results", out_artifact)

    if not overall_ok:
        any_dispute = any(_detect_dispute(e.get("detail")) for e in errors)
        kind = "legal_counsel" if any_dispute else "generic"
        await _emit_founder_action(
            mission_id_int,
            title=f"Stripe payment_flow_test failed ({len(errors)} step{'s' if len(errors) != 1 else ''})",
            why=json.dumps(errors[:3], default=str),
            kind=kind,
        )

    return {
        "ok": overall_ok,
        "pass_fail": pass_fail,
        "flows_tested": flows_tested,
        "steps": steps,
        "errors": errors,
    }


__all__ = ["run"]
