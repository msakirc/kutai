"""Z6 T5B — provision Stripe products + prices from monetization_strategy.

Reads ``monetization_strategy.products[]`` and, for each item, ensures a
Stripe Product + Price pair exists. Idempotent via a deterministic
``metadata.kutay_id`` tag: re-runs are no-ops once products are
provisioned. The output artifact ``stripe_provisioned`` carries the
``(kutay_id, stripe_product_id, stripe_price_id)`` map downstream steps
(notably 13.12 ``payment_flow_test``) consume.

All vendor I/O goes through the ``vendor_call`` mechanical's underlying
helper — we do not import requests/httpx here. The wrapper handles
adapter resolution, auth, and failure → founder_action emission.

Idempotency
-----------
``kutay_id`` is ``sha1(mission_id + product_name + currency + interval)``
truncated to 16 hex chars. Before creating, we list_products with
``metadata[kutay_id]=<id>``; if any row matches, skip the create. The
same hash is reused on the price side via product_id lookup.

Failure handling
----------------
On any vendor_call failure for an individual product we abort the whole
run — half-provisioned states are confusing downstream. The
``vendor_call`` helper already emits a generic founder_action with the
service / action / error context.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.stripe_provision_products")


# ── artifact helpers ──────────────────────────────────────────────────────


async def _load_artifact_dict(mission_id: int, name: str) -> dict | None:
    try:
        from src.workflows.engine.hooks import get_artifact_store
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
        logger.debug("artifact store retrieve failed: %s", exc)
    try:
        from src.collaboration.blackboard import read_blackboard
        artifacts = await read_blackboard(int(mission_id), "artifacts")
        if isinstance(artifacts, dict):
            v = artifacts.get(name)
            if isinstance(v, dict):
                return v
            if isinstance(v, str) and v.strip():
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("blackboard read failed: %s", exc)
    return None


async def _persist_artifact(mission_id: int, name: str, value: dict) -> None:
    try:
        from src.workflows.engine.hooks import get_artifact_store
        await get_artifact_store().store(mission_id, name, json.dumps(value))
    except Exception as exc:  # noqa: BLE001
        logger.debug("artifact store persist failed: %s", exc)


# ── product schema validation ─────────────────────────────────────────────


def _kutay_id_for(mission_id: int, product: dict) -> str:
    seed = (
        f"{mission_id}|{product.get('name', '')}|"
        f"{(product.get('currency') or '').lower()}|"
        f"{product.get('interval') or 'one_time'}|"
        f"{product.get('price_cents') or 0}"
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _validate_product(product: Any) -> tuple[bool, str]:
    if not isinstance(product, dict):
        return False, "product is not a dict"
    if not product.get("name"):
        return False, "product.name missing"
    pc = product.get("price_cents")
    if not isinstance(pc, int) or pc <= 0:
        return False, "product.price_cents must be a positive int"
    cur = product.get("currency")
    if not isinstance(cur, str) or len(cur) != 3:
        return False, "product.currency must be a 3-letter ISO code"
    return True, ""


# ── vendor_call indirection (factored out so tests can patch) ─────────────


async def _vc(task: dict, service: str, action: str, params: dict) -> dict:
    """Call the vendor_call executor with a synthetic spec.

    Reuses the same dispatch surface the workflow uses so retries +
    failure cards behave identically.
    """
    from mr_roboto.executors.vendor_call import run as vendor_call_run
    sub_task = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "workflow_step_id": (
            (task.get("context") or {}).get("workflow_step_id")
            if isinstance(task.get("context"), dict)
            else None
        ) or task.get("workflow_step_id"),
        "context": {
            "post_hook": {
                "service": service,
                "action": action,
                "params": params,
            }
        },
    }
    return await vendor_call_run(sub_task)


# ── lookup helpers ────────────────────────────────────────────────────────


def _find_existing_product(list_result: dict, kutay_id: str) -> dict | None:
    """Scan list_products response for a row with matching metadata.kutay_id."""
    if not isinstance(list_result, dict):
        return None
    data = list_result.get("data") or list_result.get("result")
    if isinstance(data, dict):
        data = data.get("data")
    if not isinstance(data, list):
        return None
    for row in data:
        if not isinstance(row, dict):
            continue
        meta = row.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("kutay_id") == kutay_id:
            return row
    return None


# ── main entrypoint ───────────────────────────────────────────────────────


async def run(task: dict[str, Any]) -> dict[str, Any]:
    mission_id = task.get("mission_id")
    if mission_id is None:
        return {"ok": False, "reason": "missing_mission_id"}
    try:
        mission_id_int = int(mission_id)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "invalid_mission_id"}

    monet = await _load_artifact_dict(mission_id_int, "monetization_strategy")
    if not isinstance(monet, dict):
        return {"ok": False, "reason": "monetization_strategy_missing"}

    products = monet.get("products")
    if not isinstance(products, list) or not products:
        return {
            "ok": True,
            "skipped": True,
            "reason": "no_products_to_provision",
            "provisioned": [],
        }

    provisioned: list[dict] = []
    for prod in products:
        ok, err = _validate_product(prod)
        if not ok:
            return {
                "ok": False,
                "reason": "invalid_product_schema",
                "detail": err,
                "product": prod if isinstance(prod, dict) else None,
                "provisioned": provisioned,
            }

        kutay_id = _kutay_id_for(mission_id_int, prod)

        # 1. Look for existing product via list_products metadata filter.
        list_res = await _vc(
            task, "stripe", "list_products",
            {"limit": 100},
        )
        if not list_res.get("ok"):
            return {
                "ok": False,
                "reason": "list_products_failed",
                "detail": list_res,
                "provisioned": provisioned,
            }
        existing = _find_existing_product(list_res.get("result") or {}, kutay_id)

        if existing:
            stripe_product_id = existing.get("id")
            stripe_price_id = None
            # Existing product may already carry a default_price.
            default_price = existing.get("default_price")
            if isinstance(default_price, str):
                stripe_price_id = default_price
            elif isinstance(default_price, dict):
                stripe_price_id = default_price.get("id")
        else:
            create_res = await _vc(
                task, "stripe", "create_product",
                {
                    "name": prod["name"],
                    "metadata": {"kutay_id": kutay_id},
                },
            )
            if not create_res.get("ok"):
                return {
                    "ok": False,
                    "reason": "create_product_failed",
                    "detail": create_res,
                    "provisioned": provisioned,
                }
            stripe_product_id = (create_res.get("result") or {}).get("id")
            stripe_price_id = None

        # 2. Create price if we don't already have one.
        if not stripe_price_id and stripe_product_id:
            price_params: dict = {
                "product": stripe_product_id,
                "unit_amount": prod["price_cents"],
                "currency": prod["currency"].lower(),
                "metadata": {"kutay_id": kutay_id},
            }
            interval = prod.get("interval")
            if interval:
                price_params["recurring"] = {"interval": interval}
            price_res = await _vc(
                task, "stripe", "create_price", price_params,
            )
            if not price_res.get("ok"):
                return {
                    "ok": False,
                    "reason": "create_price_failed",
                    "detail": price_res,
                    "provisioned": provisioned,
                }
            stripe_price_id = (price_res.get("result") or {}).get("id")

        provisioned.append(
            {
                "product_name": prod["name"],
                "kutay_id": kutay_id,
                "stripe_product_id": stripe_product_id,
                "stripe_price_id": stripe_price_id,
            }
        )

    out_artifact = {
        "_schema_version": "1",
        "items": provisioned,
    }
    await _persist_artifact(mission_id_int, "stripe_provisioned", out_artifact)
    return {"ok": True, "provisioned": provisioned}


__all__ = ["run"]
