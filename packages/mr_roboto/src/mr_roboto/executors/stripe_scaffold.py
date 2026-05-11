"""Z6 T5A — build-phase Stripe scaffold executor.

Materialises a minimal payment / webhook scaffold under
``mission_<id>/api/`` when the mission's ``monetization_strategy`` calls
for Stripe billing. Pure file emission — no live Stripe calls. The
generated stubs are placeholders the agent fills in during Phase 7
implementation; their job is to ensure every Stripe-billing mission has
the same skeleton (env keys, webhook signature check, README) on disk
so later steps (T5B provisioning, T5C payment_flow_test) have somewhere
to wire into.

Inputs
------
* ``monetization_strategy`` artifact — when ``billing.provider`` is not
  ``stripe`` (or absent and the strategy has no products/pricing), the
  executor short-circuits with ``skipped=True``.
* ``system_architecture`` artifact — read for a stack hint
  (``backend.framework`` / ``stack``). For v1 we default to Python +
  FastAPI when no hint is found; TypeScript / Node stacks will follow
  in a future tier.

Outputs
-------
* Writes to ``<workspace_root>/mission_<mission_id>/api/``:
    - ``checkout/create_session.<ext>``
    - ``webhook/stripe.<ext>``
    - ``.env.example``
    - ``README_STRIPE.md``
* Returns ``{ok, files_written, stack, skipped?}``.

The executor itself is invoked from i2p_v3.json step ``7.0a.stripe_scaffold``
(see T5A workflow patch) with ``payload.action == "stripe_scaffold"``.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.stripe_scaffold")


# Where mission workspaces live on disk. Centralised so tests can monkeypatch.
def _mission_workspace_root() -> str:
    return os.environ.get("MISSION_WORKSPACE_ROOT", os.getcwd())


# ── stack detection ────────────────────────────────────────────────────────

_PYTHON_HINTS = ("fastapi", "flask", "django", "python", "starlette")
_TS_HINTS = ("typescript", "nextjs", "next.js", "node", "express", "ts")


def _detect_stack(system_architecture: dict | None) -> tuple[str, str]:
    """Return ``(stack_label, file_extension)``.

    Default: ``("python-fastapi", "py")``. TODO: expand to TS/Node when
    we wire those scaffolds.
    """
    if not isinstance(system_architecture, dict):
        return "python-fastapi", "py"
    blob = json.dumps(system_architecture, default=str).lower()
    if any(h in blob for h in _TS_HINTS) and not any(h in blob for h in _PYTHON_HINTS):
        # TODO(z6,t5): emit TS Express/Next-API scaffold instead.
        return "python-fastapi", "py"
    return "python-fastapi", "py"


# ── artifact load helpers ─────────────────────────────────────────────────

async def _load_artifact_dict(mission_id: int, name: str) -> dict | None:
    """Best-effort dict load from artifact store, blackboard fallback."""
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


def _wants_stripe(monetization: dict | None) -> bool:
    """Best-effort 'does this strategy call for Stripe billing?'.

    True when:
    * ``billing.provider == 'stripe'`` (canonical) OR
    * a ``products`` array exists with at least one priced item AND no
      conflicting provider name is set.

    False when the strategy is missing or explicitly names a non-Stripe
    provider.
    """
    if not isinstance(monetization, dict):
        return False
    billing = monetization.get("billing") or {}
    provider = None
    if isinstance(billing, dict):
        provider = (billing.get("provider") or "").strip().lower() or None
    if provider == "stripe":
        return True
    if provider and provider != "stripe":
        return False
    products = monetization.get("products")
    if isinstance(products, list) and products:
        return True
    return False


# ── scaffold templates (python-fastapi v1) ────────────────────────────────

_CHECKOUT_PY = '''"""Stripe Checkout session creation — scaffold (Z6 T5A).

Replace with real implementation during Phase 7 build. Wire the
``/checkout`` route into your app router.
"""
from __future__ import annotations

import os
from typing import Any

import stripe  # type: ignore  # add `stripe` to requirements.txt
from fastapi import APIRouter, HTTPException, Request


router = APIRouter()

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")


@router.post("/checkout/session")
async def create_checkout_session(request: Request) -> dict[str, Any]:
    """Create a Checkout Session for a price id provided by the client.

    Body: ``{"price_id": "price_...", "success_url": "...", "cancel_url": "..."}``
    """
    body = await request.json()
    price_id = body.get("price_id")
    if not price_id:
        raise HTTPException(status_code=400, detail="price_id required")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=body.get("success_url", "https://example.com/ok"),
            cancel_url=body.get("cancel_url", "https://example.com/cancel"),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"id": session.id, "url": session.url}
'''

_WEBHOOK_PY = '''"""Stripe webhook receiver — scaffold (Z6 T5A).

Replace with real implementation during Phase 7 build. Mount this
router and configure the endpoint URL in the Stripe dashboard. The
signature check below is the *bare minimum* — Stripe expects you to
verify ``Stripe-Signature`` on every incoming request.
"""
from __future__ import annotations

import os

import stripe  # type: ignore
from fastapi import APIRouter, Header, HTTPException, Request


router = APIRouter()

_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


@router.post("/webhook/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
) -> dict[str, str]:
    payload = await request.body()
    if not _WEBHOOK_SECRET:
        # Fail-closed during scaffold review so missing config is loud.
        raise HTTPException(status_code=503, detail="STRIPE_WEBHOOK_SECRET unset")
    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=stripe_signature or "",
            secret=_WEBHOOK_SECRET,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"signature: {exc}") from exc

    # TODO: dispatch on event.type — e.g. invoice.payment_succeeded,
    # customer.subscription.deleted, charge.dispute.created.
    _ = event  # placeholder
    return {"status": "received"}
'''

_ENV_EXAMPLE = """# Stripe — Z6 T5A scaffold
# Copy to .env and fill the test/sandbox values from https://dashboard.stripe.com
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
# Optional — only set when Stripe Tax is enabled for the account.
STRIPE_TAX_ORIGIN_COUNTRY=
"""

_README = """# Stripe scaffolding

Generated by `stripe_scaffold` (Z6 T5A).

## Files

- `api/checkout/create_session.py` — `/checkout/session` route stub.
- `api/webhook/stripe.py` — `/webhook/stripe` signature-verifying endpoint.
- `.env.example` — required env keys.

## Wire-up

1. `pip install stripe` (or `poetry add stripe`).
2. Include both routers in your FastAPI app:
   ```python
   from api.checkout.create_session import router as checkout_router
   from api.webhook.stripe import router as webhook_router
   app.include_router(checkout_router)
   app.include_router(webhook_router)
   ```
3. In the Stripe dashboard, set the webhook URL to `https://<your-host>/webhook/stripe`
   and copy the signing secret into `STRIPE_WEBHOOK_SECRET`.
4. The `stripe_provision_products` mechanical (Z6 T5B) creates the
   products + prices; reuse the returned `price_id` values in your
   client.

## Test

`payment_flow_test` (Z6 T5C, step 13.12) exercises the real test API
once products are provisioned. No manual test-card flow needed for CI.
"""


# ── main entrypoint ────────────────────────────────────────────────────────

async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Materialise the Stripe scaffold for ``task.mission_id``."""
    mission_id = task.get("mission_id")
    if mission_id is None:
        return {"ok": False, "reason": "missing_mission_id"}
    try:
        mission_id_int = int(mission_id)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "invalid_mission_id"}

    monetization = await _load_artifact_dict(mission_id_int, "monetization_strategy")
    if not _wants_stripe(monetization):
        return {
            "ok": True,
            "skipped": True,
            "reason": "no_stripe_billing",
            "files_written": [],
        }

    system_architecture = await _load_artifact_dict(
        mission_id_int, "system_architecture"
    )
    stack, ext = _detect_stack(system_architecture)

    root = _mission_workspace_root()
    base = os.path.join(root, f"mission_{mission_id_int}", "api")
    checkout_dir = os.path.join(base, "checkout")
    webhook_dir = os.path.join(base, "webhook")
    os.makedirs(checkout_dir, exist_ok=True)
    os.makedirs(webhook_dir, exist_ok=True)

    files_written: list[str] = []

    def _write(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        files_written.append(path)

    # Pick template by stack. Today: python-fastapi only.
    checkout_path = os.path.join(checkout_dir, f"create_session.{ext}")
    webhook_path = os.path.join(webhook_dir, f"stripe.{ext}")
    env_path = os.path.join(os.path.dirname(base), ".env.example")
    readme_path = os.path.join(os.path.dirname(base), "README_STRIPE.md")

    _write(checkout_path, _CHECKOUT_PY)
    _write(webhook_path, _WEBHOOK_PY)
    _write(env_path, _ENV_EXAMPLE)
    _write(readme_path, _README)

    logger.info(
        "stripe_scaffold wrote %d files mission=%s stack=%s",
        len(files_written), mission_id_int, stack,
    )
    return {
        "ok": True,
        "stack": stack,
        "files_written": files_written,
    }


__all__ = ["run"]
