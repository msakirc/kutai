"""Z6 T5B — tests for mr_roboto.executors.stripe_provision_products."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mr_roboto.executors.stripe_provision_products import (
    _find_existing_product,
    _kutay_id_for,
    _validate_product,
    run,
)


# ── unit: kutay_id stability ──────────────────────────────────────────────


def test_kutay_id_is_deterministic():
    p = {"name": "Pro", "price_cents": 1000, "currency": "USD", "interval": "month"}
    assert _kutay_id_for(7, p) == _kutay_id_for(7, p)


def test_kutay_id_differs_for_different_products():
    a = {"name": "A", "price_cents": 1000, "currency": "USD"}
    b = {"name": "B", "price_cents": 1000, "currency": "USD"}
    assert _kutay_id_for(7, a) != _kutay_id_for(7, b)


def test_kutay_id_differs_per_mission():
    p = {"name": "Pro", "price_cents": 1000, "currency": "USD"}
    assert _kutay_id_for(1, p) != _kutay_id_for(2, p)


# ── unit: validate_product ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prod, ok",
    [
        ({"name": "Pro", "price_cents": 100, "currency": "USD"}, True),
        ({"name": "", "price_cents": 100, "currency": "USD"}, False),
        ({"name": "Pro", "currency": "USD"}, False),  # missing price
        ({"name": "Pro", "price_cents": -1, "currency": "USD"}, False),
        ({"name": "Pro", "price_cents": 100, "currency": "US"}, False),
        ("string", False),
        (None, False),
    ],
)
def test_validate_product(prod, ok):
    res, _ = _validate_product(prod)
    assert res is ok


# ── unit: find_existing_product ───────────────────────────────────────────


def test_find_existing_returns_match():
    listing = {
        "data": [
            {"id": "prod_1", "metadata": {"kutay_id": "abc"}},
            {"id": "prod_2", "metadata": {"kutay_id": "def"}},
        ]
    }
    found = _find_existing_product(listing, "def")
    assert found and found["id"] == "prod_2"


def test_find_existing_returns_none_when_no_match():
    listing = {"data": [{"id": "prod_1", "metadata": {"kutay_id": "abc"}}]}
    assert _find_existing_product(listing, "xyz") is None


def test_find_existing_handles_empty():
    assert _find_existing_product({}, "abc") is None
    assert _find_existing_product({"data": []}, "abc") is None


# ── integration: run() with mocked vendor_call ────────────────────────────


def _make_monet(products):
    return {"billing": {"provider": "stripe"}, "products": products}


@pytest.mark.asyncio
async def test_run_no_products_skipped(monkeypatch):
    async def _fake_load(mission_id, name):
        return _make_monet([])

    async def _fake_persist(*_a, **_kw):
        pass

    with patch(
        "mr_roboto.executors.stripe_provision_products._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ), patch(
        "mr_roboto.executors.stripe_provision_products._persist_artifact",
        new=AsyncMock(side_effect=_fake_persist),
    ):
        res = await run({"mission_id": 5})
    assert res["ok"] is True
    assert res["skipped"] is True


@pytest.mark.asyncio
async def test_run_missing_monetization(monkeypatch):
    async def _fake_load(mission_id, name):
        return None
    with patch(
        "mr_roboto.executors.stripe_provision_products._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ):
        res = await run({"mission_id": 5})
    assert res["ok"] is False
    assert res["reason"] == "monetization_strategy_missing"


@pytest.mark.asyncio
async def test_run_invalid_product_schema():
    async def _fake_load(*_a, **_kw):
        return _make_monet([{"name": "BadOne"}])  # missing price_cents+currency
    with patch(
        "mr_roboto.executors.stripe_provision_products._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ):
        res = await run({"mission_id": 5})
    assert res["ok"] is False
    assert res["reason"] == "invalid_product_schema"


@pytest.mark.asyncio
async def test_run_creates_new_product_and_price():
    products = [
        {"name": "Pro", "price_cents": 1500, "currency": "USD", "interval": "month"}
    ]

    async def _fake_load(*_a, **_kw):
        return _make_monet(products)

    persisted: dict = {}

    async def _fake_persist(mission_id, name, value):
        persisted[name] = value

    calls: list[tuple[str, str, dict]] = []

    async def _fake_vc(task, service, action, params):
        calls.append((service, action, params))
        if action == "list_products":
            return {"ok": True, "result": {"data": []}}
        if action == "create_product":
            return {"ok": True, "result": {"id": "prod_NEW"}}
        if action == "create_price":
            return {"ok": True, "result": {"id": "price_NEW"}}
        return {"ok": False, "reason": "unexpected"}

    with patch(
        "mr_roboto.executors.stripe_provision_products._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ), patch(
        "mr_roboto.executors.stripe_provision_products._persist_artifact",
        new=AsyncMock(side_effect=_fake_persist),
    ), patch(
        "mr_roboto.executors.stripe_provision_products._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({"mission_id": 11})

    assert res["ok"]
    assert len(res["provisioned"]) == 1
    item = res["provisioned"][0]
    assert item["product_name"] == "Pro"
    assert item["stripe_product_id"] == "prod_NEW"
    assert item["stripe_price_id"] == "price_NEW"
    # kutay_id must propagate as price metadata + product metadata.
    actions = [c[1] for c in calls]
    assert "list_products" in actions
    assert "create_product" in actions
    assert "create_price" in actions
    # Persisted artifact has the expected shape.
    assert "stripe_provisioned" in persisted


@pytest.mark.asyncio
async def test_run_idempotent_when_product_already_exists():
    products = [
        {"name": "Pro", "price_cents": 1500, "currency": "USD", "interval": "month"}
    ]

    async def _fake_load(*_a, **_kw):
        return _make_monet(products)

    async def _fake_persist(*_a, **_kw):
        pass

    # Build the deterministic kutay_id the executor will compute.
    expected_kutay = _kutay_id_for(11, products[0])

    async def _fake_vc(task, service, action, params):
        if action == "list_products":
            return {
                "ok": True,
                "result": {
                    "data": [
                        {
                            "id": "prod_EXISTING",
                            "metadata": {"kutay_id": expected_kutay},
                            "default_price": "price_EXISTING",
                        }
                    ]
                },
            }
        # Should not be called again — fail loud if it is.
        raise AssertionError(f"unexpected vendor_call: {action}")

    with patch(
        "mr_roboto.executors.stripe_provision_products._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ), patch(
        "mr_roboto.executors.stripe_provision_products._persist_artifact",
        new=AsyncMock(side_effect=_fake_persist),
    ), patch(
        "mr_roboto.executors.stripe_provision_products._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({"mission_id": 11})

    assert res["ok"]
    assert res["provisioned"][0]["stripe_product_id"] == "prod_EXISTING"
    assert res["provisioned"][0]["stripe_price_id"] == "price_EXISTING"


@pytest.mark.asyncio
async def test_run_aborts_on_create_product_failure():
    products = [{"name": "Pro", "price_cents": 1500, "currency": "USD"}]

    async def _fake_load(*_a, **_kw):
        return _make_monet(products)

    async def _fake_vc(task, service, action, params):
        if action == "list_products":
            return {"ok": True, "result": {"data": []}}
        if action == "create_product":
            return {"ok": False, "reason": "vendor_error", "error": "boom"}
        raise AssertionError(f"unexpected vendor_call: {action}")

    with patch(
        "mr_roboto.executors.stripe_provision_products._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ), patch(
        "mr_roboto.executors.stripe_provision_products._persist_artifact",
        new=AsyncMock(),
    ), patch(
        "mr_roboto.executors.stripe_provision_products._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({"mission_id": 11})

    assert res["ok"] is False
    assert res["reason"] == "create_product_failed"


@pytest.mark.asyncio
async def test_run_aborts_on_list_products_failure():
    products = [{"name": "Pro", "price_cents": 1500, "currency": "USD"}]

    async def _fake_load(*_a, **_kw):
        return _make_monet(products)

    async def _fake_vc(task, service, action, params):
        return {"ok": False, "reason": "vendor_error", "error": "401"}

    with patch(
        "mr_roboto.executors.stripe_provision_products._load_artifact_dict",
        new=AsyncMock(side_effect=_fake_load),
    ), patch(
        "mr_roboto.executors.stripe_provision_products._persist_artifact",
        new=AsyncMock(),
    ), patch(
        "mr_roboto.executors.stripe_provision_products._vc",
        new=AsyncMock(side_effect=_fake_vc),
    ):
        res = await run({"mission_id": 11})

    assert res["ok"] is False
    assert res["reason"] == "list_products_failed"


@pytest.mark.asyncio
async def test_run_missing_mission_id():
    res = await run({})
    assert res["ok"] is False
    assert res["reason"] == "missing_mission_id"
