"""Tests for remaining_budget_usd filter in Selector.select() and SelectionFailure type."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from fatih_hoca.selector import Selector


def _mk_model(name, cost_estimate):
    m = MagicMock()
    m.name = name
    m.estimated_cost_usd = cost_estimate
    m.is_eligible_for = MagicMock(return_value=True)
    m.is_local = cost_estimate == 0.0
    m.is_loaded = cost_estimate == 0.0
    m.provider = "local" if cost_estimate == 0.0 else "cloud"
    return m


def _make_selector(models):
    registry = MagicMock()
    registry.all_models.return_value = models
    nerd_herd = MagicMock()
    # Provide a realistic snapshot
    snap = MagicMock()
    snap.vram_available_mb = 8192
    snap.in_flight_calls = []
    snap.cloud = {}
    nerd_herd.snapshot.return_value = snap
    nerd_herd.recent_swap_count.return_value = 0
    return Selector(
        registry=registry,
        nerd_herd=nerd_herd,
        available_providers=None,
    )


def test_filters_models_above_remaining_budget(monkeypatch):
    """Models whose estimated_cost_usd > remaining_budget_usd are excluded before scoring."""
    models = [
        _mk_model("local-free", 0.0),
        _mk_model("cloud-cheap", 0.10),
        _mk_model("cloud-expensive", 5.00),
    ]
    selector = _make_selector(models)

    seen = []

    def fake_rank(candidates, **_kw):
        seen.append([m.name for m in candidates])
        result = MagicMock()
        result.model = candidates[0]
        result.score = 8.0
        result.urgency = 0.0
        result.pool = "local"
        result.capability_score = 0.8
        return [result]

    monkeypatch.setattr(
        "fatih_hoca.selector.rank_candidates",
        fake_rank,
    )

    # Patch eligibility to pass all
    monkeypatch.setattr(selector, "_check_eligibility", lambda **_kw: None)

    selector.select(task="test", remaining_budget_usd=0.50)

    assert seen, "rank_candidates not called"
    assert "cloud-expensive" not in seen[0], "expensive model should be filtered out"
    assert "local-free" in seen[0]
    assert "cloud-cheap" in seen[0]


def test_zero_budget_keeps_only_free_models(monkeypatch):
    """remaining_budget_usd=0.0 keeps only models with estimated_cost_usd == 0.0."""
    models = [
        _mk_model("local-free", 0.0),
        _mk_model("cloud-any", 0.01),
    ]
    selector = _make_selector(models)

    seen = []

    def fake_rank(candidates, **_kw):
        seen.append([m.name for m in candidates])
        result = MagicMock()
        result.model = candidates[0]
        result.score = 5.0
        result.urgency = 0.0
        result.pool = "local"
        result.capability_score = 0.5
        return [result]

    monkeypatch.setattr("fatih_hoca.selector.rank_candidates", fake_rank)
    monkeypatch.setattr(selector, "_check_eligibility", lambda **_kw: None)

    selector.select(task="test", remaining_budget_usd=0.0)

    assert seen, "rank_candidates not called"
    assert seen[0] == ["local-free"]


def test_none_budget_no_filter(monkeypatch):
    """remaining_budget_usd=None means no budget filter — all eligible candidates pass."""
    models = [
        _mk_model("local-free", 0.0),
        _mk_model("cloud-any", 1000.0),
    ]
    selector = _make_selector(models)

    seen = []

    def fake_rank(candidates, **_kw):
        seen.append([m.name for m in candidates])
        result = MagicMock()
        result.model = candidates[0]
        result.score = 5.0
        result.urgency = 0.0
        result.pool = "local"
        result.capability_score = 0.5
        return [result]

    monkeypatch.setattr("fatih_hoca.selector.rank_candidates", fake_rank)
    monkeypatch.setattr(selector, "_check_eligibility", lambda **_kw: None)

    selector.select(task="test", remaining_budget_usd=None)

    assert seen, "rank_candidates not called"
    assert "cloud-any" in seen[0], "expensive model should NOT be filtered when budget=None"


def test_empty_pool_returns_failure(monkeypatch):
    """If budget filter empties the candidate pool, SelectionFailure(reason='budget') is returned."""
    models = [_mk_model("only-cloud", 0.50)]
    selector = _make_selector(models)

    # Pass eligibility for all
    monkeypatch.setattr(selector, "_check_eligibility", lambda **_kw: None)

    result = selector.select(task="test", remaining_budget_usd=0.10)

    from fatih_hoca.types import SelectionFailure
    assert isinstance(result, SelectionFailure), f"expected SelectionFailure, got {type(result)}"
    assert result.reason == "budget"
