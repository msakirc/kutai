"""S12 — fleet-relative pool-balance signal."""
from types import SimpleNamespace

from nerd_herd.signals.s12_pool_balance import s12_pool_balance


def _free(provider: str):
    return SimpleNamespace(provider=provider, is_free=True, is_local=False)


def test_zero_at_cycle_start_no_consumption():
    """All providers at 0 consumed → no deficit → S12=0 (no spurious pull)."""
    fleet = {"gemini": 0.0, "groq": 0.0}
    assert s12_pool_balance(_free("gemini"), fleet_consumed=fleet) == 0.0


def test_pulls_toward_underused_provider():
    """groq has taken all the work; gemini none → gemini gets strong pull,
    groq gets none (it is over its fair share)."""
    fleet = {"gemini": 0.0, "groq": 100.0}
    gem = s12_pool_balance(_free("gemini"), fleet_consumed=fleet)
    grq = s12_pool_balance(_free("groq"), fleet_consumed=fleet)
    assert gem > 0.9          # gemini fully starved → near-max pull
    assert grq == 0.0         # groq over fair share → no pull


def test_pull_is_continuous_and_monotonic():
    """As a provider falls further behind fair share, its pull rises
    smoothly (no threshold/cliff)."""
    prev = -1.0
    for groq_consumed in (10, 30, 60, 100, 200):
        fleet = {"gemini": 0.0, "groq": float(groq_consumed)}
        val = s12_pool_balance(_free("gemini"), fleet_consumed=fleet)
        assert val >= prev      # monotonic non-decreasing
        prev = val
    assert prev > 0.9


def test_partial_deficit_between_zero_and_one():
    """A provider modestly below fair share gets a partial (not saturated,
    not zero) pull — proves the ramp is graded, not binary."""
    # fair = (20+40)/2 = 30; gemini consumed 20 → deficit (30-20)/30 = 0.33
    fleet = {"gemini": 20.0, "groq": 40.0}
    val = s12_pool_balance(_free("gemini"), fleet_consumed=fleet)
    assert 0.0 < val < 1.0


def test_paid_and_local_get_zero():
    """Only free (time_bucketed) quota perishes at a reset — paid/local
    models are not load-balanced by S12."""
    fleet = {"gemini": 0.0, "groq": 100.0}
    paid = SimpleNamespace(provider="anthropic", is_free=False, is_local=False)
    local = SimpleNamespace(provider="local", is_free=False, is_local=True)
    assert s12_pool_balance(paid, fleet_consumed=fleet) == 0.0
    assert s12_pool_balance(local, fleet_consumed=fleet) == 0.0


def test_single_provider_or_empty_no_signal():
    """Nothing to balance across → 0. Also guards the None default used by
    pressure-only unit tests."""
    assert s12_pool_balance(_free("gemini"), fleet_consumed={"gemini": 50.0}) == 0.0
    assert s12_pool_balance(_free("gemini"), fleet_consumed={}) == 0.0
    assert s12_pool_balance(_free("gemini"), fleet_consumed=None) == 0.0


def test_provider_not_in_fleet_no_signal():
    """A free model whose provider was dropped from the fleet map (e.g.
    fully exhausted → no capacity) gets no pull."""
    fleet = {"groq": 100.0, "openrouter": 0.0}
    assert s12_pool_balance(_free("gemini"), fleet_consumed=fleet) == 0.0
