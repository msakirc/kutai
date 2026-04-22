"""Pytest wrapper around the admission sim scenarios."""
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
_TESTS_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)

from adm_sim.runner import run_ticks  # noqa: E402
from adm_sim.scenarios import SCENARIOS  # noqa: E402


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
@pytest.mark.asyncio
async def test_admission_scenario(scenario):
    state = scenario.state_factory()
    metrics = await run_ticks(state, ticks=scenario.ticks)
    err = scenario.assertion(metrics, state)
    assert err is None, err
