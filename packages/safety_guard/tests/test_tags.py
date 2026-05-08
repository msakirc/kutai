import pytest
from safety_guard.tags import Reversibility, resolve


def test_default_full_when_unspecified():
    assert resolve({}, None) == Reversibility.FULL


def test_static_returned_when_no_runtime_override():
    step = {"reversibility": "partial"}
    assert resolve(step, None) == Reversibility.PARTIAL


def test_locked_ignores_runtime_override():
    step = {"reversibility": "none", "locked": True}
    assert resolve(step, Reversibility.FULL) == Reversibility.NONE
    assert resolve(step, Reversibility.PARTIAL) == Reversibility.NONE


def test_runtime_escalation_accepted():
    step = {"reversibility": "full"}
    assert resolve(step, Reversibility.PARTIAL) == Reversibility.PARTIAL
    assert resolve(step, Reversibility.NONE) == Reversibility.NONE


def test_runtime_downgrade_rejected():
    step = {"reversibility": "none"}
    assert resolve(step, Reversibility.FULL) == Reversibility.NONE
    assert resolve(step, Reversibility.PARTIAL) == Reversibility.NONE


def test_unknown_reversibility_value_falls_back_to_full(caplog):
    step = {"reversibility": "garbage"}
    with caplog.at_level("WARNING"):
        assert resolve(step, None) == Reversibility.FULL
    assert any("garbage" in m for m in caplog.messages)
