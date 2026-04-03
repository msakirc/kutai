import pytest


def test_seed_skills_have_neutral_confidence():
    """Seed skills (0 injections) get neutral confidence score."""
    from src.memory.skills import _injection_success_rate
    skill = {"injection_count": 0, "injection_success": 0}
    rate = _injection_success_rate(skill)
    assert rate == 0.5  # neutral — not enough data


def test_proven_skill_confidence():
    """Skills with enough data reflect actual success rate."""
    from src.memory.skills import _injection_success_rate
    skill = {"injection_count": 10, "injection_success": 8}
    rate = _injection_success_rate(skill)
    assert rate == 0.8
