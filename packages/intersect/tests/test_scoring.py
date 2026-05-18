"""Unit tests for intersect.scoring."""
import pytest

from intersect import scoring


def test_confidence_is_product_of_factors(fake_artifact):
    art = fake_artifact(score=0.8)
    conf = scoring.score_artifact(
        art, source_trust=0.9, owner_trust=1.0, hint_bonus=1.0,
    )
    assert conf == pytest.approx(0.8 * 0.9 * 1.0 * 1.0)


def test_hint_bonus_lifts_confidence(fake_artifact):
    art = fake_artifact(score=0.6)
    base = scoring.score_artifact(art, source_trust=1.0, owner_trust=1.0,
                                  hint_bonus=1.0)
    boosted = scoring.score_artifact(art, source_trust=1.0, owner_trust=1.0,
                                     hint_bonus=1.25)
    assert boosted > base
    assert boosted == pytest.approx(base * 1.25)


def test_confidence_clamped_to_unit_interval(fake_artifact):
    art = fake_artifact(score=1.0)
    conf = scoring.score_artifact(art, source_trust=1.0, owner_trust=1.0,
                                  hint_bonus=3.0)
    assert conf == 1.0


def test_hint_bonus_for_matching_recipe_hint(fake_artifact):
    art = fake_artifact(
        name="cc-pypackage",
        intent_keywords=["python", "package", "scaffold", "pyproject"],
    )
    bonus = scoring.compute_hint_bonus(art, recipe_hint="python package scaffold")
    assert bonus > 1.0


def test_hint_bonus_neutral_when_no_hint(fake_artifact):
    art = fake_artifact(intent_keywords=["django", "web"])
    assert scoring.compute_hint_bonus(art, recipe_hint=None) == 1.0


def test_hint_bonus_neutral_on_keyword_miss(fake_artifact):
    art = fake_artifact(intent_keywords=["matlab", "simulink"])
    assert scoring.compute_hint_bonus(art, recipe_hint="react frontend") == 1.0
