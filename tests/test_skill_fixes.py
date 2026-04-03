import pytest
import re


def test_seed_skills_returned_without_success_count():
    """Seed skills (success_count=0) should be returned by find_relevant_skills."""
    from src.memory.skills import _skill_score

    seed_skill = {"success_count": 0, "failure_count": 0, "name": "weather_api_routing"}
    score = _skill_score(seed_skill)
    assert score > 0


def test_auto_skill_regex_escape():
    """Auto-captured skill patterns must not contain unescaped brackets."""
    title = "[0.1] raw_idea_intake"
    # Simulate the fix: strip i2p prefixes, then re.escape words
    clean_title = re.sub(r"\[\d+\.?\d*[a-z]?\]\s*", "", title)
    words = [re.escape(w.lower().strip(".,!?")) for w in clean_title.split() if len(w) >= 3]
    pattern = "|".join(words)

    compiled = re.compile(pattern)
    assert compiled is not None

    # Should NOT match single digits
    assert not re.search(pattern, "0")
    assert not re.search(pattern, "1")

    # Should match the actual words
    assert re.search(pattern, "raw_idea_intake")
