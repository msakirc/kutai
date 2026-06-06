from src.agents import get_agent, AGENT_REGISTRY


def test_synthesizer_registered():
    assert "prior_art_synthesizer" in AGENT_REGISTRY
    assert get_agent("prior_art_synthesizer").name == "prior_art_synthesizer"


def test_synthesizer_prompt_invariants():
    p = get_agent("prior_art_synthesizer").get_system_prompt({})
    assert p.lstrip().startswith("You are ")
    low = p.lower()
    assert ("must" in low or "always" in low)
    assert ("don't" in low or "never" in low)
    assert "final_answer" in p
    assert "```json" in p


def test_synthesizer_prompt_forbids_inventing_urls():
    p = get_agent("prior_art_synthesizer").get_system_prompt({}).lower()
    assert "candidates" in p
    assert ("only" in p and "url" in p)
