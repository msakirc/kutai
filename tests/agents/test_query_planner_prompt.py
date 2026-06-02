from src.agents import get_agent, AGENT_REGISTRY


def test_query_planner_registered():
    assert "query_planner" in AGENT_REGISTRY
    agent = get_agent("query_planner")
    assert agent.name == "query_planner"


def test_query_planner_prompt_invariants():
    agent = get_agent("query_planner")
    p = agent.get_system_prompt({})
    assert p.lstrip().startswith("You are ")
    low = p.lower()
    assert ("must" in low or "always" in low)
    assert ("don't" in low or "never" in low)
    assert "final_answer" in p
    assert "```json" in p


def test_query_planner_no_fetch_tools():
    agent = get_agent("query_planner")
    assert "find_prior_art" not in agent.allowed_tools
    assert "web_search" not in agent.allowed_tools
