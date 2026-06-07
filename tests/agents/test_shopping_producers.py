"""Producer-agent tests for the shopping_v3 migration (grouper/labeler/synthesizer
+ the shopping_clarifier single_shot -> react flip)."""
from src.agents.shopping_grouper import ShoppingGrouperAgent
from src.agents.shopping_labeler import ShoppingLabelerAgent
from src.agents.shopping_synthesizer import ShoppingSynthesizerAgent
from src.agents.shopping_clarifier import ShoppingClarifierAgent


def _assert_invariants(prompt: str):
    assert prompt.lstrip().startswith("You are ")
    low = prompt.lower()
    assert ("must" in low or "always" in low)
    assert ("don't" in low or "never" in low or "do not" in low)
    assert "final_answer" in low
    assert "```json" in prompt


def test_grouper_prompt_invariants():
    _assert_invariants(ShoppingGrouperAgent().get_system_prompt({}))


def test_grouper_is_prompt_only_react():
    a = ShoppingGrouperAgent()
    assert a.name == "shopping_grouper"
    assert a.allowed_tools == []
    assert getattr(a, "execution_pattern", "react_loop") == "react_loop"


def test_labeler_prompt_invariants():
    p = ShoppingLabelerAgent().get_system_prompt({})
    _assert_invariants(p)
    low = p.lower()
    assert "line_id" in low and "base_model" in low and "product_type" in low


def test_labeler_is_prompt_only_react():
    a = ShoppingLabelerAgent()
    assert a.name == "shopping_labeler"
    assert a.allowed_tools == []
    assert getattr(a, "execution_pattern", "react_loop") == "react_loop"


def test_synthesizer_prompt_invariants():
    p = ShoppingSynthesizerAgent().get_system_prompt({})
    _assert_invariants(p)
    low = p.lower()
    assert "aspects" in low and "insufficient_data" in low


def test_synthesizer_is_prompt_only_react():
    a = ShoppingSynthesizerAgent()
    assert a.name == "shopping_synthesizer"
    assert a.allowed_tools == []
    assert getattr(a, "execution_pattern", "react_loop") == "react_loop"


# ── T9: clarifier single_shot -> react + canonical clarify action shape ──────

def test_clarifier_is_react_not_single_shot():
    # single_shot drops the clarify question (returns result="") — the workflow
    # clarify pause only fires when the react loop emits status=needs_clarification
    # off a canonical action:"clarify".
    a = ShoppingClarifierAgent()
    assert getattr(a, "execution_pattern", "react_loop") == "react_loop"


def test_clarifier_emits_canonical_clarify_action():
    # react.py:1651 keys on action=="clarify" with the question under "question";
    # the old action:"needs_clarification"/"clarification" shape was consumed by
    # neither runner. The prompt must instruct the canonical shape.
    p = ShoppingClarifierAgent().get_system_prompt({})
    assert '"action": "clarify"' in p
    assert '"question"' in p
    # Path A still returns the query as a normal final_answer
    assert "final_answer" in p
