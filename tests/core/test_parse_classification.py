"""parse_classification: pure (no LLM, no Beckman) mapping of an LLM result
dict -> TaskClassification. This is the 'intelligence' extracted from the old
inline _classify_with_llm post-await block so it is testable synchronously."""
from src.core.task_classifier import parse_classification, TaskClassification


def test_parse_basic_fields():
    cls = parse_classification(
        {"content": '{"agent_type": "coder", "difficulty": 7, "needs_tools": true}'},
        title="build a parser", description="write a JSON parser module",
    )
    assert isinstance(cls, TaskClassification)
    assert cls.agent_type == "coder"
    assert cls.difficulty == 7
    assert cls.needs_tools is True
    assert cls.method == "llm"


def test_parse_clamps_difficulty():
    cls = parse_classification(
        {"content": '{"agent_type": "executor", "difficulty": 99}'},
        title="x", description="y",
    )
    assert cls.difficulty == 10


def test_parse_vision_guarded_to_visual_reviewer():
    # needs_vision only honored when agent_type == visual_reviewer
    cls = parse_classification(
        {"content": '{"agent_type": "coder", "needs_vision": true}'},
        title="ui work", description="design the layout",
    )
    assert cls.needs_vision is False


def test_parse_shopping_sub_intent_attached():
    cls = parse_classification(
        {"content": '{"agent_type": "shopping_advisor"}'},
        title="coffee machine", description="en ucuz kahve makinesi",
    )
    assert cls.agent_type == "shopping_advisor"
    assert cls.shopping_sub_intent is not None


def test_parse_falls_back_to_keywords_on_bad_json():
    cls = parse_classification(
        {"content": "not json at all"},
        title="fix the bug", description="error in auth",
    )
    # bad JSON -> keyword fallback, still a valid classification
    assert cls.method == "keyword"
    assert cls.agent_type == "fixer"


def test_parse_content_list_parts_coalesced():
    cls = parse_classification(
        {"content": [{"type": "text", "text": '{"agent_type": "planner"'},
                     {"type": "text", "text": ', "difficulty": 5}'}]},
        title="plan it", description="roadmap",
    )
    assert cls.agent_type == "planner"
