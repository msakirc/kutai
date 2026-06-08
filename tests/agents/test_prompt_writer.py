import re
from src.agents.prompt_writer import PromptWriterAgent
from src.agents import AGENT_REGISTRY, get_agent


def test_registered():
    assert "prompt_writer" in AGENT_REGISTRY
    inst = get_agent("prompt_writer")
    assert isinstance(inst, PromptWriterAgent)


def test_pure_config():
    body = open("src/agents/prompt_writer.py", encoding="utf-8").read()
    methods = re.findall(r"^    def (\w+)\(", body, flags=re.MULTILINE)
    assert set(methods) <= {"get_system_prompt"}, methods


def test_config_fields():
    a = PromptWriterAgent()
    assert a.name == "prompt_writer"
    assert a.default_tier == "cheap"
    assert a.max_iterations == 1
    assert a.enable_self_reflection is False
    assert a.allowed_tools == []


def test_system_prompt_satisfies_three_invariants():
    p = PromptWriterAgent().get_system_prompt({})
    first_line = p.strip().splitlines()[0]
    assert first_line.startswith("You are "), first_line
    body = p.lower()
    assert "must" in body or "always" in body
    assert "don't" in body or "never" in body
    assert "final_answer" in p
    assert "```json" in p
    assert "placeholder_id" in p
    assert "_schema_version" in p


def test_system_prompt_mentions_template_slots():
    p = PromptWriterAgent().get_system_prompt({})
    for slot in ("design_tokens", "brand_voice", "section_intent"):
        assert slot in p.lower()
