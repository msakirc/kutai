"""prompt_writer is a Foundry data profile (image-gen Plan 3).

The agent config + system prompt live in
``packages/finch/src/finch/profiles/prompt_writer.yaml`` and resolve via
``get_agent("prompt_writer")`` (a finch ``Profile``). ``src/agents/prompt_writer.py``
retains only the artifact schema + template loader. These tests assert the
same invariants the class-backed agent used to guarantee, now against the
profile.
"""
from finch import Profile, PROFILE_REGISTRY
from src.agents import AGENT_REGISTRY, get_agent


def test_registered():
    assert "prompt_writer" in AGENT_REGISTRY
    assert "prompt_writer" in PROFILE_REGISTRY
    inst = get_agent("prompt_writer")
    assert isinstance(inst, Profile)
    assert inst.name == "prompt_writer"


def test_pure_config():
    # The agent is pure data (a profile); the legacy class is gone. The .py
    # module retains only the artifact schema + template loader — no agent class.
    body = open("src/agents/prompt_writer.py", encoding="utf-8").read()
    assert "class PromptWriterAgent" not in body
    assert "PROMPT_WRITER_ARTIFACT_SCHEMA" in body


def test_config_fields():
    a = get_agent("prompt_writer")
    assert a.name == "prompt_writer"
    assert a.default_tier == "cheap"
    assert a.max_iterations == 1
    assert a.enable_self_reflection is False
    assert a.allowed_tools == []


def test_system_prompt_satisfies_three_invariants():
    p = get_agent("prompt_writer").get_system_prompt({})
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
    p = get_agent("prompt_writer").get_system_prompt({})
    for slot in ("design_tokens", "brand_voice", "section_intent"):
        assert slot in p.lower()
