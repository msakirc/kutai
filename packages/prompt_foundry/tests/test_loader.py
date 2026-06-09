from prompt_foundry.loader import get_profile, PROFILE_REGISTRY
from prompt_foundry.profile import Profile

def test_summarizer_loaded_from_yaml():
    p = get_profile("summarizer")
    assert isinstance(p, Profile)
    assert p.name == "summarizer"
    assert p.max_iterations == 3
    assert p.allowed_tools == ["read_file", "file_tree", "web_search"]
    assert p.get_system_prompt({}).startswith("You are a summarization specialist.")
    # literal backslash-n preserved (not a real newline) inside the JSON example:
    assert "\\n\\n### Key Points" in p.get_system_prompt({})

def test_singleton_identity():
    assert get_profile("summarizer") is get_profile("summarizer")

def test_registry_contains_summarizer():
    assert "summarizer" in PROFILE_REGISTRY
