import pytest
import yaml
from finch import loader
from finch.loader import get_profile, PROFILE_REGISTRY
from finch.profile import Profile


def test_unknown_yaml_key_is_ignored(tmp_path, monkeypatch):
    """YAML files with unrecognised keys must load without error; unknown keys are silently dropped."""
    profile_yaml = tmp_path / "x.yaml"
    profile_yaml.write_text(
        'name: x\n'
        'system_prompt: "You are x. must never. final_answer ```json {} ```"\n'
        'bogus_key: 1\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "_PROFILES_DIR", tmp_path)
    result = loader._load_all()
    assert "x" in result
    p = result["x"]
    assert p.name == "x"
    assert not hasattr(p, "bogus_key")


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
