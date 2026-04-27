import pytest
from fatih_hoca.cloud.family import normalize


@pytest.mark.parametrize("provider,litellm_name,expected", [
    ("groq", "groq/llama-3.3-70b-versatile", "llama-3.3-70b"),
    ("cerebras", "cerebras/llama3.3-70b", "llama-3.3-70b"),
    ("sambanova", "sambanova/Meta-Llama-3.3-70B-Instruct", "llama-3.3-70b"),
    ("groq", "groq/llama-3.1-8b-instant", "llama-3.1-8b"),
    ("sambanova", "sambanova/Qwen3-32B", "qwen3-32b"),
    ("anthropic", "claude-sonnet-4-20250514", "claude-sonnet-4"),
    ("anthropic", "claude-3-5-sonnet-20241022", "claude-3.5-sonnet"),
    ("openai", "gpt-4o", "gpt-4o"),
    ("openai", "gpt-4o-mini", "gpt-4o-mini"),
    ("openai", "o1-preview", "o1-preview"),
    ("gemini", "gemini/gemini-2.0-flash", "gemini-2.0-flash"),
    ("gemini", "gemini/gemini-2.5-flash-preview-05-20", "gemini-2.5-flash"),
    ("openrouter", "openrouter/meta-llama/llama-3.3-70b-instruct", "llama-3.3-70b"),
])
def test_normalize_known_families(provider, litellm_name, expected):
    assert normalize(provider, litellm_name) == expected


def test_normalize_unknown_falls_back_to_litellm_name():
    out = normalize("groq", "groq/some-future-model-v9-2030")
    assert out == "some-future-model-v9-2030"


def test_normalize_unknown_marked():
    from fatih_hoca.cloud.family import normalize, is_known_family
    assert is_known_family("llama-3.3-70b") is True
    assert is_known_family("some-future-model-v9-2030") is False
