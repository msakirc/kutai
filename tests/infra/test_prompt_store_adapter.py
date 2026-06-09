import pytest
from src.infra.prompt_store_adapter import DbPromptStore


@pytest.mark.asyncio
async def test_adapter_satisfies_protocol():
    from prompt_foundry.store import PromptStore
    assert isinstance(DbPromptStore(), PromptStore)


@pytest.mark.asyncio
async def test_get_active_delegates_to_prompt_versions(monkeypatch):
    called = {}

    async def fake_get_active_prompt(agent_type):
        called["arg"] = agent_type
        return "DBVAL"

    monkeypatch.setattr("src.memory.prompt_versions.get_active_prompt", fake_get_active_prompt)
    out = await DbPromptStore().get_active("coder")
    assert out == "DBVAL"
    assert called["arg"] == "coder"
