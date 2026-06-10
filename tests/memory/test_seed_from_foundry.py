"""Task 15: seed_from_agents reads the Foundry registry, not AGENT_REGISTRY."""
import pytest
from src.memory.prompt_versions import seed_from_agents


@pytest.mark.asyncio
async def test_seed_reads_foundry(monkeypatch):
    saved = {}

    async def fake_get_active(at):
        return None

    async def fake_save(agent_type, prompt_text, notes="", activate=False):
        saved[agent_type] = prompt_text
        return 1

    monkeypatch.setattr("src.memory.prompt_versions.get_active_prompt", fake_get_active)
    monkeypatch.setattr("src.memory.prompt_versions.save_prompt_version", fake_save)

    n = await seed_from_agents()
    assert n >= 27, f"expected at least 27 seeded, got {n}"
    assert "summarizer" in saved
    assert saved["summarizer"].startswith("You are a summarization")
