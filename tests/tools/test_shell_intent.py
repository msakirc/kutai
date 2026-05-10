"""Z10-T1B — run_shell records reversibility_intent in its log line."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools import shell


@pytest.mark.asyncio
async def test_run_shell_logs_reversibility_intent(monkeypatch) -> None:
    seen: dict = {}

    real_logger = shell.logger
    fake = MagicMock()

    def _capture(msg, **kwargs):
        # First call is "shell invocation" — capture it.
        if msg == "shell invocation":
            seen.update(kwargs)

    fake.info.side_effect = _capture
    fake.debug = real_logger.debug
    fake.warning = real_logger.warning
    fake.exception = real_logger.exception

    monkeypatch.setattr(shell, "logger", fake)
    monkeypatch.setattr(shell, "SANDBOX_MODE", "none")

    # SANDBOX_MODE=none short-circuits to a "skipped" string after our log.
    result = await shell.run_shell(
        "echo hello",
        timeout=5,
        reversibility_intent="irreversible",
    )

    assert seen.get("reversibility_intent") == "irreversible"
    # We don't assert on `result` shape — local docker availability varies
    # across hosts; the structured log line is the contract under test.
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_shell_default_intent_none(monkeypatch) -> None:
    seen: dict = {}
    fake = MagicMock()

    def _capture(msg, **kwargs):
        if msg == "shell invocation":
            seen.update(kwargs)

    fake.info.side_effect = _capture
    monkeypatch.setattr(shell, "logger", fake)
    monkeypatch.setattr(shell, "SANDBOX_MODE", "none")

    await shell.run_shell("echo hi", timeout=5)
    assert "reversibility_intent" in seen
    assert seen["reversibility_intent"] is None
