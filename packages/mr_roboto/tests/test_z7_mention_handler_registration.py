"""Z7 A11.r1 — mention domain must register a handler in the agent registry.

The handler-registry refactor (coulson.agent_handlers.registry) lets new
domains plug into the oncall_agent dispatch chain via
``register_handler('mention', ...)``. oncall_action.run() already searches
the 'mention' domain — but mention_polls.py never registered anything, so
the mention domain had ZERO runtime registrations and mention events could
only reach the executor via the direct cron path, never the agent dispatch.

These tests exercise the REAL registry: importing mention_polls must leave
a callable handler discoverable for the 'mention' domain, and dispatching a
'mention' verb through oncall_action.run() must reach that handler.
"""
from __future__ import annotations

import pytest


def test_importing_mention_polls_registers_a_mention_handler():
    """After importing mention_polls, the registry has a 'mention' handler.

    Exercises the real coulson.agent_handlers.registry — no mocking.
    """
    # Importing the module must perform its import-time registration.
    import mr_roboto.mention_polls  # noqa: F401
    from coulson.agent_handlers.registry import list_verbs, lookup_handler

    mention_verbs = list_verbs("mention")
    assert mention_verbs, (
        "the 'mention' domain has no registered handlers — mention events "
        "cannot route through the agent dispatch chain"
    )
    # Every registered verb must resolve to a real callable.
    for verb in mention_verbs:
        handler = lookup_handler("mention", verb)
        assert callable(handler), f"mention verb {verb!r} has no callable handler"


@pytest.mark.asyncio
async def test_oncall_action_dispatches_a_mention_verb(tmp_path, monkeypatch):
    """A 'mention' verb routed through oncall_action.run() reaches the handler.

    Drives the real oncall_action.run -> registry lookup -> mention handler
    path. The handler's poll work is intercepted at the mention_polls
    seam (poll_source) so the test does not hit the network — but the
    registry routing itself is fully real.
    """
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    import mr_roboto.mention_polls as _mp
    from coulson.agent_handlers.registry import list_verbs

    mention_verbs = sorted(list_verbs("mention"))
    assert mention_verbs, "no mention verb registered"
    verb = mention_verbs[0]

    # Intercept only the network-bound poll, not the registry routing.
    seen: dict = {}

    async def _fake_poll(source, product_id, product_name, config=None):
        seen["called"] = (source, product_id, product_name)
        return {"ingested": 1, "immediate": 0, "digest": 1, "silent": 0,
                "skipped": 0, "crisis_triggered": False}

    monkeypatch.setattr(_mp, "poll_source", _fake_poll)

    from mr_roboto.executors.oncall_action import run as oncall_run

    task = {
        "mission_id": 42,
        "payload": {
            "domain": "mention",
            "verb": verb,
            "params": {
                "source": "hn",
                "product_id": "acme",
                "product_name": "AcmeApp",
            },
        },
    }
    result = await oncall_run(task)
    assert result.get("status") not in ("refused_not_whitelisted", "failed"), (
        f"mention verb {verb!r} was not routed to a handler: {result}"
    )
    assert seen.get("called"), (
        "the registered mention handler never reached the poll seam"
    )
