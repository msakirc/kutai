"""WS-1 (handoff 2026-05-25) — kill litellm's leaked aiohttp ClientSession.

Production symptom: recurring `🟠 [ERROR] asyncio: Unclosed client session` in
the orchestrator process (kutai.jsonl), 14 distinct sessions over 3 days,
clustered around LLM `begin_call` events. Hunt ruled out all 4 of KutAI's own
non-`async with` ClientSession sites — the leak is litellm's aiohttp transport
(``litellm/llms/custom_httpx/aiohttp_handler.py``, litellm issue #12443): it
creates aiohttp ClientSessions whose async close cannot run reliably from
``__del__``, so aiohttp's own finalizer logs the warning at GC time.

litellm 1.81.x defaults ``disable_aiohttp_transport=False`` (aiohttp on). Setting
it True makes litellm use its pooled httpx transport, which it closes properly —
no aiohttp sessions are ever created. Importing the caller module must flip the
flag so the whole orchestrator process inherits it before any LLM call.
"""
import litellm

# Import for its module-level side effect (the config block at caller.py top).
import hallederiz_kadir.caller  # noqa: F401


def test_aiohttp_transport_disabled_after_import():
    # If this regresses, the orchestrator leaks an aiohttp ClientSession per
    # litellm async-client (re)creation and spams asyncio ERROR at GC time.
    assert litellm.disable_aiohttp_transport is True
