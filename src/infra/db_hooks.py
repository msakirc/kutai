"""Wire app-level services into the dabidabi engine's hook registry.

The engine (``dabidabi``) must not import ``src.*``. Instead it exposes
``dabidabi.hooks`` and calls registered callables. This module owns the
app→engine wiring: thin wrappers that lazy-import the real ``src.*``
implementation on first call (so ``wire()`` itself stays cheap and
import-cycle-free), registered via ``dabidabi.hooks.register``.

``wire()`` is called once at orchestrator startup (``src/app/run.py``).
Idempotent — re-registering just rebinds the same wrappers.
"""
import dabidabi


async def _ensure_mission_container(mission_id: int):
    # Engine is sandbox-mode agnostic; the gate lives here.
    from src.tools import shell
    if shell.SANDBOX_MODE not in ("none", "local"):
        return await shell.ensure_mission_container(mission_id)
    return None


async def _embed_and_store(**kwargs):
    from src.memory.vector_store import embed_and_store
    return await embed_and_store(**kwargs)


async def _vector_query(**kwargs):
    from src.memory.vector_store import query
    return await query(**kwargs)


async def _purge_mission_chroma(mission_id: int) -> int:
    from src.memory.vector_store import purge_mission_chroma_collections
    return await purge_mission_chroma_collections(mission_id)


async def _quarantine_task(**kwargs):
    from src.infra.dead_letter import quarantine_task
    return await quarantine_task(**kwargs)


def wire() -> None:
    """Register all engine→app service hooks. Call once at startup."""
    dabidabi.hooks.register(
        ensure_mission_container=_ensure_mission_container,
        embed_and_store=_embed_and_store,
        vector_query=_vector_query,
        purge_mission_chroma=_purge_mission_chroma,
        quarantine_task=_quarantine_task,
    )
