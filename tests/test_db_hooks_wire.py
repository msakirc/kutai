"""App→engine hook wiring (Phase B §5a). wire() registers the lazy service
wrappers into dabidabi.hooks; the engine calls them instead of importing
src.*. Wrappers lazy-import the real impl on first call, so this test stays
cheap (no chromadb/shell import)."""
from dabidabi import hooks
from src.infra import db_hooks


def test_wire_registers_all_service_hooks():
    hooks.reset()
    assert hooks.embed_and_store is None  # unset before wiring
    db_hooks.wire()
    assert hooks.ensure_mission_container is db_hooks._ensure_mission_container
    assert hooks.embed_and_store is db_hooks._embed_and_store
    assert hooks.vector_query is db_hooks._vector_query
    assert hooks.purge_mission_chroma is db_hooks._purge_mission_chroma
    assert hooks.quarantine_task is db_hooks._quarantine_task
    hooks.reset()


def test_wire_is_idempotent():
    hooks.reset()
    db_hooks.wire()
    db_hooks.wire()  # must not raise
    assert hooks.quarantine_task is db_hooks._quarantine_task
    hooks.reset()
