"""conftest for hallederiz_kadir tests.

Prevents aiosqlite _connection_worker_thread from keeping pytest alive after
test completion. record_call_tokens in caller.py opens a persistent aiosqlite
connection on first call; in unit-test environments the DB is not set up and
the connection's non-daemon thread blocks process exit for minutes.
"""
from unittest.mock import AsyncMock, patch
import asyncio
import pytest


@pytest.fixture(autouse=True)
def _no_db_telemetry():
    """Stub out DB telemetry to prevent aiosqlite threads from blocking exit."""
    from unittest.mock import MagicMock
    with patch("src.infra.db.record_call_tokens", new=AsyncMock()), \
         patch("src.infra.db.audit_log", new=AsyncMock(), create=True), \
         patch("src.infra.audit.audit", new=AsyncMock(), create=True), \
         patch("src.infra.metrics.track_model_call_metrics", new=MagicMock(), create=True):
        yield


@pytest.fixture(scope="session", autouse=True)
def _close_db_connection_on_exit():
    """Close any lingering aiosqlite connection after the session to release threads."""
    yield
    try:
        import src.infra.db as _db
        conn = _db._db_connection
        if conn is not None:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(conn.close())
            loop.close()
            _db._db_connection = None
    except Exception:
        pass
