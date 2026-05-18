"""yalayut.admin — Phase 3 auth + MCP control ops."""
import pytest

from yalayut import admin


@pytest.mark.asyncio
async def test_set_secret_writes_and_revets(monkeypatch):
    written = {}
    revetted = []

    async def fake_set_secret(key, value):
        written[key] = value

    async def fake_revet(key):
        revetted.append(key)

    monkeypatch.setattr("yalayut.secrets.set_secret", fake_set_secret)
    monkeypatch.setattr(admin, "_revet_artifacts_for_env", fake_revet)

    res = await admin.set_secret("OPENAQ_API_KEY", "the-value")
    assert res["ok"] is True
    assert written["OPENAQ_API_KEY"] == "the-value"
    assert "OPENAQ_API_KEY" in revetted


@pytest.mark.asyncio
async def test_missing_auth_lists_blocked(monkeypatch):
    async def fake_query(sql, params=()):
        return [(7, "api-virustotal", "missing_VIRUSTOTAL_API_KEY"),
                (8, "mcp-cloudflare", "missing_CLOUDFLARE_API_TOKEN")]

    monkeypatch.setattr(admin, "_db_query", fake_query)
    rows = await admin.missing_auth()
    assert len(rows) == 2
    assert rows[0]["name"] == "api-virustotal"
    assert rows[0]["missing_key"] == "VIRUSTOTAL_API_KEY"


@pytest.mark.asyncio
async def test_mcp_status_reports_manager(monkeypatch):
    from yalayut.mcp_manager import get_manager

    monkeypatch.setattr(get_manager(), "status",
                        lambda: [{"artifact_id": 9, "pid": 1234,
                                  "health": "ready", "fails": 0,
                                  "last_probe": 1.0}])
    rows = await admin.mcp_status()
    assert rows[0]["artifact_id"] == 9
    assert rows[0]["health"] == "ready"


@pytest.mark.asyncio
async def test_mcp_kill(monkeypatch):
    from yalayut.mcp_manager import get_manager

    killed = []

    async def fake_shutdown(aid):
        killed.append(aid)

    monkeypatch.setattr(get_manager(), "shutdown", fake_shutdown)
    res = await admin.mcp_kill(9)
    assert res["ok"] is True
    assert killed == [9]
