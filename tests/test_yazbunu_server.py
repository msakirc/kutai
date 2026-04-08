"""Tests for yazbunu log viewer server API."""
import asyncio
import json
import os
import sys
import time

import pytest
import pytest_asyncio
from aiohttp import web, WSMsgType
from aiohttp.test_utils import AioHTTPTestCase, TestClient

from yazbunu.server import create_app


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def log_dir(tmp_path):
    """Create a temp log dir with sample JSONL data (5000 lines, every 10th ERROR)."""
    lines = []
    for i in range(5000):
        level = "ERROR" if i % 10 == 0 else "INFO"
        doc = {
            "ts": f"2026-04-06T12:{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}.000Z",
            "level": level,
            "src": "kutai.core.orchestrator" if i % 3 != 0 else "kutai.agents.base",
            "msg": f"line {i}",
            "task": str(i % 100),
        }
        lines.append(json.dumps(doc))
    (tmp_path / "kutai.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "shopping.jsonl").write_text(
        json.dumps({"ts": "2026-04-06T13:00:00.000Z", "level": "INFO",
                     "src": "shopping.scraper", "msg": "scrape done"}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def auth_dir(tmp_path):
    """Create a temp auth dir."""
    d = tmp_path / "auth"
    d.mkdir()
    return d


@pytest.fixture
def log_dir_with_context(tmp_path):
    """Create log dir with lines containing specific context fields."""
    lines = []
    for i in range(50):
        doc = {
            "ts": f"2026-04-06T14:00:{i:02d}.000Z",
            "level": "INFO",
            "src": "kutai.core.orchestrator",
            "msg": f"context line {i}",
            "task": "42" if i % 5 == 0 else str(i),
            "mission": "m-7" if i % 10 == 0 else "m-1",
        }
        lines.append(json.dumps(doc))
    (tmp_path / "ctx.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


@pytest_asyncio.fixture
async def client(log_dir, aiohttp_client):
    app = create_app(str(log_dir))
    return await aiohttp_client(app)


@pytest_asyncio.fixture
async def auth_client(log_dir, auth_dir, aiohttp_client):
    app = create_app(str(log_dir), auth_dir=str(auth_dir), require_auth=True)
    return await aiohttp_client(app)


@pytest_asyncio.fixture
async def context_client(log_dir_with_context, aiohttp_client):
    app = create_app(str(log_dir_with_context))
    return await aiohttp_client(app)


# ─── Existing tests (kept) ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_files(client):
    resp = await client.get("/api/files")
    assert resp.status == 200
    data = await resp.json()
    names = [f["name"] for f in data["files"]]
    assert "kutai.jsonl" in names
    assert "shopping.jsonl" in names


@pytest.mark.asyncio
async def test_file_not_found(client):
    resp = await client.get("/api/logs?file=nonexistent.jsonl")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_path_traversal_blocked(client):
    resp = await client.get("/api/logs?file=../../../etc/passwd")
    assert resp.status == 400


# ─── Task 1: Streaming file reads ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_logs_returns_last_n_lines(client):
    """Verify last N lines returned correctly."""
    resp = await client.get("/api/logs?file=kutai.jsonl&lines=5")
    assert resp.status == 200
    data = await resp.json()
    assert len(data["lines"]) == 5
    # Should be the LAST 5 lines
    last = json.loads(data["lines"][-1])
    assert last["msg"] == "line 4999"
    first = json.loads(data["lines"][0])
    assert first["msg"] == "line 4995"


@pytest.mark.asyncio
async def test_get_logs_default_1000_lines(client):
    """Verify default limit is 1000."""
    resp = await client.get("/api/logs?file=kutai.jsonl")
    assert resp.status == 200
    data = await resp.json()
    assert len(data["lines"]) == 1000
    assert data["total"] == 5000


@pytest.mark.asyncio
async def test_tail_returns_lines_after_timestamp(client):
    """Verify time filtering works with substring extraction."""
    # Lines with ts > this timestamp
    resp = await client.get("/api/tail?file=kutai.jsonl&after=2026-04-06T12:01:23:50.000Z")
    assert resp.status == 200
    data = await resp.json()
    for line_str in data["lines"]:
        doc = json.loads(line_str)
        assert doc["ts"] > "2026-04-06T12:01:23:50.000Z"


# ─── Task 3: /health endpoint ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_token_and_url(log_dir, auth_dir, aiohttp_client):
    """Verify /health response contains expected fields."""
    app = create_app(str(log_dir), auth_dir=str(auth_dir), require_auth=True)
    client = await aiohttp_client(app)
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.2.0"
    assert "uptime_sec" in data
    assert isinstance(data["uptime_sec"], float)
    assert len(data["token"]) > 0
    assert data["token"] in data["url"]


# ─── Task 7: Auth middleware ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auth_blocks_without_token(auth_client):
    """Verify 401 when require_auth=True and no token provided."""
    resp = await auth_client.get("/api/files")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_auth_allows_with_query_token(log_dir, auth_dir, aiohttp_client):
    """Verify token in query param works."""
    from yazbunu.auth import get_or_create_token
    token = get_or_create_token(str(auth_dir))
    app = create_app(str(log_dir), auth_dir=str(auth_dir), require_auth=True)
    client = await aiohttp_client(app)
    resp = await client.get(f"/api/files?token={token}")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_auth_allows_with_bearer_header(log_dir, auth_dir, aiohttp_client):
    """Verify Bearer token in Authorization header works."""
    from yazbunu.auth import get_or_create_token
    token = get_or_create_token(str(auth_dir))
    app = create_app(str(log_dir), auth_dir=str(auth_dir), require_auth=True)
    client = await aiohttp_client(app)
    resp = await client.get("/api/files", headers={"Authorization": f"Bearer {token}"})
    assert resp.status == 200


@pytest.mark.asyncio
async def test_auth_disabled_by_default(client):
    """Verify no auth needed when require_auth is not set."""
    resp = await client.get("/api/files")
    assert resp.status == 200


# ─── Task 4: /api/stats ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_returns_level_distribution(client):
    """Verify stats response with level counts."""
    resp = await client.get("/api/stats?file=kutai.jsonl")
    assert resp.status == 200
    data = await resp.json()
    assert data["total"] == 5000
    # Every 10th line is ERROR
    assert data["levels"]["ERROR"] == 500
    assert data["levels"]["INFO"] == 4500
    assert data["time_range"]["first"] != ""
    assert data["time_range"]["last"] != ""
    assert len(data["top_sources"]) > 0
    assert data["indexed"] is False


# ─── Task 5: /api/context ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_filter_by_task(context_client):
    """Verify field filtering for task=42."""
    resp = await context_client.get("/api/context?file=ctx.jsonl&field=task&value=42")
    assert resp.status == 200
    data = await resp.json()
    # Every 5th line (i%5==0) has task="42", plus i=42 itself → 11 matches
    assert len(data["lines"]) == 11
    for line_str in data["lines"]:
        doc = json.loads(line_str)
        assert doc["task"] == "42"


# ─── Task 6: WebSocket tail ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ws_tail_receives_new_lines(log_dir, aiohttp_client):
    """Verify WS pushes new lines when file grows."""
    app = create_app(str(log_dir))
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws/tail?file=kutai.jsonl")

    # Give the WS loop time to start and record initial offset
    await asyncio.sleep(0.5)

    # Append new lines to the file
    log_path = log_dir / "kutai.jsonl"
    new_line = json.dumps({
        "ts": "2026-04-06T15:00:00.000Z",
        "level": "INFO",
        "src": "test",
        "msg": "ws_test_line",
    })
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(new_line + "\n")

    # Wait for the poll cycle (2s) plus margin
    msg = await asyncio.wait_for(ws.receive_json(), timeout=5)
    assert msg["type"] == "lines"
    assert any("ws_test_line" in l for l in msg["lines"])

    await ws.close()


# ─── Task 8: QR auth page ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_qr_page_returns_html_with_url(log_dir, auth_dir, aiohttp_client):
    """Verify QR page has token URL."""
    from yazbunu.auth import get_or_create_token
    token = get_or_create_token(str(auth_dir))
    app = create_app(str(log_dir), auth_dir=str(auth_dir), require_auth=True)
    client = await aiohttp_client(app)
    # /auth/qr is always open even with require_auth
    resp = await client.get("/auth/qr")
    assert resp.status == 200
    text = await resp.text()
    assert "text/html" in resp.content_type
    assert token in text
