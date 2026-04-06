"""Tests for yazbunu log viewer server API."""
import json
import os
import sys
import time

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yazbunu"))

from yazbunu.server import create_app


@pytest.fixture
def log_dir(tmp_path):
    """Create a temp log dir with sample JSONL data."""
    ts_base = "2026-04-06T12:00:0"
    lines = []
    for i in range(50):
        doc = {
            "ts": f"{ts_base}{i % 10}.000Z",
            "level": "ERROR" if i == 49 else "INFO",
            "src": "kutai.core.orchestrator",
            "msg": f"line {i}",
            "task": str(i),
        }
        lines.append(json.dumps(doc))
    (tmp_path / "kutai.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "shopping.jsonl").write_text(
        json.dumps({"ts": "2026-04-06T13:00:00.000Z", "level": "INFO",
                     "src": "shopping.scraper", "msg": "scrape done"}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest_asyncio.fixture
async def client(log_dir, aiohttp_client):
    app = create_app(str(log_dir))
    return await aiohttp_client(app)


@pytest.mark.asyncio
async def test_list_files(client):
    resp = await client.get("/api/files")
    assert resp.status == 200
    data = await resp.json()
    names = [f["name"] for f in data["files"]]
    assert "kutai.jsonl" in names
    assert "shopping.jsonl" in names


@pytest.mark.asyncio
async def test_get_logs_last_n(client):
    resp = await client.get("/api/logs?file=kutai.jsonl&lines=5")
    assert resp.status == 200
    data = await resp.json()
    assert len(data["lines"]) == 5
    # Should be the LAST 5 lines
    assert json.loads(data["lines"][-1])["msg"] == "line 49"


@pytest.mark.asyncio
async def test_get_logs_default_1000(client):
    resp = await client.get("/api/logs?file=kutai.jsonl")
    assert resp.status == 200
    data = await resp.json()
    assert len(data["lines"]) == 50  # all lines (fewer than 1000)


@pytest.mark.asyncio
async def test_tail_after_timestamp(client):
    resp = await client.get("/api/tail?file=kutai.jsonl&after=2026-04-06T12:00:05.000Z")
    assert resp.status == 200
    data = await resp.json()
    # Should return lines with ts > the given timestamp
    for line_str in data["lines"]:
        doc = json.loads(line_str)
        assert doc["ts"] >= "2026-04-06T12:00:05.000Z"


@pytest.mark.asyncio
async def test_file_not_found(client):
    resp = await client.get("/api/logs?file=nonexistent.jsonl")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_path_traversal_blocked(client):
    resp = await client.get("/api/logs?file=../../../etc/passwd")
    assert resp.status == 400
