"""Tests for mr_roboto.http_check — the staging health-check verb.

We use aiohttp's pytest plugin to stand up a tiny app and hit it for real.
That covers the success / 5xx-retry / 4xx-fail-fast / body-match paths
without monkeypatching the verb's HTTP layer.
"""
from __future__ import annotations

import pytest
from aiohttp import web

from mr_roboto.http_check import http_check
import mr_roboto


@pytest.fixture
async def server(aiohttp_server):
    counts: dict[str, int] = {"flaky": 0}

    async def ok(_req):
        return web.Response(text="ok body", status=200)

    async def teapot(_req):
        return web.Response(status=418, text="nope")

    async def flaky(_req):
        counts["flaky"] += 1
        if counts["flaky"] < 3:
            return web.Response(status=503)
        return web.Response(status=200, text="recovered")

    async def slow(_req):
        import asyncio
        await asyncio.sleep(2.0)
        return web.Response(status=200)

    app = web.Application()
    app.router.add_get("/ok", ok)
    app.router.add_get("/teapot", teapot)
    app.router.add_get("/flaky", flaky)
    app.router.add_get("/slow", slow)
    srv = await aiohttp_server(app)
    srv._counts = counts
    return srv


def _url(srv, path: str) -> str:
    return f"http://{srv.host}:{srv.port}{path}"


@pytest.mark.asyncio
async def test_http_check_invalid_url():
    res = await http_check("not-a-url")
    assert res["ok"] is False
    assert res["final_error"] == "invalid url"


@pytest.mark.asyncio
async def test_http_check_unsupported_method():
    res = await http_check("https://example.com/", method="POST")
    assert res["ok"] is False
    assert "unsupported" in res["final_error"]


@pytest.mark.asyncio
async def test_http_check_200_ok(server):
    res = await http_check(_url(server, "/ok"), max_attempts=1)
    assert res["ok"] is True
    assert res["final_status"] == 200
    assert res["attempts"] == 1


@pytest.mark.asyncio
async def test_http_check_4xx_fails_fast(server):
    res = await http_check(_url(server, "/teapot"), max_attempts=4)
    assert res["ok"] is False
    assert res["final_status"] == 418
    # 418 is not in retry set — should not retry.
    assert res["attempts"] == 1


@pytest.mark.asyncio
async def test_http_check_5xx_retries_then_succeeds(server):
    res = await http_check(
        _url(server, "/flaky"),
        max_attempts=5,
        backoff_base_s=0.01,
        backoff_cap_s=0.05,
    )
    assert res["ok"] is True, res
    assert res["final_status"] == 200
    assert res["attempts"] >= 3


@pytest.mark.asyncio
async def test_http_check_timeout_fails(server):
    res = await http_check(
        _url(server, "/slow"),
        timeout_s=0.2,
        max_attempts=2,
        backoff_base_s=0.01,
        backoff_cap_s=0.02,
    )
    assert res["ok"] is False
    assert res["final_error"] is not None


@pytest.mark.asyncio
async def test_http_check_body_match_passes(server):
    res = await http_check(
        _url(server, "/ok"),
        max_attempts=1,
        expect_body_contains="ok body",
    )
    assert res["ok"] is True
    assert res["body_match"] is True


@pytest.mark.asyncio
async def test_http_check_body_match_fails(server):
    res = await http_check(
        _url(server, "/ok"),
        max_attempts=1,
        expect_body_contains="not present",
    )
    assert res["ok"] is False
    assert res["body_match"] is False


@pytest.mark.asyncio
async def test_http_check_via_dispatcher(server):
    action = await mr_roboto.run({
        "mission_id": None,
        "payload": {
            "action": "http_check",
            "url": _url(server, "/ok"),
            "max_attempts": 1,
        },
    })
    assert action.status == "completed", action
    assert action.result["ok"] is True


@pytest.mark.asyncio
async def test_http_check_dispatcher_fail(server):
    action = await mr_roboto.run({
        "mission_id": None,
        "payload": {
            "action": "http_check",
            "url": _url(server, "/teapot"),
            "max_attempts": 1,
        },
    })
    assert action.status == "failed"
    assert "418" in (action.error or "")
