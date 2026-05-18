"""Z3 residual R2 — pending-URL filter extends to contract + perf.

The accessibility verb (``run_axe``) already soft-skipped on
``pending:<reason>`` markers.  This residual hardens
``run_schemathesis`` (contract review) and ``run_lighthouse``
(performance / web mode) to do the same so a mission with an
unbound preview tunnel does not produce phantom blockers.
"""
from __future__ import annotations

import asyncio

import pytest

from mr_roboto.preview_url import is_real_url
from mr_roboto.run_axe import run_axe
from mr_roboto.run_lighthouse import run_lighthouse
from mr_roboto.run_schemathesis import run_schemathesis


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestIsRealUrl:
    def test_https(self):
        assert is_real_url("https://example.com") is True

    def test_http(self):
        assert is_real_url("http://localhost:3000") is True

    def test_pending_marker(self):
        assert is_real_url("pending:tunnel-not-ready") is False

    def test_pending_with_colon_payload(self):
        assert is_real_url("pending:cloudflared:starting") is False

    def test_empty(self):
        assert is_real_url("") is False

    def test_whitespace(self):
        assert is_real_url("   ") is False

    def test_none(self):
        assert is_real_url(None) is False

    def test_non_http_scheme(self):
        assert is_real_url("ftp://files.example.com") is False

    def test_bare_host(self):
        assert is_real_url("example.com") is False

    def test_strips_surrounding_whitespace(self):
        assert is_real_url("  https://x.io  ") is True


class TestSchemathesisPendingSkip:
    def test_pending_base_url_soft_skips(self):
        r = _run(run_schemathesis(spec_path="/tmp/missing.yaml", base_url="pending:no-tunnel"))
        assert r["skipped"] is True
        assert "pending" in r["reason"].lower()
        assert r["verdict"] == "pass"
        assert r["findings"] == []

    def test_real_url_still_proceeds(self, monkeypatch):
        # Patch schemathesis locator to raise FileNotFoundError so we fall to
        # the "tool missing" skip — verifying we passed the URL filter.
        from mr_roboto import run_schemathesis as mod
        monkeypatch.setattr(mod, "_locate_schemathesis",
                            lambda: (_ for _ in ()).throw(FileNotFoundError()))
        # Need an existing spec file so we hit the locator branch.
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"openapi: 3.0.0\n")
            spec = f.name
        try:
            r = _run(run_schemathesis(spec_path=spec, base_url="http://localhost:8000"))
        finally:
            os.unlink(spec)
        assert r["skipped"] is True
        assert "schemathesis" in r["reason"].lower()

    def test_empty_base_url_soft_skips(self):
        r = _run(run_schemathesis(spec_path="/tmp/x.yaml", base_url=""))
        assert r["skipped"] is True
        assert "base_url" in r["reason"].lower()


class TestLighthousePendingSkip:
    def test_pending_preview_url_soft_skips(self):
        r = _run(run_lighthouse(preview_url="pending:cloudflared-bootstrap"))
        assert r["skipped"] is True
        assert "pending" in r["reason"].lower()
        assert r["verdict"] == "pass"

    def test_empty_preview_url_soft_skips(self):
        r = _run(run_lighthouse(preview_url=""))
        assert r["skipped"] is True

    def test_non_http_scheme_soft_skips(self):
        r = _run(run_lighthouse(preview_url="file:///tmp/index.html"))
        assert r["skipped"] is True


class TestAxeStillRespectsPending:
    """Regression: existing run_axe filter must still soft-skip."""

    def test_pending_marker(self):
        r = _run(run_axe(preview_url="pending:tunnel"))
        assert r["skipped"] is True
        assert r["verdict"] == "pass"
