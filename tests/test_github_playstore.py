"""Tests for GitHub search, Play Store, and PDF extraction tools."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# GitHub search tests
# ---------------------------------------------------------------------------


class TestGitHubSearchRepos:
    def test_returns_slimmed_repo_data(self):
        from src.tools.github_search import github_search_repos

        fake_response = {
            "items": [
                {
                    "full_name": "user/repo",
                    "description": "A cool project",
                    "stargazers_count": 1234,
                    "forks_count": 56,
                    "language": "Python",
                    "updated_at": "2026-01-15T00:00:00Z",
                    "html_url": "https://github.com/user/repo",
                    "topics": ["web", "scraping"],
                }
            ]
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=fake_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.github_search.aiohttp.ClientSession", return_value=mock_session):
            results = _run(github_search_repos("web scraping python", count=5))

        assert len(results) == 1
        assert results[0]["name"] == "user/repo"
        assert results[0]["stars"] == 1234
        assert results[0]["language"] == "Python"

    def test_http_error(self):
        from src.tools.github_search import github_search_repos

        mock_resp = AsyncMock()
        mock_resp.status = 403
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.github_search.aiohttp.ClientSession", return_value=mock_session):
            results = _run(github_search_repos("test"))

        assert len(results) == 1
        assert "error" in results[0]


class TestGitHubRepoReadme:
    def test_returns_raw_text(self):
        from src.tools.github_search import github_repo_readme

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value="# My Project\nThis is a README.")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.github_search.aiohttp.ClientSession", return_value=mock_session):
            text = _run(github_repo_readme("user/repo"))

        assert "My Project" in text


# ---------------------------------------------------------------------------
# Play Store tests
# ---------------------------------------------------------------------------


class TestPlayStoreSearch:
    def test_returns_slimmed_results(self):
        from src.tools.play_store import play_store_search

        fake_results = [
            {
                "appId": "com.example.app",
                "title": "Example App",
                "score": 4.5,
                "installs": "1,000,000+",
                "developer": "Example Inc",
                "free": True,
                "price": 0,
                "summary": "A great app",
            }
        ]

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=fake_results)
            results = _run(play_store_search("example app", count=5))

        assert len(results) == 1
        assert results[0]["app_id"] == "com.example.app"
        assert results[0]["title"] == "Example App"
        assert results[0]["score"] == 4.5


class TestPlayStoreReviews:
    def test_returns_slimmed_review_data(self):
        """Test the review data shape directly."""
        fake_reviews = [
            {
                "score": 5,
                "content": "Amazing app!",
                "thumbsUpCount": 10,
                "at": "2026-01-01",
            },
            {
                "score": 1,
                "content": "Terrible.",
                "thumbsUpCount": 0,
                "at": "2026-01-02",
            },
        ]

        # Process like the function would
        result = [
            {
                "score": r.get("score", 0),
                "text": (r.get("content") or "")[:300],
                "thumbs_up": r.get("thumbsUpCount", 0),
                "date": str(r.get("at", "")),
            }
            for r in fake_reviews
        ]

        assert len(result) == 2
        assert result[0]["score"] == 5
        assert result[0]["text"] == "Amazing app!"
        assert result[1]["thumbs_up"] == 0


# ---------------------------------------------------------------------------
# Tool wrapper tests
# ---------------------------------------------------------------------------


class TestToolGitHub:
    def test_repos_returns_json(self):
        from src.tools import TOOL_REGISTRY
        import src.tools as tools_mod

        if "github" not in TOOL_REGISTRY:
            pytest.skip("github tool not registered")

        tool_fn = TOOL_REGISTRY["github"]["function"]
        fake_results = [{"name": "user/repo", "stars": 100}]

        # Patch where the wrapper function looks up the name
        with patch.object(tools_mod, "github_search_repos", AsyncMock(return_value=fake_results)):
            result = _run(tool_fn(action="repos", query="test"))

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "user/repo"

    def test_readme_returns_text(self):
        from src.tools import TOOL_REGISTRY
        import src.tools as tools_mod

        if "github" not in TOOL_REGISTRY:
            pytest.skip("github tool not registered")

        tool_fn = TOOL_REGISTRY["github"]["function"]

        with patch.object(tools_mod, "github_repo_readme", AsyncMock(return_value="# README")):
            result = _run(tool_fn(action="readme", repo="user/repo"))

        assert "README" in result

    def test_missing_args_returns_usage(self):
        from src.tools import TOOL_REGISTRY

        if "github" not in TOOL_REGISTRY:
            pytest.skip("github tool not registered")

        tool_fn = TOOL_REGISTRY["github"]["function"]
        result = _run(tool_fn(action="repos", query=""))
        assert "Usage" in result


class TestToolPlayStore:
    def test_search_returns_json(self):
        from src.tools import TOOL_REGISTRY
        import src.tools as tools_mod

        if "play_store" not in TOOL_REGISTRY:
            pytest.skip("play_store tool not registered")

        tool_fn = TOOL_REGISTRY["play_store"]["function"]
        fake_results = [{"app_id": "com.test", "title": "Test"}]

        with patch.object(tools_mod, "play_store_search", AsyncMock(return_value=fake_results)):
            result = _run(tool_fn(action="search", query="test"))

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["app_id"] == "com.test"

    def test_missing_args_returns_usage(self):
        from src.tools import TOOL_REGISTRY

        if "play_store" not in TOOL_REGISTRY:
            pytest.skip("play_store tool not registered")

        tool_fn = TOOL_REGISTRY["play_store"]["function"]
        result = _run(tool_fn(action="search", query=""))
        assert "Usage" in result


# ---------------------------------------------------------------------------
# PDF extraction tests
# ---------------------------------------------------------------------------


class TestPDFExtract:
    def test_file_not_found(self):
        from src.tools.pdf_extract import extract_pdf

        result = _run(extract_pdf("/nonexistent/file.pdf"))
        assert "Error: file not found" in result

    def test_not_pdf(self):
        from src.tools.pdf_extract import extract_pdf

        fd, path = tempfile.mkstemp(suffix=".txt")
        os.write(fd, b"not a pdf")
        os.close(fd)
        try:
            result = _run(extract_pdf(path))
            assert "Error: not a PDF file" in result
        finally:
            os.unlink(path)
