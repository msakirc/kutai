import asyncio

import pytest

from yalayut.discovery.sources.github_topic import GithubTopicAdapter
from yalayut.discovery.sources.awesome_list_md import AwesomeListAdapter
from yalayut.discovery.sources.web_markdown import WebMarkdownAdapter
from yalayut.discovery.sources.clawhub_api import ClawHubAdapter
from yalayut.contracts import SourceConfig


_AWESOME_README = """\
## Cloud
- **[mcp-server-cloudflare](https://github.com/cloudflare/mcp-server-cloudflare)** - Manage Cloudflare Workers, KV, R2.
- **[open-museum-mcp](https://github.com/x/open-museum-mcp)** - Federated museum collections.

## Browser
- **[mcp-browser-use](https://github.com/y/mcp-browser-use)** - Browser automation via Playwright.
"""

_SKILL_MD = """\
---
name: pdf
description: Use this skill for working with PDF files — extract, merge, split.
license: Proprietary
---
# PDF skill body
Detailed instructions here.
"""


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_clawhub_is_stub(loop):
    async def _run():
        adapter = ClawHubAdapter()
        assert adapter.source_type == "clawhub_api"
        cfg = SourceConfig(source_id="clawhub:stub", source_type="clawhub_api",
                           endpoint="", owner="clawhub")
        refs = await adapter.discover(cfg)
        assert refs == []
    loop.run_until_complete(_run())


def test_awesome_list_parses_bullets(loop):
    async def _run():
        adapter = AwesomeListAdapter()
        refs = adapter.parse_readme(_AWESOME_README,
                                    source_id="github:punkpeye/awesome-mcp-servers")
        names = {r.name_original for r in refs}
        assert "mcp-server-cloudflare" in names
        assert "open-museum-mcp" in names
        assert "mcp-browser-use" in names
        cf = next(r for r in refs if r.name_original == "mcp-server-cloudflare")
        assert cf.owner == "cloudflare"
        assert cf.raw_url.endswith("mcp-server-cloudflare")
    loop.run_until_complete(_run())


def test_web_markdown_parses_frontmatter(loop):
    async def _run():
        adapter = WebMarkdownAdapter()
        ref, body = adapter.parse_skill_md(
            _SKILL_MD, url="https://example.com/SKILL.md")
        assert ref.name_original == "pdf"
        assert ref.native_format == "frontmatter"
        assert "PDF skill body" in body
    loop.run_until_complete(_run())


def test_github_topic_canonical_slug(loop):
    async def _run():
        adapter = GithubTopicAdapter()
        # name canonicalization: <source-slug>-<original>, dedup org prefix.
        assert adapter.canonical_name("anthropics", "pdf") == "anthropics-pdf"
        assert adapter.canonical_name("matlab", "matlab-live-script") == \
            "matlab-live-script"
        assert adapter.canonical_name("cookiecutter",
                                      "cookiecutter-django") == "cc-django"
    loop.run_until_complete(_run())
