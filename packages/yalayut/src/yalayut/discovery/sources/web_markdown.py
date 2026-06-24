"""Yalayut Phase 4 — web_markdown source adapter.

Fetches a single generic SKILL.md URL and parses YAML frontmatter
mechanically (no LLM). Same shape as github_path; different fetch.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import frontmatter
import httpx

from yazbunu import get_logger
from yalayut.contracts import ArtifactRef, SourceConfig

logger = get_logger("yalayut.adapter.web_markdown")


class WebMarkdownAdapter:
    source_type: str = "web_markdown"

    def parse_skill_md(self, text: str, url: str) -> tuple[ArtifactRef, str]:
        """Parse a SKILL.md string → (ArtifactRef, body). Mechanical."""
        post = frontmatter.loads(text)
        name_original = str(post.get("name") or "").strip()
        if not name_original:
            raise ValueError(f"SKILL.md at {url} has no 'name' frontmatter")
        owner = url.split("//", 1)[-1].split("/", 1)[0]
        ref = ArtifactRef(
            name=f"web-{name_original}",
            name_original=name_original,
            owner=owner,
            source=f"web:{url}",
            raw_url=url,
            native_format="frontmatter",
        )
        return ref, post.content

    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]:
        url = source_cfg.endpoint
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        ref, _body = self.parse_skill_md(resp.text, url)
        return [ref]

    async def fetch(self, ref: ArtifactRef) -> Path:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(ref.raw_url)
            resp.raise_for_status()
        staging = Path(tempfile.mkdtemp(prefix="yalayut_web_"))
        body_path = staging / "SKILL.md"
        body_path.write_text(resp.text, encoding="utf-8")
        logger.info("web_markdown fetched", url=ref.raw_url)
        return body_path
