"""Yalayut Phase 4 — awesome_list_md source adapter.

Two-pass: (1) regex extracts ``name + repo URL + raw description`` from
``**[name](url)** - desc`` bullets; (2) LLM-normalizes each bullet to
intent_keywords / auth_env / install_cmd via discovery.synthesize.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import httpx

from src.infra.logging_config import get_logger
from yalayut.contracts import ArtifactRef, SourceConfig

logger = get_logger("yalayut.adapter.awesome_list")

#: **[name](url)** - description   (markdown bullet, awesome-list house style)
_BULLET = re.compile(
    r"^\s*[-*]\s*\*\*\[(?P<name>[^\]]+)\]\((?P<url>https?://[^)]+)\)\*\*"
    r"\s*[-–:]?\s*(?P<desc>.*)$"
)


class AwesomeListAdapter:
    source_type: str = "awesome_list_md"

    def parse_readme(self, text: str, source_id: str) -> list[ArtifactRef]:
        """Pass 1 — mechanical bullet parse. No LLM."""
        refs: list[ArtifactRef] = []
        for line in text.splitlines():
            m = _BULLET.match(line)
            if not m:
                continue
            name_original = m.group("name").strip()
            url = m.group("url").strip()
            # owner = the GitHub org segment of the URL.
            parts = url.split("//", 1)[-1].split("/")
            owner = parts[1] if len(parts) > 1 else "unknown"
            refs.append(ArtifactRef(
                name=f"{owner}-{name_original}",
                name_original=name_original,
                owner=owner,
                source=source_id,
                raw_url=url,
                native_format="awesome_bullet",
                raw_description=m.group("desc").strip(),
            ))
        logger.info("awesome_list parsed", source=source_id, count=len(refs))
        return refs

    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            resp = await c.get(source_cfg.endpoint)
            resp.raise_for_status()
        return self.parse_readme(resp.text, source_cfg.source_id)

    async def fetch(self, ref: ArtifactRef) -> Path:
        """Fetch the linked repo's README for LLM synthesis input."""
        readme_url = ref.raw_url.rstrip("/") + "/raw/HEAD/README.md"
        staging = Path(tempfile.mkdtemp(prefix="yalayut_awesome_"))
        body_path = staging / "README.md"
        try:
            async with httpx.AsyncClient(timeout=30,
                                         follow_redirects=True) as c:
                resp = await c.get(readme_url)
                resp.raise_for_status()
            body_path.write_text(resp.text, encoding="utf-8")
        except Exception as e:
            # README unreachable — fall back to the bullet description so
            # synthesis still has text to work with.
            logger.debug("awesome_list README fetch failed: %s", e)
            body_path.write_text(ref.raw_description or "", encoding="utf-8")
        return body_path
