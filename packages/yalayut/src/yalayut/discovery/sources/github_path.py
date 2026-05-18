"""github_path source adapter — anthropics/skills, obra/superpowers, etc.

Dead-mechanical: glob '<path>/*/SKILL.md', parse YAML frontmatter. No LLM.
Confidence 0.95 per recon. Uses the GitHub REST API for the tree listing and
raw.githubusercontent.com for body fetches; honors GITHUB_TOKEN if present
(higher rate limit) but works unauthenticated.
"""
from __future__ import annotations

import os
from pathlib import Path

import frontmatter
import httpx

from yalayut.contracts import ArtifactRef, SourceConfig
from yalayut.discovery.fetch import stage_dir

_API = "https://api.github.com"
_RAW = "https://raw.githubusercontent.com"


def parse_skill_md(raw: bytes) -> tuple[dict, str]:
    """Parse a SKILL.md: (frontmatter dict, body string)."""
    post = frontmatter.loads(raw.decode("utf-8", errors="replace"))
    return dict(post.metadata), post.content


def _parse_source_id(source_id: str) -> tuple[str, str, str]:
    """'github:anthropics/skills@/skills' -> (owner, repo, path)."""
    body = source_id.split("github:", 1)[-1]
    repo_part, _, path = body.partition("@")
    owner, _, repo = repo_part.partition("/")
    return owner, repo, path.strip("/") or ""


class GithubPathAdapter:
    """SourceAdapter for repos that host one SKILL.md per directory."""

    source_type = "github_path"

    def _headers(self) -> dict:
        tok = os.environ.get("GITHUB_TOKEN")
        h = {"Accept": "application/vnd.github+json"}
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        return h

    async def _http_get(self, url: str) -> bytes:
        """Raw GET — body bytes. Overridden in tests."""
        async with httpx.AsyncClient(timeout=30.0) as c:
            resp = await c.get(url, headers=self._headers())
            resp.raise_for_status()
            return resp.content

    async def _list_tree(
        self, owner: str, repo: str, path: str
    ) -> list[str]:
        """Recursive git tree paths under the repo. Overridden in tests."""
        url = f"{_API}/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
        async with httpx.AsyncClient(timeout=30.0) as c:
            resp = await c.get(url, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        return [
            t["path"] for t in data.get("tree", [])
            if t.get("type") == "blob"
        ]

    async def discover(
        self, source_cfg: SourceConfig
    ) -> list[ArtifactRef]:
        """List every '<path>/<name>/SKILL.md' as an ArtifactRef."""
        owner, repo, path = _parse_source_id(source_cfg.source_id)
        prefix = (path + "/") if path else ""
        all_paths = await self._list_tree(owner, repo, path)
        refs: list[ArtifactRef] = []
        for p in all_paths:
            if not p.startswith(prefix) or not p.endswith("/SKILL.md"):
                continue
            rel = p[len(prefix):]                 # '<name>/SKILL.md'
            name = rel.split("/", 1)[0]
            if not name or "/" in rel.rstrip("/SKILL.md").strip("/") \
                    and rel.count("/") > 1:
                # only direct '<name>/SKILL.md', skip nested helpers
                continue
            refs.append(ArtifactRef(
                source_id=source_cfg.source_id,
                name=name,
                fetch_url=f"{_RAW}/{owner}/{repo}/HEAD/{p}",
                owner=owner,
                raw_meta={"path": p},
            ))
        return refs

    async def fetch(self, ref: ArtifactRef) -> Path:
        """Download the SKILL.md into staging. Returns the body file path."""
        body = await self._http_get(ref.fetch_url)
        staging = stage_dir(ref.source_id, ref.name)
        out = staging / "SKILL.md"
        out.write_bytes(body)
        return out
