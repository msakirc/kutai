"""Yalayut Phase 4 — github_topic source adapter.

Stage 1: GitHub topic search → repo list (mechanical).
Stage 2: per-repo SKILL.md probe; canonical frontmatter parses mechanically,
otherwise the README is handed to LLM-fallback synthesis.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import frontmatter
import httpx

from yazbunu import get_logger
from yalayut.contracts import ArtifactRef, SourceConfig

logger = get_logger("yalayut.adapter.github_topic")

#: Per-run repo cap — prevents a topic search flooding the catalog.
TOPIC_REPO_CAP: int = 25


class GithubTopicAdapter:
    source_type: str = "github_topic"

    def canonical_name(self, owner: str, name_original: str) -> str:
        """``<source-slug>-<original>`` with the dumb-prefix fixes from recon:
        - drop org prefix when the name already starts with it
          (``matlab`` + ``matlab-live-script`` → ``matlab-live-script``);
        - cookiecutter repos collapse to the ``cc-`` slug
          (``cookiecutter`` + ``cookiecutter-django`` → ``cc-django``)."""
        n = name_original.strip()
        o = owner.strip().lower()
        if o == "cookiecutter" or n.startswith("cookiecutter-"):
            return "cc-" + n.removeprefix("cookiecutter-")
        if n.lower().startswith(o + "-") or n.lower() == o:
            return n
        return f"{o}-{n}"

    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]:
        """Topic search via the public GitHub REST API. ``endpoint`` carries
        the topic string (e.g. ``claude-skill``)."""
        topic = source_cfg.endpoint or "claude-skill"
        url = (f"https://api.github.com/search/repositories"
               f"?q=topic:{topic}&sort=stars&per_page={TOPIC_REPO_CAP}")
        headers = {"Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 403:
                logger.warning("github_topic rate limited — skipping run")
                return []
            resp.raise_for_status()
            items = resp.json().get("items", [])
        refs: list[ArtifactRef] = []
        for repo in items[:TOPIC_REPO_CAP]:
            owner = (repo.get("owner") or {}).get("login", "unknown")
            name_original = repo.get("name", "")
            if not name_original:
                continue
            default_branch = repo.get("default_branch", "main")
            skill_url = (f"https://raw.githubusercontent.com/{owner}/"
                         f"{name_original}/{default_branch}/SKILL.md")
            refs.append(ArtifactRef(
                name=self.canonical_name(owner, name_original),
                name_original=name_original,
                owner=owner,
                source=f"github_topic:{topic}",
                raw_url=skill_url,
                native_format="unknown",
            ))
        logger.info("github_topic discovered", topic=topic, count=len(refs))
        return refs

    async def fetch(self, ref: ArtifactRef) -> Path:
        """Probe SKILL.md; mark native_format so synthesis picks the path."""
        staging = Path(tempfile.mkdtemp(prefix="yalayut_topic_"))
        body_path = staging / "SKILL.md"
        async with httpx.AsyncClient(timeout=30,
                                     follow_redirects=True) as client:
            resp = await client.get(ref.raw_url)
        if resp.status_code == 200 and resp.text.strip().startswith("---"):
            try:
                frontmatter.loads(resp.text)
                ref.native_format = "frontmatter"
            except Exception:
                ref.native_format = "freeform"
            body_path.write_text(resp.text, encoding="utf-8")
        else:
            # No canonical SKILL.md — fetch the README for LLM fallback.
            ref.native_format = "freeform"
            readme_url = ref.raw_url.rsplit("/", 1)[0] + "/README.md"
            try:
                async with httpx.AsyncClient(timeout=30,
                                             follow_redirects=True) as c:
                    r2 = await c.get(readme_url)
                    r2.raise_for_status()
                body_path.write_text(r2.text, encoding="utf-8")
            except Exception:
                body_path.write_text("", encoding="utf-8")
        return body_path
