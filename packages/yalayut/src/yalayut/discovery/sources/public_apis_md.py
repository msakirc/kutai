"""public_apis_md discovery adapter.

Mechanical (no LLM). Parses the public-apis/public-apis README markdown tables
(``API | Description | Auth | HTTPS | CORS``) into ``api`` artifact manifests.
Recon confirmed this is the cleanest non-frontmatter source; confidence 0.9.
"""
from __future__ import annotations

import re
from typing import Any

import aiohttp

from yazbunu import get_logger

logger = get_logger("yalayut.adapter.public_apis")

# Markdown link in the API cell: [Name](url)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# A markdown table data row: starts and ends with a pipe.
_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
# Separator row: |---|---| ...
_SEP_RE = re.compile(r"^\|[\s:|-]+\|\s*$")

_STOPWORDS = frozenset(
    {"and", "or", "the", "a", "an", "for", "with", "data", "api", "to", "of",
     "in", "on", "your", "this", "free", "realtime", "historical"}
)


def _slugify_name(raw: str) -> str:
    """``CoinGecko`` -> ``api-coingecko``; ``Alpha Vantage`` -> ``api-alpha-vantage``."""
    base = re.sub(r"[^a-z0-9]+", "-", raw.strip().lower()).strip("-")
    return f"api-{base or 'unknown'}"


def _auth_env_var(canonical_name: str) -> str:
    """``api-alpha-vantage`` -> ``ALPHA_VANTAGE_API_KEY``."""
    stem = canonical_name[len("api-"):] if canonical_name.startswith("api-") else canonical_name
    return re.sub(r"[^A-Z0-9]+", "_", stem.upper()).strip("_") + "_API_KEY"


def _intent_keywords(description: str) -> list[str]:
    tokens = re.findall(r"[a-z]{3,}", description.lower())
    seen: list[str] = []
    for tok in tokens:
        if tok not in _STOPWORDS and tok not in seen:
            seen.append(tok)
    return seen[:8]


def _classify_auth(auth_cell: str) -> tuple[str, bool]:
    """Return (auth_type, requires_key). auth_type in {none, apikey, oauth}."""
    cell = auth_cell.strip().strip("`").lower()
    if cell in ("", "no", "none"):
        return "none", False
    if "oauth" in cell:
        return "oauth", True
    return "apikey", True


def _parse_row(cells: list[str]) -> dict[str, Any] | None:
    """Parse one table data row into an api manifest, or None if malformed."""
    if len(cells) < 5:
        return None
    api_cell, desc, auth_cell, https_cell, _cors = cells[:5]
    m = _LINK_RE.search(api_cell)
    if not m:
        return None
    name_original, url = m.group(1).strip(), m.group(2).strip()
    canonical = _slugify_name(name_original)
    auth_type, requires_key = _classify_auth(auth_cell)
    https = https_cell.strip().lower() in ("yes", "true")
    base_url = url
    if not base_url.startswith("http"):
        base_url = "https://" + base_url

    return {
        "name": canonical,
        "name_original": name_original,
        "version": "1.0.0",
        "artifact_type": "api",
        "kind": None,
        "source": "github:public-apis/public-apis",
        "owner": "public-apis",
        "license": None,
        "mechanizable": False,
        "model_hint": None,
        "intent_keywords": _intent_keywords(desc),
        "api": {
            "base_url": base_url,
            "doc_url": url,
            "auth_type": auth_type,
            "auth_env": _auth_env_var(canonical) if requires_key else None,
            "https": https,
            "description": desc.strip(),
        },
        "disabled_imports_check": True,
    }


def parse_public_apis_md(md_text: str) -> list[dict[str, Any]]:
    """Parse every API table row in a public-apis README into manifests."""
    manifests: list[dict[str, Any]] = []
    for line in md_text.splitlines():
        line = line.rstrip()
        if _SEP_RE.match(line):
            continue
        row = _ROW_RE.match(line)
        if not row:
            continue
        cells = [c.strip() for c in row.group(1).split("|")]
        # Header row ("API | Description | Auth | ...") has no markdown link.
        if not _LINK_RE.search(cells[0]):
            continue
        manifest = _parse_row(cells)
        if manifest is not None:
            manifests.append(manifest)
    return manifests


class PublicApisAdapter:
    """SourceAdapter for the public-apis/public-apis README."""

    source_type = "public_apis_md"

    async def _fetch_md(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"public-apis fetch HTTP {resp.status}")
                return await resp.text()

    async def discover(self, source_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        endpoint = source_cfg.get("endpoint")
        if not endpoint:
            logger.warning("public_apis source_cfg missing endpoint", cfg=source_cfg)
            return []
        try:
            md_text = await self._fetch_md(endpoint)
        except Exception as e:
            logger.warning("public-apis discovery failed", err=str(e))
            return []
        refs: list[dict[str, Any]] = []
        for manifest in parse_public_apis_md(md_text):
            refs.append({
                "source_id": source_cfg.get("source_id",
                                             "github:public-apis/public-apis"),
                "name": manifest["name"],
                "manifest": manifest,
                "body": "",
            })
        return refs

    async def fetch(self, ref: dict[str, Any]):
        return None
