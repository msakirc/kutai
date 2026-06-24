"""cookiecutter_template discovery adapter.

Mechanical (no LLM). For a cookiecutter template repo, fetch its
``cookiecutter.json`` from GitHub raw and synthesize a yalayut ``shell_recipe``
manifest: the ``inputs_schema`` is lifted directly from the JSON variables and
the invocation is a single ``uvx cookiecutter --no-input gh:<owner>/<repo>``.

Recon confirmed cookiecutter.json IS the input schema; there is no YAML
frontmatter to parse. Confidence 0.85.
"""
from __future__ import annotations

import json
from typing import Any

import aiohttp

from yazbunu import get_logger

logger = get_logger("yalayut.adapter.cookiecutter")

_RAW_URL = "https://raw.githubusercontent.com/{owner}/{repo}/HEAD/cookiecutter.json"


def _canonical_name(repo: str) -> str:
    """``cookiecutter-django`` -> ``cc-django``; ``flask-x`` -> ``cc-flask-x``."""
    base = repo
    if base.startswith("cookiecutter-"):
        base = base[len("cookiecutter-"):]
    base = base.strip("-") or "template"
    return f"cc-{base}"


def _infer_field(key: str, value: Any) -> dict[str, Any] | None:
    """Map one cookiecutter.json entry to an inputs_schema field.

    Returns ``None`` for entries that are not user inputs (private keys, and
    Jinja-templated derived values).
    """
    if key.startswith("_"):
        return None
    if isinstance(value, list):
        # cookiecutter list = choice; first element is the default.
        choices = [str(v) for v in value]
        return {
            "type": "choice",
            "choices": choices,
            "default": choices[0] if choices else None,
        }
    if isinstance(value, bool):
        return {"type": "bool", "default": value}
    if isinstance(value, str):
        # A Jinja expression ({{ ... }}) is a derived slug, not an input.
        if "{{" in value and "}}" in value:
            return None
        # y/n strings are cookiecutter's boolean idiom.
        if value.strip().lower() in ("y", "n", "yes", "no"):
            return {"type": "bool",
                    "default": value.strip().lower() in ("y", "yes")}
        return {"type": "string", "default": value}
    if isinstance(value, (int, float)):
        return {"type": "number", "default": value}
    return {"type": "string", "default": str(value)}


def cookiecutter_json_to_manifest(
    cc_json: dict[str, Any], owner: str, repo: str
) -> dict[str, Any]:
    """Synthesize a shell_recipe manifest from a parsed cookiecutter.json."""
    inputs_schema: dict[str, Any] = {}
    for key, value in (cc_json or {}).items():
        field = _infer_field(key, value)
        if field is not None:
            inputs_schema[key] = field

    return {
        "name": _canonical_name(repo),
        "name_original": repo,
        "version": "1.0.0",
        "artifact_type": "skill",
        "kind": "shell_recipe",
        "source": f"github:{owner}/{repo}",
        "owner": owner,
        "license": None,
        "mechanizable": True,
        "model_hint": None,
        "intent_keywords": [w for w in repo.replace("cookiecutter-", "").split("-")
                            if w],
        "inputs_schema": inputs_schema,
        "invocation": {
            "steps": [
                {"cmd": f"uvx cookiecutter --no-input gh:{owner}/{repo}"}
            ]
        },
        "artifacts": [],
        "disabled_imports_check": True,
    }


class CookiecutterAdapter:
    """SourceAdapter for individual cookiecutter template repos."""

    source_type = "cookiecutter_template"

    async def _fetch_cookiecutter_json(
        self, owner: str, repo: str
    ) -> dict[str, Any]:
        url = _RAW_URL.format(owner=owner, repo=repo)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"cookiecutter.json fetch HTTP {resp.status} for {owner}/{repo}"
                    )
                text = await resp.text()
        return json.loads(text)

    async def discover(self, source_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a single ArtifactRef (one repo = one shell_recipe)."""
        owner = source_cfg.get("owner")
        repo = source_cfg.get("repo")
        if not owner or not repo:
            logger.warning("cookiecutter source_cfg missing owner/repo",
                            cfg=source_cfg)
            return []
        try:
            cc_json = await self._fetch_cookiecutter_json(owner, repo)
        except Exception as e:
            logger.warning("cookiecutter.json discovery failed",
                            owner=owner, repo=repo, err=str(e))
            return []
        manifest = cookiecutter_json_to_manifest(cc_json, owner=owner, repo=repo)
        return [{
            "source_id": source_cfg.get("source_id", f"github:{owner}/{repo}"),
            "name": manifest["name"],
            "manifest": manifest,
            "body": "",
        }]

    async def fetch(self, ref: dict[str, Any]):
        """No body to fetch — cookiecutter recipes are pure invocation."""
        return None
