"""yalayut.plugins.api — AccessPlugin for the ``api`` artifact type.

An api artifact is exposed almost exclusively via the ``tool`` class. This
plugin renders the namespaced tool registration payload (one tool per declared
verb, ``api_<slug>__<verb>``) and provides ``execute_api_tool`` — the path an
agent's tool-call reaches. HTTP execution delegates to the existing
``src/tools/free_apis.py::call_api`` (it already handles auth-header / apikey
substitution and response truncation).
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger

logger = get_logger("yalayut.plugin.api")


def _slug(name: str) -> str:
    """Canonical name -> tool-namespace slug. ``api-coingecko`` -> ``api_coingecko``."""
    return name.replace("-", "_")


def _verbs(api: dict[str, Any]) -> list[dict[str, Any]]:
    """Declared verbs, or a single synthetic ``get`` verb against the base URL.

    The ``public_apis_md`` discovery adapter yields api manifests with only a
    ``base_url`` — public-apis README rows carry no per-endpoint data. Without
    a fallback such artifacts would expose zero callable tools. The synthetic
    ``get`` verb makes the base URL reachable; richer manifests (cookiecutter-
    seeded or hand-authored) keep their explicit ``verbs``.
    """
    declared = api.get("verbs") or []
    if declared:
        return declared
    return [{
        "verb": "get",
        "endpoint": "",
        "params_schema": {},
        "description": api.get("description", ""),
    }]


class ApiPlugin:
    """AccessPlugin for api artifacts."""

    artifact_type = "api"

    def to_application(
        self, row: dict[str, Any], task_ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a SkillApplication-shaped dict for an api artifact.

        Returns the envelope entry intersect attaches to ``task['skills']``.
        When ``env_status`` is not 'ready' the tool list is empty (intersect
        also filters these at match time; this is defence in depth).
        """
        manifest = row.get("manifest") or {}
        api = manifest.get("api") or {}
        slug = _slug(row.get("name") or manifest.get("name") or "api_unknown")
        env_status = row.get("env_status", "ready")

        tools: list[dict[str, Any]] = []
        if env_status == "ready":
            base_url = api.get("base_url", "")
            auth_type = api.get("auth_type", "none")
            auth_env = api.get("auth_env")
            for verb in _verbs(api):
                vname = verb.get("verb")
                if not vname:
                    continue
                tools.append({
                    "tool_name": f"{slug}__{vname}",
                    "base_url": base_url,
                    "endpoint": verb.get("endpoint", ""),
                    "params_schema": verb.get("params_schema") or {},
                    "auth_type": auth_type,
                    "auth_env": auth_env,
                    "description": verb.get("description")
                    or api.get("description", ""),
                })

        return {
            "artifact_id": row.get("id"),
            "name": row.get("name"),
            "exposure_class": "tool",
            "applies_to": "execution",
            "render": None,
            "payload": {
                "kind": "api",
                "tools": tools,
                "skipped_reason": None if env_status == "ready" else env_status,
            },
            "confidence": float(task_ctx.get("_confidence", 0.0)),
        }

    def bind_args(self, row: dict[str, Any], task_ctx: dict[str, Any]) -> dict | None:
        """api artifacts are not parametric recipes — no static binding."""
        return None

    async def execute(
        self, row: dict[str, Any], task_ctx: dict[str, Any], inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Single-tool convenience execute (not on the hot path; tests/CLI)."""
        manifest = row.get("manifest") or {}
        api = manifest.get("api") or {}
        verbs = _verbs(api)
        tool_spec = {
            "tool_name": f"{_slug(row.get('name', 'api'))}__{verbs[0]['verb']}",
            "base_url": api.get("base_url", ""),
            "endpoint": verbs[0].get("endpoint", ""),
            "auth_type": api.get("auth_type", "none"),
            "auth_env": api.get("auth_env"),
        }
        return await execute_api_tool(tool_spec, inputs)


async def execute_api_tool(
    tool_spec: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Execute an api tool-call. Reached when an agent calls ``api_<slug>__<verb>``.

    ``tool_spec`` is one entry from ``ApiPlugin.to_application()['payload']['tools']``.
    Returns ``{"ok": bool, "response": str | None, "error": str | None}``.
    """
    from src.tools.free_apis import call_api

    base = (tool_spec.get("base_url") or "").rstrip("/")
    endpoint = tool_spec.get("endpoint") or ""
    full_url = base + ("/" + endpoint.lstrip("/") if endpoint else "")

    # Build a minimal FreeAPI-compatible dict for call_api (it accepts a dict).
    auth_type = tool_spec.get("auth_type", "none")
    auth_env = tool_spec.get("auth_env")
    api_dict = {
        "name": tool_spec.get("tool_name", "yalayut-api"),
        "base_url": base,
        "example_endpoint": full_url,
        "auth_type": (
            "apikey_header" if auth_type in ("apikey", "oauth") else "none"
        ),
        "env_var": auth_env,
        "key_header": None,
    }
    try:
        text = await call_api(api_dict, endpoint=full_url, params=params or {})
    except Exception as e:
        logger.warning("api tool execution raised",
                        tool=tool_spec.get("tool_name"), err=str(e))
        return {"ok": False, "response": None, "error": str(e)}

    is_error = isinstance(text, str) and text.startswith(
        ("API error", "Error:", "API timeout")
    )
    return {
        "ok": not is_error,
        "response": None if is_error else text,
        "error": text if is_error else None,
    }
