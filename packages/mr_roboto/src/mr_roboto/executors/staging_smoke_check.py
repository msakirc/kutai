"""Staging smoke-check executor.

Reads the per-feature ``staging_deployment_result`` artifact (the template
expander prefixes it with ``{feature_id}__``), pulls the staging URL out,
runs :func:`mr_roboto.http_check.http_check` against it.

Honest scope: verifies the deployment URL responds with 2xx. Does NOT
cover visual screenshot diff, multi-breakpoint render, or interactive
smoke flows — those need a Playwright/Puppeteer adapter that hasn't landed
yet. The artifact_schema for feat.14 was tightened accordingly: only
``smoke_tests_passed`` is asserted here; ``visual_check_passed`` was
removed because we have no real verifier for it.

Failure modes (each surfaces a clear error string back to Beckman):

- staging_deployment_result artifact missing → fail (feat.13 didn't run)
- artifact present but no ``url`` field → fail
- URL HTTP check fails 5 retries → fail with status / error tail
"""

from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

from mr_roboto.http_check import http_check

logger = get_logger("mr_roboto.staging_smoke_check")


_DEPLOY_ARTIFACT_SUFFIX = "staging_deployment_result"
_URL_FIELD_CANDIDATES = ("url", "staging_url", "deploy_url", "preview_url")


async def _load_artifact(mission_id: int, name: str) -> dict | None:
    try:
        from src.workflows.engine.hooks import get_artifact_store
        store = get_artifact_store()
        raw = await store.retrieve(mission_id, name)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
    except Exception as exc:
        logger.warning("artifact-store retrieve failed: %s", exc)

    try:
        from src.collaboration.blackboard import read_blackboard
        artifacts = await read_blackboard(int(mission_id), "artifacts")
        if isinstance(artifacts, dict):
            raw2 = artifacts.get(name)
            if isinstance(raw2, dict):
                return raw2
            if isinstance(raw2, str) and raw2.strip():
                try:
                    return json.loads(raw2)
                except json.JSONDecodeError:
                    return None
    except Exception as exc:
        logger.warning("blackboard read failed: %s", exc)
    return None


def _resolve_artifact_name(task: dict, payload: dict) -> str | None:
    """Find the staging_deployment_result artifact name to fetch.

    For template-expanded steps the expander prefixes input_artifacts with
    ``{feature_id}__``. We pick whichever input_artifacts entry ends with
    the canonical suffix. Caller may also override via
    ``payload.artifact``.
    """
    explicit = payload.get("artifact")
    if isinstance(explicit, str) and explicit:
        return explicit

    ctx = task.get("context") or {}
    inputs = ctx.get("input_artifacts") or task.get("input_artifacts") or []
    for name in inputs:
        if isinstance(name, str) and name.endswith(_DEPLOY_ARTIFACT_SUFFIX):
            return name
    if _DEPLOY_ARTIFACT_SUFFIX in (inputs or []):
        return _DEPLOY_ARTIFACT_SUFFIX
    return None


def _extract_url(artifact: dict) -> str | None:
    for field in _URL_FIELD_CANDIDATES:
        v = artifact.get(field)
        if isinstance(v, str) and v.startswith(("http://", "https://")):
            return v
    return None


async def run(task: dict[str, Any]) -> dict[str, Any]:
    mission_id = task.get("mission_id")
    if mission_id is None:
        return {"ok": False, "smoke_tests_passed": False, "error": "no mission_id"}

    payload = task.get("payload") or {}
    artifact_name = _resolve_artifact_name(task, payload)
    if not artifact_name:
        return {
            "ok": False,
            "smoke_tests_passed": False,
            "error": (
                f"no input artifact ending with {_DEPLOY_ARTIFACT_SUFFIX!r}; "
                "feat.13 staging_deploy must run first"
            ),
        }

    artifact = await _load_artifact(int(mission_id), artifact_name)
    if not isinstance(artifact, dict) or not artifact:
        return {
            "ok": False,
            "smoke_tests_passed": False,
            "error": f"artifact {artifact_name!r} missing or empty",
        }

    url = _extract_url(artifact)
    if not url:
        return {
            "ok": False,
            "smoke_tests_passed": False,
            "error": (
                f"artifact {artifact_name!r} has no url field; "
                f"tried {list(_URL_FIELD_CANDIDATES)}"
            ),
        }

    res = await http_check(
        url,
        method=str(payload.get("method", "GET")),
        timeout_s=float(payload.get("timeout_s", 15.0)),
        max_attempts=int(payload.get("max_attempts", 5)),
        backoff_base_s=float(payload.get("backoff_base_s", 2.0)),
        backoff_cap_s=float(payload.get("backoff_cap_s", 16.0)),
        expect_body_contains=payload.get("expect_body_contains"),
    )

    return {
        "ok": bool(res.get("ok")),
        "smoke_tests_passed": bool(res.get("ok")),
        "url": url,
        "final_status": res.get("final_status"),
        "attempts": res.get("attempts"),
        "elapsed_s": res.get("elapsed_s"),
        "final_error": res.get("final_error"),
    }
