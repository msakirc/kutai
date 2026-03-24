"""Deployment tool for workflow-produced applications.

Uses the integration layer (Gap 5) to deploy to cloud platforms.
Validates prerequisites before deploying.
"""

import asyncio
import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("tools.deploy")

SUPPORTED_TARGETS = ("vercel", "railway")

# Retry settings for post-deploy health check
_HEALTH_CHECK_RETRIES = 3
_HEALTH_CHECK_DELAY = 10  # seconds between retries
_HEALTH_CHECK_TOTAL_TIMEOUT = 45  # total seconds for all retries combined


async def _check_credential(target: str) -> dict | None:
    """Check if credentials exist for the target platform."""
    try:
        from ..security.credential_store import get_credential
        return await get_credential(target)
    except Exception as exc:
        logger.warning("Credential check failed for %s: %s", target, exc)
        return None


async def _check_quality_gate(mission_id: int, artifact_store: Any) -> tuple[bool, str]:
    """Check that the phase_13 quality gate has passed.

    Returns (passed, message).
    """
    result = await artifact_store.retrieve(mission_id, "phase_13_gate_result")
    if result is None:
        return False, "Quality gate artifact 'phase_13_gate_result' not found"

    # Accept JSON or plain text
    try:
        data = json.loads(result)
        if isinstance(data, dict) and data.get("passed"):
            return True, "Quality gate passed"
        if isinstance(data, dict):
            return False, f"Quality gate not passed: {data}"
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    upper = result.strip().upper()
    if upper == "PASSED":
        return True, "Quality gate passed"

    return False, f"Quality gate not passed: {result}"


async def _health_check(url: str) -> dict:
    """HTTP GET health check with retries.

    Returns dict with 'healthy' (bool), 'status_code' (int|None),
    'attempts' (int), and 'error' (str|None).
    """
    last_error: str | None = None

    for attempt in range(1, _HEALTH_CHECK_RETRIES + 1):
        try:
            # Lazy import — prefer httpx, fall back to urllib
            try:
                import httpx
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(url)
                    if 200 <= resp.status_code < 400:
                        return {
                            "healthy": True,
                            "status_code": resp.status_code,
                            "attempts": attempt,
                            "error": None,
                        }
                    last_error = f"HTTP {resp.status_code}"
            except ImportError:
                import urllib.request
                import urllib.error

                def _do_get():
                    req = urllib.request.Request(url, method="GET")
                    try:
                        with urllib.request.urlopen(req, timeout=15) as resp:
                            return resp.status
                    except urllib.error.HTTPError as e:
                        return e.code

                loop = asyncio.get_event_loop()
                status = await loop.run_in_executor(None, _do_get)
                if 200 <= status < 400:
                    return {
                        "healthy": True,
                        "status_code": status,
                        "attempts": attempt,
                        "error": None,
                    }
                last_error = f"HTTP {status}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < _HEALTH_CHECK_RETRIES:
            await asyncio.sleep(_HEALTH_CHECK_DELAY)

    return {
        "healthy": False,
        "status_code": None,
        "attempts": _HEALTH_CHECK_RETRIES,
        "error": last_error,
    }


def _extract_url(target: str, deploy_response: dict) -> str | None:
    """Extract the deployed URL from the integration response."""
    data = deploy_response.get("data", {})

    if not isinstance(data, dict):
        return None

    # Vercel: data.url or data.alias[0]
    if target == "vercel":
        url = data.get("url")
        if url:
            if not url.startswith("http"):
                url = f"https://{url}"
            return url
        aliases = data.get("alias", [])
        if aliases:
            alias = aliases[0]
            if not alias.startswith("http"):
                alias = f"https://{alias}"
            return alias

    # Railway: look for common URL fields
    if target == "railway":
        url = data.get("url") or data.get("serviceUrl") or data.get("deploymentUrl")
        if url:
            if not url.startswith("http"):
                url = f"https://{url}"
            return url

    # Generic fallback
    for key in ("url", "deployment_url", "deploymentUrl", "serviceUrl"):
        val = data.get(key)
        if val and isinstance(val, str):
            if not val.startswith("http"):
                val = f"https://{val}"
            return val

    return None


async def deploy(
    target: str,
    project_path: str,
    env_vars: dict | None = None,
    mission_id: int | None = None,
) -> dict:
    """Deploy a project to the target platform.

    Steps:
    1. Pre-deploy validation (credentials, quality gate)
    2. Execute deployment via integration layer
    3. Post-deploy verification (health check)
    4. Store result as artifact if mission_id provided

    Returns a result dict with status, url, verification, and error info.
    """
    if target not in SUPPORTED_TARGETS:
        return {
            "status": "error",
            "error": f"Unsupported target '{target}'. Supported: {', '.join(SUPPORTED_TARGETS)}",
        }

    # ── 1. Pre-deploy validation ──────────────────────────────────────────
    cred = await _check_credential(target)
    if cred is None:
        return {
            "status": "error",
            "error": f"No credentials found for '{target}'. Use /credential add to store them.",
        }

    # Quality gate check (only when mission_id is provided)
    if mission_id is not None:
        try:
            from ..workflows.engine.artifacts import ArtifactStore
            artifact_store = ArtifactStore(use_db=True)
            gate_passed, gate_msg = await _check_quality_gate(mission_id, artifact_store)
            if not gate_passed:
                return {
                    "status": "error",
                    "error": f"Pre-deploy quality gate failed: {gate_msg}",
                }
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Quality gate check error: {exc}",
            }

    # ── 2. Execute deployment ─────────────────────────────────────────────
    try:
        from ..integrations.registry import get_integration_registry

        registry = get_integration_registry()
        integration = registry.get(target)
        if integration is None:
            return {
                "status": "error",
                "error": f"No integration configured for '{target}'.",
            }

        # Build deploy params based on target
        deploy_params: dict[str, Any] = {}
        if target == "vercel":
            deploy_params["name"] = project_path.rstrip("/").split("/")[-1]
            deploy_params["gitSource"] = {"type": "github", "ref": "main"}
            if env_vars:
                deploy_params["env"] = env_vars
        elif target == "railway":
            deploy_params["query"] = (
                'mutation { deploymentCreate(input: {serviceId: "auto"}) { id status } }'
            )
            if env_vars:
                deploy_params["variables"] = env_vars

        result = await integration.execute("deploy", deploy_params)

        if result.get("status") == "error":
            return {
                "status": "error",
                "error": f"Deployment failed: {result.get('error', 'unknown')}",
                "details": result,
            }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Deployment execution error: {type(exc).__name__}: {exc}",
        }

    # ── 3. Post-deploy verification ───────────────────────────────────────
    deployed_url = _extract_url(target, result)
    verification = None

    if deployed_url:
        try:
            verification = await asyncio.wait_for(
                _health_check(deployed_url),
                timeout=_HEALTH_CHECK_TOTAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            verification = {
                "healthy": False,
                "status_code": None,
                "attempts": _HEALTH_CHECK_RETRIES,
                "error": f"Health check timed out after {_HEALTH_CHECK_TOTAL_TIMEOUT}s",
            }
    else:
        verification = {
            "healthy": None,
            "status_code": None,
            "attempts": 0,
            "error": "Could not extract deployment URL for health check",
        }

    # ── 4. Build final result and store artifact ──────────────────────────
    deploy_result = {
        "status": "ok",
        "target": target,
        "url": deployed_url,
        "verification": verification,
        "deploy_response": result,
    }

    if mission_id is not None:
        try:
            from ..workflows.engine.artifacts import ArtifactStore
            artifact_store = ArtifactStore(use_db=True)
            await artifact_store.store(
                mission_id, "deployment_result", json.dumps(deploy_result, default=str)
            )
        except Exception as exc:
            logger.warning("Failed to store deployment artifact: %s", exc)

    return deploy_result
