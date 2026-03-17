# http_integration.py
"""
Generic HTTP/REST integration driven by JSON configuration files.

Each config defines a service_name, base_url, auth pattern, and a set of
actions with method/path/required_params.
"""

import asyncio
import json
import logging
import os
from typing import Any

from .base import BaseIntegration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP client — prefer httpx, fall back to aiohttp, then urllib
# ---------------------------------------------------------------------------

_http_client = None


async def _httpx_request(method: str, url: str, headers: dict,
                         json_body: dict | None = None,
                         params: dict | None = None) -> dict:
    """Make an HTTP request using httpx."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(
            method, url, headers=headers, json=json_body, params=params
        )
        return {
            "status_code": resp.status_code,
            "body": resp.text,
            "headers": dict(resp.headers),
        }


async def _aiohttp_request(method: str, url: str, headers: dict,
                            json_body: dict | None = None,
                            params: dict | None = None) -> dict:
    """Make an HTTP request using aiohttp."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        kwargs: dict[str, Any] = {"headers": headers}
        if json_body is not None:
            kwargs["json"] = json_body
        if params:
            kwargs["params"] = params

        async with session.request(method, url, **kwargs) as resp:
            body = await resp.text()
            return {
                "status_code": resp.status,
                "body": body,
                "headers": dict(resp.headers),
            }


async def _urllib_request(method: str, url: str, headers: dict,
                          json_body: dict | None = None,
                          params: dict | None = None) -> dict:
    """Synchronous fallback using urllib (runs in executor)."""
    import asyncio
    import urllib.request
    import urllib.parse
    import urllib.error

    def _do():
        if params:
            url_with_params = url + "?" + urllib.parse.urlencode(params)
        else:
            url_with_params = url

        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url_with_params, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode()
                return {
                    "status_code": resp.status,
                    "body": body,
                    "headers": dict(resp.headers),
                }
        except urllib.error.HTTPError as e:
            return {
                "status_code": e.code,
                "body": e.read().decode() if e.fp else str(e),
                "headers": dict(e.headers) if e.headers else {},
            }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do)


def _get_http_func():
    """Select the best available HTTP backend."""
    global _http_client
    if _http_client is not None:
        return _http_client

    try:
        import httpx  # noqa: F401
        _http_client = _httpx_request
        return _http_client
    except ImportError:
        pass

    try:
        import aiohttp  # noqa: F401
        _http_client = _aiohttp_request
        return _http_client
    except ImportError:
        pass

    _http_client = _urllib_request
    return _http_client


# ---------------------------------------------------------------------------
# HttpIntegration
# ---------------------------------------------------------------------------

class HttpIntegration(BaseIntegration):
    """
    A generic REST API integration configured via a JSON spec.

    Config format:
    {
        "service_name": "github",
        "base_url": "https://api.github.com",
        "auth_type": "bearer",        # "bearer" | "header" | "query" | "none"
        "auth_header": "Authorization", # header name (for bearer/header)
        "auth_query_param": "token",   # query param name (for query auth)
        "actions": {
            "list_repos": {
                "method": "GET",
                "path": "/user/repos",
                "required_params": []
            }
        }
    }
    """

    def __init__(self, config: dict):
        self._config = config
        self.service_name = config["service_name"]
        self._base_url = config["base_url"].rstrip("/")
        self._auth_type = config.get("auth_type", "bearer")
        self._auth_header = config.get("auth_header", "Authorization")
        self._auth_query_param = config.get("auth_query_param", "token")
        self._actions = config.get("actions", {})

    @classmethod
    def from_config_file(cls, config_path: str) -> "HttpIntegration":
        """Load an HttpIntegration from a JSON config file."""
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config)

    @classmethod
    def from_service_name(cls, service_name: str) -> "HttpIntegration":
        """Load from the built-in configs directory."""
        configs_dir = os.path.join(os.path.dirname(__file__), "configs")
        path = os.path.join(configs_dir, f"{service_name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No config found for service '{service_name}' at {path}"
            )
        return cls.from_config_file(path)

    async def validate(self) -> bool:
        """Check that credentials exist for this service."""
        try:
            from ..security.credential_store import get_credential

            cred = await get_credential(self.service_name)
            return cred is not None
        except Exception:
            return False

    def capabilities(self) -> list[str]:
        """Return the list of configured action names."""
        return sorted(self._actions.keys())

    async def execute(self, action: str, params: dict) -> dict:
        """Execute an API action."""
        if action not in self._actions:
            available = ", ".join(self.capabilities())
            return {
                "status": "error",
                "error": f"Unknown action '{action}'. Available: {available}",
            }

        action_spec = self._actions[action]
        method = action_spec["method"].upper()
        path = action_spec["path"]

        # Check required params
        required = action_spec.get("required_params", [])
        missing = [p for p in required if p not in params]
        if missing:
            return {
                "status": "error",
                "error": f"Missing required params: {missing}",
            }

        # Get credentials
        try:
            from ..security.credential_store import get_credential

            cred = await get_credential(self.service_name)
        except Exception as e:
            return {"status": "error", "error": f"Credential lookup failed: {e}"}

        if cred is None:
            return {
                "status": "error",
                "error": (
                    f"No credentials stored for '{self.service_name}'. "
                    "Use /credential add to store them."
                ),
            }

        # Build URL
        url = self._base_url + path

        # Substitute path parameters like {owner}, {repo}
        for key, value in params.items():
            placeholder = "{" + key + "}"
            if placeholder in url:
                url = url.replace(placeholder, str(value))

        # Build headers and auth
        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "kutay-orchestrator/1.0",
        }

        query_params: dict[str, str] = {}
        token = cred.get("token") or cred.get("api_key") or cred.get("key", "")

        if self._auth_type == "bearer":
            headers[self._auth_header] = f"Bearer {token}"
        elif self._auth_type == "header":
            headers[self._auth_header] = token
        elif self._auth_type == "query":
            query_params[self._auth_query_param] = token

        # Separate body params from query params for GET
        body_params = None
        if method in ("POST", "PUT", "PATCH"):
            # Remove path-substituted params from body
            body_params = {
                k: v for k, v in params.items()
                if "{" + k + "}" not in action_spec["path"]
            }
        elif method == "GET":
            # Non-path params become query params
            for k, v in params.items():
                if "{" + k + "}" not in action_spec["path"]:
                    query_params[k] = str(v)

        # Make the request with retry/backoff for transient failures
        max_retries = action_spec.get("max_retries", 3)
        backoff_delays = [1, 3, 8]  # seconds between retries

        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                http_func = _get_http_func()
                result = await http_func(
                    method, url, headers,
                    json_body=body_params,
                    params=query_params if query_params else None,
                )

                status_code = result["status_code"]
                body = result["body"]

                # Try to parse JSON response
                try:
                    parsed = json.loads(body)
                except (json.JSONDecodeError, ValueError):
                    parsed = body

                if 200 <= status_code < 300:
                    return {"status": "ok", "data": parsed, "status_code": status_code}

                # Retry on transient HTTP errors (429, 500, 502, 503, 504)
                if status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    # Respect Retry-After header if present
                    retry_after = result.get("headers", {}).get("Retry-After")
                    if retry_after:
                        try:
                            delay = min(float(retry_after), 60)
                        except (ValueError, TypeError):
                            delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    else:
                        delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]

                    logger.warning(
                        "[%s] HTTP %d on %s %s — retrying in %.1fs "
                        "(attempt %d/%d)",
                        self.service_name, status_code, method, path,
                        delay, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(delay)
                    last_error = f"HTTP {status_code}"
                    continue

                return {
                    "status": "error",
                    "error": f"HTTP {status_code}",
                    "data": parsed,
                    "status_code": status_code,
                }

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                # Retry on connection/timeout errors
                if attempt < max_retries:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    logger.warning(
                        "[%s] %s on %s %s — retrying in %.1fs "
                        "(attempt %d/%d)",
                        self.service_name, last_error, method, path,
                        delay, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue

                return {
                    "status": "error",
                    "error": last_error,
                }

        return {
            "status": "error",
            "error": f"All {max_retries + 1} attempts failed: {last_error}",
        }
