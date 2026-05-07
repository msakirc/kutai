"""HTTP health check with retry/backoff.

Mechanical executor. No LLM. Used by staging-validation steps to actually
hit a deployment URL and confirm it responds, instead of letting an LLM
narrate "health check passed".

Behaviour:

- GET (or HEAD if ``method="HEAD"``) the URL with timeout.
- Retry on transport failure, 408, 425, 429, 5xx — exponential backoff
  capped at ``max_attempts`` total tries.
- 4xx (except retry-eligible) are treated as fail-fast — they won't fix
  themselves on retry.
- ``ok`` is True iff the final response status falls in ``expect_status``
  (default 200..299) AND optional ``expect_body_contains`` substring is
  present.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("salako.http_check")


_DEFAULT_TIMEOUT = 15.0
_DEFAULT_UA = "Mozilla/5.0 (compatible; KutAI-HealthCheck/1.0)"
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _matches(status: int, expect: tuple[int, int] | list[int]) -> bool:
    if isinstance(expect, tuple) and len(expect) == 2:
        lo, hi = expect
        return lo <= status <= hi
    if isinstance(expect, (list, set, frozenset)):
        return status in expect
    return False


async def http_check(
    url: str,
    method: str = "GET",
    timeout_s: float = _DEFAULT_TIMEOUT,
    max_attempts: int = 5,
    backoff_base_s: float = 1.0,
    backoff_cap_s: float = 8.0,
    expect_status: tuple[int, int] | list[int] = (200, 299),
    expect_body_contains: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Hit ``url`` with retry. Returns a verdict dict.

    Returns
    -------
    dict
        ``{"ok", "url", "method", "attempts", "final_status", "final_error",
        "elapsed_s", "body_match"}``.
    """
    import aiohttp

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return {
            "ok": False,
            "url": url,
            "method": method,
            "attempts": 0,
            "final_status": 0,
            "final_error": "invalid url",
            "elapsed_s": 0.0,
            "body_match": None,
        }

    method_u = method.upper()
    if method_u not in ("GET", "HEAD"):
        return {
            "ok": False,
            "url": url,
            "method": method,
            "attempts": 0,
            "final_status": 0,
            "final_error": f"unsupported method {method!r}",
            "elapsed_s": 0.0,
            "body_match": None,
        }

    req_headers = {"User-Agent": _DEFAULT_UA}
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items()})

    loop = asyncio.get_event_loop()
    started = loop.time()
    last_status = 0
    last_error: str | None = None
    body_match: bool | None = None

    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with aiohttp.ClientSession(
                timeout=timeout, headers=req_headers,
            ) as session:
                req = session.get(url) if method_u == "GET" else session.head(url)
                async with req as resp:
                    last_status = resp.status
                    last_error = None
                    body_text = ""
                    if method_u == "GET" and expect_body_contains is not None:
                        # Cap body read at 256 KB — health endpoints should
                        # not return novels; runaway bodies shouldn't OOM us.
                        body_text = (await resp.content.read(256 * 1024)).decode(
                            "utf-8", errors="replace"
                        )
                        body_match = expect_body_contains in body_text
                    if last_status in _RETRY_STATUS and attempt < max_attempts:
                        raise _Retryable(f"retry status {last_status}")
                    break
        except _Retryable as r:
            last_error = str(r)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt >= max_attempts:
                break

        delay = min(backoff_cap_s, backoff_base_s * (2 ** (attempt - 1)))
        await asyncio.sleep(delay)

    elapsed = round(loop.time() - started, 3)
    status_ok = _matches(last_status, expect_status)
    ok = (
        status_ok
        and last_error is None
        and (expect_body_contains is None or body_match is True)
    )

    return {
        "ok": ok,
        "url": url,
        "method": method_u,
        "attempts": attempt,
        "final_status": last_status,
        "final_error": last_error,
        "elapsed_s": elapsed,
        "body_match": body_match,
    }


class _Retryable(Exception):
    """Internal sentinel — used so retry-eligible 5xx flow into the same
    backoff path as transport errors without us hand-rolling control flow."""
