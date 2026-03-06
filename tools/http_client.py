# tools/http_client.py
"""
Generic HTTP client tool for agents to interact with REST APIs.
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

MAX_RESPONSE_SIZE = 10_000  # chars


async def http_request(
    method: str,
    url: str,
    headers: dict | None = None,
    body: str | None = None,
    timeout: int = 30,
) -> str:
    """
    Make an HTTP request and return the response.

    Args:
        method:  HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD).
        url:     Full URL to request.
        headers: Optional dict of HTTP headers.
        body:    Optional request body (JSON string or raw text).
        timeout: Request timeout in seconds (max 60).

    Returns:
        Response status, headers summary, and body.
    """
    method = method.upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
        return f"❌ Unsupported method: {method}"

    if not url or not url.startswith(("http://", "https://")):
        return "❌ URL must start with http:// or https://"

    timeout = min(timeout, 60)

    try:
        import httpx
    except ImportError:
        return await _curl_fallback(method, url, headers, body, timeout)

    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True,
        ) as client:
            kwargs: dict = {"method": method, "url": url}
            if headers:
                kwargs["headers"] = headers
            if body and method in ("POST", "PUT", "PATCH"):
                # Auto-detect JSON
                try:
                    json.loads(body)
                    kwargs["content"] = body
                    if not headers or "content-type" not in {
                        k.lower() for k in headers
                    }:
                        kwargs.setdefault("headers", {})
                        kwargs["headers"]["Content-Type"] = "application/json"
                except (json.JSONDecodeError, TypeError):
                    kwargs["content"] = body

            response = await client.request(**kwargs)

        return _format_response(response.status_code, dict(response.headers), response.text)

    except httpx.TimeoutException:
        return f"❌ Request timed out after {timeout}s"
    except Exception as exc:
        return f"❌ Request failed: {exc}"


async def _curl_fallback(
    method: str, url: str, headers: dict | None,
    body: str | None, timeout: int,
) -> str:
    """Fallback to curl when httpx is not available."""
    cmd = ["curl", "-s", "-w", "\n%{http_code}", "-X", method]
    cmd += ["-m", str(timeout)]
    cmd += ["--max-filesize", "1000000"]  # 1MB limit

    if headers:
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]
    if body and method in ("POST", "PUT", "PATCH"):
        cmd += ["-d", body]

    cmd.append(url)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout + 5,
        )
        output = stdout.decode("utf-8", errors="replace")
        lines = output.rsplit("\n", 1)
        if len(lines) == 2:
            resp_body = lines[0]
            try:
                status_code = int(lines[1].strip())
            except ValueError:
                status_code = 0
                resp_body = output
        else:
            resp_body = output
            status_code = 0

        return _format_response(status_code, {}, resp_body)

    except asyncio.TimeoutError:
        return f"❌ curl timed out after {timeout}s"
    except Exception as exc:
        return f"❌ curl failed: {exc}"


def _format_response(
    status_code: int, headers: dict, body: str,
) -> str:
    """Format HTTP response for agent consumption."""
    status_emoji = "✅" if 200 <= status_code < 400 else "❌"
    parts = [f"{status_emoji} HTTP {status_code}"]

    # Show content-type if available
    ct = headers.get("content-type", headers.get("Content-Type", ""))
    if ct:
        parts.append(f"Content-Type: {ct}")

    # Truncate body
    if len(body) > MAX_RESPONSE_SIZE:
        body = body[:MAX_RESPONSE_SIZE] + f"\n... [{len(body)} chars total, truncated]"

    parts.append(f"\n{body}")
    return "\n".join(parts)
