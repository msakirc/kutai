"""Fetch a URL and extract Open Graph + Twitter Card meta tags.

Mechanical executor. No LLM. Used by social-preview verification steps to
ground the "do my OG tags actually serve" question against the live page.

Honest scope: this verifies the **served HTML contains the meta tags**.
It does NOT replace platform debuggers (Facebook Sharing Debugger / Twitter
Card Validator / LinkedIn Post Inspector) which crawl with their own UA,
respect platform-specific quirks, and refresh their own caches. A pass here
means tags are present + image URL is reachable; a real platform may still
reject (e.g. image too small, title too long).
"""

from __future__ import annotations

import re
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.parse_og_tags")


_DEFAULT_TIMEOUT = 15.0
_DEFAULT_UA = (
    "Mozilla/5.0 (compatible; KutAI-PreviewCheck/1.0; "
    "+social-preview-test)"
)

# Required by all three big platforms. Missing any of these fails the gate.
_REQUIRED_OG = ("og:title", "og:description", "og:image")

_META_RE = re.compile(
    r"<meta\s+[^>]*?"
    r"(?:property|name)\s*=\s*[\"']([^\"']+)[\"']"
    r"[^>]*?content\s*=\s*[\"']([^\"']*)[\"']",
    re.IGNORECASE | re.DOTALL,
)
_META_RE_REVERSED = re.compile(
    r"<meta\s+[^>]*?"
    r"content\s*=\s*[\"']([^\"']*)[\"']"
    r"[^>]*?(?:property|name)\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE | re.DOTALL,
)


def _extract_meta(html: str) -> dict[str, str]:
    """Extract <meta property=... content=...> and <meta name=... content=...>.

    Pure regex — we don't need a full HTML parser for this and pulling in
    BeautifulSoup just for tag attributes is overkill. The two regexes
    handle the two common attribute orderings.
    """
    out: dict[str, str] = {}
    for m in _META_RE.finditer(html):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if key and key not in out:
            out[key] = val
    for m in _META_RE_REVERSED.finditer(html):
        key = m.group(2).strip().lower()
        val = m.group(1).strip()
        if key and key not in out:
            out[key] = val
    # Title fallback (some sites only set <title> without og:title).
    if "og:title" not in out:
        tm = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if tm:
            out["title"] = tm.group(1).strip()
    return out


async def _fetch(url: str, timeout_s: float) -> tuple[int, str, str | None]:
    """Return (status, body, error). Status 0 means transport failure."""
    import aiohttp

    headers = {"User-Agent": _DEFAULT_UA, "Accept": "text/html,*/*"}
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as s:
            async with s.get(url, allow_redirects=True) as resp:
                body = await resp.text(errors="replace")
                return resp.status, body, None
    except Exception as e:
        return 0, "", f"{type(e).__name__}: {e}"


async def _head(url: str, timeout_s: float) -> tuple[int, str | None]:
    import aiohttp

    headers = {"User-Agent": _DEFAULT_UA}
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as s:
            async with s.head(url, allow_redirects=True) as resp:
                return resp.status, None
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


async def parse_og_tags(
    url: str,
    timeout_s: float = _DEFAULT_TIMEOUT,
    check_image: bool = True,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch ``url``, extract OG/Twitter meta tags, optionally HEAD og:image.

    Returns
    -------
    dict
        ``{"ok", "url", "status", "tags", "missing", "image_reachable",
        "errors"}``. ``ok`` is True iff fetch returned 2xx, all required
        tags are present + non-empty, and (when ``check_image``) og:image
        HEAD returned 2xx.
    """
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return {
            "ok": False,
            "url": url,
            "status": 0,
            "tags": {},
            "missing": list(required or _REQUIRED_OG),
            "image_reachable": None,
            "errors": ["invalid url"],
        }

    required_keys = list(required) if required else list(_REQUIRED_OG)
    errors: list[str] = []

    status, body, fetch_err = await _fetch(url, timeout_s)
    if fetch_err:
        errors.append(f"fetch: {fetch_err}")
    if status < 200 or status >= 300:
        if not fetch_err:
            errors.append(f"fetch status {status}")

    tags = _extract_meta(body) if body else {}
    missing = [k for k in required_keys if not tags.get(k)]

    image_reachable: bool | None = None
    if check_image and not missing and tags.get("og:image"):
        img_status, img_err = await _head(tags["og:image"], timeout_s)
        if img_err:
            errors.append(f"image HEAD: {img_err}")
            image_reachable = False
        else:
            image_reachable = 200 <= img_status < 300
            if not image_reachable:
                errors.append(f"og:image HEAD status {img_status}")

    ok = (
        200 <= status < 300
        and not missing
        and (not check_image or image_reachable is True)
    )

    return {
        "ok": ok,
        "url": url,
        "status": status,
        "tags": tags,
        "missing": missing,
        "image_reachable": image_reachable,
        "errors": errors,
    }
