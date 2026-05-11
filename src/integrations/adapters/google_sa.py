"""Z6 T6C — Google service-account OAuth2 helper.

Google Play Developer API uses OAuth2 with a service account JSON. The
flow:

  1. Build a JWT assertion (RS256) signed by the SA private key, with
     iss=client_email, scope=<scopes>, aud=token_uri.
  2. POST it to the token endpoint as ``grant_type=urn:ietf:params:
     oauth:grant-type:jwt-bearer`` and receive a short-lived access
     token (1 hour).
  3. Use that access token as ``Authorization: Bearer <token>``.

Tokens are cached in-process by (client_email, scopes) until ~60 s before
expiry to avoid minting a JWT on every call.

We prefer ``google-auth`` when available (handles the token cache, the
refresh, and the JWT mint). Otherwise we fall back to manual RS256 +
httpx/urllib for the token exchange.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any

_DEFAULT_SCOPES = ["https://www.googleapis.com/auth/androidpublisher"]
_TOKEN_LIFETIME = 3600
_SAFETY_MARGIN = 60

# Per-process token cache: (client_email, scopes_key) -> (token, exp_epoch)
_TOKEN_CACHE: dict[tuple[str, str], tuple[str, float]] = {}


def _b64url(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _cache_key(sa: dict, scopes: list[str]) -> tuple[str, str]:
    return (sa.get("client_email", ""), ",".join(sorted(scopes)))


def _cached_token(sa: dict, scopes: list[str]) -> str | None:
    key = _cache_key(sa, scopes)
    rec = _TOKEN_CACHE.get(key)
    if not rec:
        return None
    token, exp = rec
    if exp - _SAFETY_MARGIN > time.time():
        return token
    _TOKEN_CACHE.pop(key, None)
    return None


def _store_token(sa: dict, scopes: list[str], token: str, ttl: int) -> None:
    key = _cache_key(sa, scopes)
    _TOKEN_CACHE[key] = (token, time.time() + ttl)


def _mint_assertion(sa: dict, scopes: list[str]) -> str:
    """Build an RS256 JWT assertion using the cryptography library."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    private_key = serialization.load_pem_private_key(
        sa["private_key"].encode(), password=None,
    )
    now = int(time.time())
    token_uri = sa.get("token_uri", "https://oauth2.googleapis.com/token")
    header = {"alg": "RS256", "typ": "JWT"}
    if sa.get("private_key_id"):
        header["kid"] = sa["private_key_id"]
    payload = {
        "iss": sa["client_email"],
        "scope": " ".join(scopes),
        "aud": token_uri,
        "iat": now,
        "exp": now + _TOKEN_LIFETIME,
    }
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + b"."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )
    sig = private_key.sign(
        signing_input, padding.PKCS1v15(), hashes.SHA256(),
    )
    return (signing_input + b"." + _b64url(sig)).decode("ascii")


async def _exchange_assertion(assertion: str, token_uri: str) -> tuple[str, int]:
    """POST the assertion and return (access_token, expires_in)."""
    body = (
        f"grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer"
        f"&assertion={assertion}"
    )
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(token_uri, headers=headers, content=body)
            data = resp.json()
    except ImportError:
        import urllib.request
        import urllib.error
        import asyncio

        def _do() -> dict:
            req = urllib.request.Request(
                token_uri,
                data=body.encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())

        data = await asyncio.get_event_loop().run_in_executor(None, _do)

    if "access_token" not in data:
        raise RuntimeError(
            f"Google token exchange failed: "
            f"{data.get('error') or data}"
        )
    return data["access_token"], int(data.get("expires_in", _TOKEN_LIFETIME))


async def mint_google_oauth_token(
    service_account_json: dict[str, Any],
    scopes: list[str] | None = None,
) -> str:
    """Return a fresh OAuth2 access token for the given service account.

    Cached per (client_email, scopes) until ~60 s before expiry. Raises
    RuntimeError if no crypto backend is available, ValueError on bad
    input.
    """
    if not isinstance(service_account_json, dict):
        raise ValueError("service_account_json must be a dict")
    for k in ("client_email", "private_key"):
        if not service_account_json.get(k):
            raise ValueError(
                f"service_account_json missing required field {k!r}"
            )
    scopes = list(scopes) if scopes else _DEFAULT_SCOPES

    cached = _cached_token(service_account_json, scopes)
    if cached:
        return cached

    # Prefer google-auth when present — handles refresh + cache itself.
    try:
        from google.oauth2 import service_account as ga_sa  # type: ignore
        import google.auth.transport.requests  # type: ignore

        creds = ga_sa.Credentials.from_service_account_info(
            service_account_json, scopes=scopes,
        )
        # google-auth refresh is sync; run in executor to keep async caller.
        import asyncio

        def _refresh() -> str:
            req = google.auth.transport.requests.Request()
            creds.refresh(req)
            return creds.token

        token = await asyncio.get_event_loop().run_in_executor(None, _refresh)
        _store_token(service_account_json, scopes, token, _TOKEN_LIFETIME)
        return token
    except ImportError:
        pass

    # Manual path needs cryptography.
    try:
        import cryptography  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Google service-account auth requires either google-auth or "
            "the cryptography package. Install one (`pip install "
            "google-auth` or `pip install cryptography`)."
        ) from e

    assertion = _mint_assertion(service_account_json, scopes)
    token_uri = service_account_json.get(
        "token_uri", "https://oauth2.googleapis.com/token",
    )
    token, ttl = await _exchange_assertion(assertion, token_uri)
    _store_token(service_account_json, scopes, token, ttl)
    return token


async def auth_header_from_credential(cred: dict[str, Any]) -> str:
    """Convenience used by HttpIntegration for ``auth_type=='oauth_service_account'``."""
    sa = cred.get("service_account_json") or cred
    scopes = cred.get("scopes") or None
    token = await mint_google_oauth_token(sa, scopes)
    return f"Bearer {token}"


def _clear_cache_for_tests() -> None:
    _TOKEN_CACHE.clear()


__all__ = [
    "mint_google_oauth_token",
    "auth_header_from_credential",
    "_clear_cache_for_tests",
]
