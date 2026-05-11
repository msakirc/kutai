"""Z6 T6C — Apple App Store Connect JWT mint.

App Store Connect doesn't accept static bearer tokens. Each request must
carry a fresh ES256-signed JWT whose claims declare the team_id (issuer),
audience ``appstoreconnect-v1``, key id (header kid), and an expiry no
more than 20 minutes out.

Stored credential shape (see ``credential_schemas/apple_appstore.json``):

    {
        "team_id":         "<Apple Team ID>",
        "key_id":          "<API Key ID>",
        "private_key_pem": "<contents of the AuthKey_*.p8 file>"
    }

Prefer PyJWT when installed; fall back to a manual JWS using
``cryptography``. Both deps are optional — we raise a clear error at
call time, never at import time, so admission tests don't crash for
unrelated reasons.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any

_APPLE_AUDIENCE = "appstoreconnect-v1"
_DEFAULT_TTL = 1200  # 20 min — Apple max


def _b64url(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _now() -> int:
    return int(time.time())


def _mint_with_pyjwt(team_id: str, key_id: str, private_key_pem: str,
                    expires_in_seconds: int) -> str:
    import jwt  # type: ignore

    now = _now()
    payload = {
        "iss": team_id,
        "iat": now,
        "exp": now + expires_in_seconds,
        "aud": _APPLE_AUDIENCE,
    }
    headers = {"kid": key_id, "typ": "JWT"}
    token = jwt.encode(
        payload, private_key_pem, algorithm="ES256", headers=headers,
    )
    if isinstance(token, bytes):
        token = token.decode("ascii")
    return token


def _mint_with_cryptography(team_id: str, key_id: str, private_key_pem: str,
                            expires_in_seconds: int) -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    key = serialization.load_pem_private_key(
        private_key_pem.encode() if isinstance(private_key_pem, str)
        else private_key_pem,
        password=None,
    )
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError(
            "apple_appstore private key must be an EC P-256 key (ES256). "
            f"got {type(key).__name__}"
        )

    now = _now()
    header = {"alg": "ES256", "typ": "JWT", "kid": key_id}
    payload = {
        "iss": team_id,
        "iat": now,
        "exp": now + expires_in_seconds,
        "aud": _APPLE_AUDIENCE,
    }
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + b"."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )
    der_sig = key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    r, s = utils.decode_dss_signature(der_sig)
    # JOSE expects fixed-length 32-byte r||s for P-256.
    raw_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return (signing_input + b"." + _b64url(raw_sig)).decode("ascii")


def mint_apple_jwt(
    team_id: str,
    key_id: str,
    private_key_pem: str,
    expires_in_seconds: int = _DEFAULT_TTL,
) -> str:
    """Return a fresh ES256 JWT for App Store Connect.

    Raises ValueError if required inputs are missing, RuntimeError if no
    crypto backend is available.
    """
    if not team_id or not key_id or not private_key_pem:
        raise ValueError(
            "mint_apple_jwt requires team_id, key_id, private_key_pem"
        )
    if expires_in_seconds <= 0 or expires_in_seconds > _DEFAULT_TTL:
        expires_in_seconds = _DEFAULT_TTL

    # Prefer PyJWT.
    try:
        import jwt  # noqa: F401
        return _mint_with_pyjwt(
            team_id, key_id, private_key_pem, expires_in_seconds,
        )
    except ImportError:
        pass

    # Fallback to cryptography.
    try:
        import cryptography  # noqa: F401
        return _mint_with_cryptography(
            team_id, key_id, private_key_pem, expires_in_seconds,
        )
    except ImportError as e:
        raise RuntimeError(
            "Apple App Store Connect requires either PyJWT or the "
            "cryptography package to mint JWTs. Install one of them "
            "(`pip install PyJWT` or `pip install cryptography`)."
        ) from e


def auth_header_from_credential(cred: dict[str, Any]) -> str:
    """Convenience: build the Authorization header value from a stored
    credential. Used by ``HttpIntegration`` when ``auth_type=='jwt_p8'``.
    """
    tok = mint_apple_jwt(
        cred["team_id"],
        cred["key_id"],
        cred["private_key_pem"],
    )
    return f"Bearer {tok}"


__all__ = ["mint_apple_jwt", "auth_header_from_credential"]
