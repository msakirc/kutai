"""Mobile auth backend — FastAPI route scaffold.

Verifies Apple / Google ID tokens issued to an Expo / React Native app and
exchanges them (plus classic email/password) for the app's own short-lived
JWT access token + a long-lived refresh token.

RECIPE_PARAM markers:
  # RECIPE_PARAM:JWT_ALGO=HS256
  # RECIPE_PARAM:JWT_SECRET_ENV=MOBILE_AUTH_JWT_SECRET
  # RECIPE_PARAM:JWT_TTL_MIN=30
  # RECIPE_PARAM:REFRESH_TTL_DAYS=30
  # RECIPE_PARAM:BCRYPT_COST=12
  # RECIPE_PARAM:APPLE_CLIENT_ID_ENV=APPLE_BUNDLE_ID
  # RECIPE_PARAM:GOOGLE_CLIENT_ID_ENV=GOOGLE_OAUTH_CLIENT_ID
  # RECIPE_PARAM:APP_NAME=MyApp

The template engine replaces RECIPE_PARAM defaults before writing the file
into the mission workspace. Leave the comment markers intact — they must
survive an ast.parse() call so the syntax checker passes before substitution.

Design notes
------------
* The mobile client NEVER receives the refresh token in a cookie — there is no
  browser. Both tokens are returned in the JSON body; the app stores the
  session token in `expo-secure-store` (keychain / keystore).
* Apple / Google ID tokens are verified against the provider JWKS. We do NOT
  trust the `sub`/`email` until the signature, `iss`, `aud` and `exp` checks
  pass. Apple additionally requires the `nonce` to match the one the client
  generated for that sign-in attempt.
"""
from __future__ import annotations

import os
import time
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, field_validator
from passlib.context import CryptContext

# jwt — using python-jose (RECIPE_PARAM:JWT_ALGO=HS256)
# python-jose validates RS256 provider tokens against a JWKS dict directly,
# which keeps the Apple/Google verification path dependency-light.
from jose import jwt, JWTError  # type: ignore[import]

# ---------------------------------------------------------------------------
# Configuration — all secrets/ids come from environment, not template literals
# ---------------------------------------------------------------------------

# RECIPE_PARAM:JWT_SECRET_ENV=MOBILE_AUTH_JWT_SECRET
_JWT_SECRET_ENV = "<<JWT_SECRET_ENV>>"  # noqa: S105 — this is an env var NAME, not a secret
JWT_SECRET = os.environ.get(_JWT_SECRET_ENV, "")
if not JWT_SECRET:
    raise RuntimeError(
        f"Missing required environment variable: {_JWT_SECRET_ENV}. "
        "Set it before starting the server."
    )

JWT_ALGO = "<<JWT_ALGO>>"  # RECIPE_PARAM:JWT_ALGO=HS256
JWT_TTL_MIN = 30      # RECIPE_PARAM:JWT_TTL_MIN=30
REFRESH_TTL_DAYS = 30  # RECIPE_PARAM:REFRESH_TTL_DAYS=30
APP_NAME = "MyApp"    # RECIPE_PARAM:APP_NAME=MyApp

# Provider client-id env var NAMES — the *expected audience* of the ID token.
# RECIPE_PARAM:APPLE_CLIENT_ID_ENV=APPLE_BUNDLE_ID
_APPLE_CLIENT_ID_ENV = "<<APPLE_CLIENT_ID_ENV>>"
# RECIPE_PARAM:GOOGLE_CLIENT_ID_ENV=GOOGLE_OAUTH_CLIENT_ID
_GOOGLE_CLIENT_ID_ENV = "<<GOOGLE_CLIENT_ID_ENV>>"

APPLE_CLIENT_ID = os.environ.get(_APPLE_CLIENT_ID_ENV, "")
GOOGLE_CLIENT_ID = os.environ.get(_GOOGLE_CLIENT_ID_ENV, "")

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"

# JWKS are cached in-process; providers rotate keys ~daily.
_JWKS_TTL_SECONDS = 3600
_jwks_cache: dict[str, tuple[float, dict]] = {}

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

# RECIPE_PARAM:BCRYPT_COST=12
_BCRYPT_ROUNDS_STR = "<<BCRYPT_COST>>"  # substituted by instantiate_recipe
_BCRYPT_ROUNDS = int(_BCRYPT_ROUNDS_STR)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=_BCRYPT_ROUNDS)


def hash_password(plain: str) -> str:
    """Return bcrypt hash of plain-text password."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches hashed."""
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# App JWT helpers — the token the mobile app actually carries
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(user_id: int, ttl_min: int = JWT_TTL_MIN) -> str:
    """Return a signed app JWT access token for user_id."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ttl_min)
    # exp claim must be int (seconds since epoch) — never an ISO string
    payload = {"sub": str(user_id), "exp": int(expire.timestamp()), "type": "access"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def create_refresh_token(user_id: int, ttl_days: int = REFRESH_TTL_DAYS) -> str:
    """Return a signed app JWT refresh token for user_id."""
    expire = datetime.now(timezone.utc) + timedelta(days=ttl_days)
    payload = {"sub": str(user_id), "exp": int(expire.timestamp()), "type": "refresh"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict:
    """Decode and verify an app JWT. Raises HTTPException 401 on failure."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Dependency that validates the app access token and returns {user_id: int}."""
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong token type")
    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing sub")
    return {"user_id": int(user_id_str)}


def _hash_token(raw: str) -> str:
    """SHA-256 hex digest of a raw token string (for storage)."""
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Provider ID-token verification (Apple + Google)
# ---------------------------------------------------------------------------

async def _fetch_jwks(url: str) -> dict:
    """Fetch and cache a provider JWKS document.

    Cached for _JWKS_TTL_SECONDS. Providers rotate signing keys, so the cache
    is short-lived; a verification failure also forces a refetch (see below).
    """
    now = time.time()
    cached = _jwks_cache.get(url)
    if cached and (now - cached[0]) < _JWKS_TTL_SECONDS:
        return cached[1]
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        jwks = resp.json()
    _jwks_cache[url] = (now, jwks)
    return jwks


def _select_jwk(jwks: dict, kid: str) -> Optional[dict]:
    """Return the JWK matching the token's `kid`, or None."""
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key
    return None


async def _verify_provider_id_token(
    id_token: str,
    *,
    jwks_url: str,
    valid_issuers: set[str],
    audience: str,
    expected_nonce: Optional[str] = None,
) -> dict:
    """Verify a provider (Apple/Google) RS256 ID token and return its claims.

    Checks, in order: header `kid` resolves against the JWKS, signature is
    valid, `iss` is one of `valid_issuers`, `aud` equals `audience`, `exp` is
    in the future, and — when supplied — the `nonce` claim equals
    `expected_nonce` (Apple requires this; Google supports it).

    Raises HTTPException 401 on any failure.
    """
    if not audience:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Provider client id not configured on the server",
        )
    try:
        header = jwt.get_unverified_header(id_token)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Malformed ID token") from exc

    kid = header.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="ID token missing kid")

    jwks = await _fetch_jwks(jwks_url)
    jwk = _select_jwk(jwks, kid)
    if jwk is None:
        # Key may have just rotated — drop the cache and retry once.
        _jwks_cache.pop(jwks_url, None)
        jwks = await _fetch_jwks(jwks_url)
        jwk = _select_jwk(jwks, kid)
    if jwk is None:
        raise HTTPException(status_code=401, detail="Unknown signing key for ID token")

    try:
        claims = jwt.decode(
            id_token,
            jwk,
            algorithms=[header.get("alg", "RS256")],
            audience=audience,
            options={"verify_at_hash": False},
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"ID token rejected: {exc}") from exc

    if claims.get("iss") not in valid_issuers:
        raise HTTPException(status_code=401, detail="ID token issuer mismatch")

    if expected_nonce is not None:
        # Apple may store a SHA-256 of the nonce when the client hashed it.
        token_nonce = claims.get("nonce")
        hashed = hashlib.sha256(expected_nonce.encode()).hexdigest()
        if token_nonce not in (expected_nonce, hashed):
            raise HTTPException(status_code=401, detail="ID token nonce mismatch")

    return claims


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class EmailRegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class EmailLoginRequest(BaseModel):
    email: EmailStr
    password: str


class AppleLoginRequest(BaseModel):
    """Payload from `expo-apple-authentication` sign-in."""
    identity_token: str
    nonce: str
    # Apple returns the full name + email ONLY on the very first sign-in.
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class GoogleLoginRequest(BaseModel):
    """Payload from an `expo-auth-session` Google sign-in."""
    id_token: str
    nonce: Optional[str] = None


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    """The session payload the mobile app stores in expo-secure-store."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: int
    expires_in: int  # seconds until the access token expires


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# DB helpers (stub — replace with your actual DB layer during instantiation)
# ---------------------------------------------------------------------------

async def _db_get_user_by_email(email: str) -> Optional[dict]:
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_get_user_by_provider(provider: str, subject: str) -> Optional[dict]:
    """Return the user linked to (provider, provider_subject), or None."""
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_create_user(
    email: Optional[str],
    password_hash: Optional[str],
    provider: str,
    provider_subject: Optional[str],
    display_name: Optional[str],
) -> int:
    """Insert a user row; return new user_id."""
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_create_session(user_id: int, refresh_token_hash: str, expires_at: datetime) -> int:
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_get_session_by_hash(token_hash: str) -> Optional[dict]:
    """Return the session row (with revoked_at / expires_at) or None."""
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_revoke_session_by_token_hash(token_hash: str) -> None:
    raise NotImplementedError("Wire to DB layer during instantiation")


# ---------------------------------------------------------------------------
# Shared session-issuing helper
# ---------------------------------------------------------------------------

async def _issue_session(user_id: int) -> TokenResponse:
    """Mint an access + refresh token pair and persist the refresh session."""
    access_token = create_access_token(user_id)
    refresh_raw = create_refresh_token(user_id)
    refresh_hash = _hash_token(refresh_raw)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TTL_DAYS)
    await _db_create_session(user_id, refresh_hash, expires_at)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_raw,
        user_id=user_id,
        expires_in=JWT_TTL_MIN * 60,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: EmailRegisterRequest) -> TokenResponse:
    """Create an email/password account and return an app session."""
    existing = await _db_get_user_by_email(str(body.email))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    password_hash = hash_password(body.password)
    user_id = await _db_create_user(
        email=str(body.email),
        password_hash=password_hash,
        provider="email",
        provider_subject=None,
        display_name=None,
    )
    return await _issue_session(user_id)


@router.post("/login", response_model=TokenResponse)
async def login(body: EmailLoginRequest) -> TokenResponse:
    """Authenticate with email/password; return an app session."""
    user = await _db_get_user_by_email(str(body.email))
    if not user or not user.get("password_hash"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return await _issue_session(int(user["id"]))


@router.post("/apple", response_model=TokenResponse)
async def apple_login(body: AppleLoginRequest) -> TokenResponse:
    """Verify a Sign in with Apple identity token and return an app session.

    The client (`expo-apple-authentication`) generates a `nonce`, hashes it,
    and passes the hash as `AppleAuthentication.signInAsync({ nonce })`. Apple
    embeds that hash in the identity token. The raw nonce is sent here so the
    server can re-hash and compare — this binds the token to this attempt.
    """
    claims = await _verify_provider_id_token(
        body.identity_token,
        jwks_url=APPLE_JWKS_URL,
        valid_issuers={APPLE_ISSUER},
        audience=APPLE_CLIENT_ID,
        expected_nonce=body.nonce,
    )
    subject = claims.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="Apple token missing sub")

    user = await _db_get_user_by_provider("apple", subject)
    if user is None:
        # Apple only sends the email on the FIRST sign-in — capture it now.
        email = claims.get("email") or (str(body.email) if body.email else None)
        user_id = await _db_create_user(
            email=email,
            password_hash=None,
            provider="apple",
            provider_subject=subject,
            display_name=body.full_name,
        )
    else:
        user_id = int(user["id"])
    return await _issue_session(user_id)


@router.post("/google", response_model=TokenResponse)
async def google_login(body: GoogleLoginRequest) -> TokenResponse:
    """Verify a Sign in with Google ID token and return an app session.

    The `aud` claim of the ID token MUST equal the OAuth client id this app
    registered with Google — a token minted for a different client is
    rejected even if its signature is valid.
    """
    claims = await _verify_provider_id_token(
        body.id_token,
        jwks_url=GOOGLE_JWKS_URL,
        valid_issuers=GOOGLE_ISSUERS,
        audience=GOOGLE_CLIENT_ID,
        expected_nonce=body.nonce,
    )
    subject = claims.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="Google token missing sub")
    if claims.get("email_verified") is False:
        raise HTTPException(status_code=401, detail="Google account email not verified")

    user = await _db_get_user_by_provider("google", subject)
    if user is None:
        user_id = await _db_create_user(
            email=claims.get("email"),
            password_hash=None,
            provider="google",
            provider_subject=subject,
            display_name=claims.get("name"),
        )
    else:
        user_id = int(user["id"])
    return await _issue_session(user_id)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest) -> TokenResponse:
    """Rotate the app session using a refresh token from the request body.

    There is no cookie — the mobile client reads the refresh token from
    `expo-secure-store` and posts it explicitly. The old session row is
    revoked and a new one issued (refresh-token rotation).
    """
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong token type")
    user_id = int(payload["sub"])

    token_hash = _hash_token(body.refresh_token)
    session = await _db_get_session_by_hash(token_hash)
    if not session or session.get("revoked_at"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session revoked or unknown")

    # Rotate: revoke the presented refresh token, mint a fresh pair.
    await _db_revoke_session_by_token_hash(token_hash)
    return await _issue_session(user_id)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    body: RefreshRequest,
    _current: dict = Depends(get_current_user),
) -> MessageResponse:
    """Revoke the refresh session. The app then deletes the secure-store key."""
    token_hash = _hash_token(body.refresh_token)
    await _db_revoke_session_by_token_hash(token_hash)
    return MessageResponse(message="Logged out")


@router.get("/me", response_model=MessageResponse)
async def me(current: dict = Depends(get_current_user)) -> MessageResponse:
    """Trivial protected endpoint — proves the access token is accepted."""
    return MessageResponse(message=f"user {current['user_id']}")
