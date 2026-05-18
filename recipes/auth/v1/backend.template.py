"""Auth backend — FastAPI route scaffold.

RECIPE_PARAM markers:
  # RECIPE_PARAM:JWT_ALGO=HS256
  # RECIPE_PARAM:JWT_SECRET_ENV=AUTH_JWT_SECRET
  # RECIPE_PARAM:JWT_TTL_MIN=15
  # RECIPE_PARAM:REFRESH_TTL_DAYS=7
  # RECIPE_PARAM:BCRYPT_COST=12
  # RECIPE_PARAM:LOGIN_RATE_LIMIT=10/minute
  # RECIPE_PARAM:APP_NAME=MyApp

The template engine replaces RECIPE_PARAM defaults before writing the file
into the mission workspace. Leave the comment markers intact — they must
survive an ast.parse() call so the syntax checker passes before substitution.
"""
from __future__ import annotations

import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, Cookie, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, field_validator
from passlib.context import CryptContext

# jwt — using python-jose (RECIPE_PARAM:JWT_ALGO=HS256)
# Rationale: python-jose has a cleaner async-compatible API than PyJWT for
# FastAPI's Depends pattern; both are acceptable — swap by replacing the
# jose import and the encode/decode calls below.
from jose import jwt, JWTError  # type: ignore[import]

# ---------------------------------------------------------------------------
# Configuration — all secrets MUST come from environment, not template literals
# ---------------------------------------------------------------------------

# RECIPE_PARAM:JWT_SECRET_ENV=AUTH_JWT_SECRET
_JWT_SECRET_ENV = "<<JWT_SECRET_ENV>>"  # noqa: S105 — this is an env var NAME, not a secret
JWT_SECRET = os.environ.get(_JWT_SECRET_ENV, "")
if not JWT_SECRET:
    raise RuntimeError(
        f"Missing required environment variable: {_JWT_SECRET_ENV}. "
        "Set it before starting the server."
    )

JWT_ALGO = "<<JWT_ALGO>>"  # RECIPE_PARAM:JWT_ALGO=HS256
JWT_TTL_MIN = 15    # RECIPE_PARAM:JWT_TTL_MIN=15
REFRESH_TTL_DAYS = 7  # RECIPE_PARAM:REFRESH_TTL_DAYS=7

APP_NAME = "MyApp"  # RECIPE_PARAM:APP_NAME=MyApp

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

# RECIPE_PARAM:BCRYPT_COST=12
# Swap path for argon2id: replace schemes=["bcrypt"] with schemes=["argon2"]
# and add argon2-cffi to requirements.txt. No other changes needed.
_BCRYPT_ROUNDS_STR = "<<BCRYPT_COST>>"  # substituted by instantiate_recipe; used below
_BCRYPT_ROUNDS = int(_BCRYPT_ROUNDS_STR)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=_BCRYPT_ROUNDS)


def hash_password(plain: str) -> str:
    """Return bcrypt hash of plain-text password."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches hashed."""
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(user_id: int, ttl_min: int = JWT_TTL_MIN) -> str:
    """Return a signed JWT access token for user_id."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ttl_min)
    # exp claim must be int (seconds since epoch) — Pydantic v2 rejects ISO strings
    payload = {"sub": str(user_id), "exp": int(expire.timestamp()), "type": "access"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def create_refresh_token(user_id: int, ttl_days: int = REFRESH_TTL_DAYS) -> str:
    """Return a signed JWT refresh token for user_id."""
    expire = datetime.now(timezone.utc) + timedelta(days=ttl_days)
    payload = {"sub": str(user_id), "exp": int(expire.timestamp()), "type": "refresh"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT. Raises HTTPException 401 on failure."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ---------------------------------------------------------------------------
# Dependency: current user
# ---------------------------------------------------------------------------

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Dependency that validates the access token and returns {user_id: int}."""
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong token type")
    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing sub")
    return {"user_id": int(user_id_str)}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class RegisterResponse(BaseModel):
    user_id: int
    email: str
    message: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int


class TokenRefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PasswordResetRequestRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def pw_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class EmailVerifyRequest(BaseModel):
    token: str


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# DB helpers (stub — replace with your actual DB layer)
# ---------------------------------------------------------------------------
# These are thin stubs. In the instantiated recipe the planner wires them to
# the project's aiosqlite/asyncpg layer.  The stubs keep the module importable
# during ast.parse() and smoke tests.

async def _db_create_user(email: str, password_hash: str) -> int:
    """Insert user row; return new user_id. Raises on duplicate email."""
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_get_user_by_email(email: str) -> Optional[dict]:
    """Return user dict or None."""
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_get_user_by_id(user_id: int) -> Optional[dict]:
    """Return user dict or None."""
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_create_session(user_id: int, refresh_token_hash: str, expires_at: datetime) -> int:
    """Insert session row; return session_id."""
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_revoke_session_by_token_hash(token_hash: str) -> None:
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_create_reset_token(user_id: int, token_hash: str, expires_at: datetime) -> None:
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_consume_reset_token(token_hash: str) -> Optional[int]:
    """Mark used; return user_id or None if not found/expired/already used."""
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_update_password(user_id: int, password_hash: str) -> None:
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_create_verify_token(user_id: int, token_hash: str, expires_at: datetime) -> None:
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_consume_verify_token(token_hash: str) -> Optional[int]:
    """Mark used; return user_id or None."""
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _db_set_email_verified(user_id: int) -> None:
    raise NotImplementedError("Wire to DB layer during instantiation")


async def _send_email(to: str, subject: str, body: str) -> None:
    """Send transactional email. Wire to SendGrid / SES / SMTP during instantiation."""
    raise NotImplementedError("Wire to email provider during instantiation")


def _hash_token(raw: str) -> str:
    """SHA-256 hex digest of a raw token string (for storage)."""
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest) -> RegisterResponse:
    """Create a new user account and send email verification."""
    password_hash = hash_password(body.password)
    try:
        user_id = await _db_create_user(str(body.email), password_hash)
    except Exception as exc:
        # Duplicate email — surface as 409
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        ) from exc

    # Send verification email
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    await _db_create_verify_token(user_id, token_hash, expires_at)
    await _send_email(
        to=str(body.email),
        subject=f"Verify your {APP_NAME} account",
        body=f"Click to verify: https://example.com/verify?token={raw_token}",
        # RECIPE_PARAM: replace example.com with your domain
    )
    return RegisterResponse(
        user_id=user_id,
        email=str(body.email),
        message="Account created. Check your email to verify.",
    )


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, response: Response) -> LoginResponse:
    """Authenticate; return access token + set refresh token cookie.

    Rate limited: RECIPE_PARAM:LOGIN_RATE_LIMIT=10/minute
    Wire slowapi limiter to this route during instantiation.
    """
    user = await _db_get_user_by_email(str(body.email))
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    user_id: int = user["id"]
    access_token = create_access_token(user_id)
    refresh_raw = create_refresh_token(user_id)
    refresh_hash = _hash_token(refresh_raw)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TTL_DAYS)
    await _db_create_session(user_id, refresh_hash, expires_at)

    # Refresh token → httpOnly cookie (RECIPE_PARAM:TOKEN_TRANSPORT=split)
    response.set_cookie(
        key="refresh_token",
        value=refresh_raw,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=REFRESH_TTL_DAYS * 86400,
        path="/auth/refresh",
    )
    return LoginResponse(access_token=access_token, user_id=user_id)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    refresh_token: Optional[str] = Cookie(default=None),
    _current: dict = Depends(get_current_user),
) -> MessageResponse:
    """Revoke the current session and clear the refresh cookie."""
    if refresh_token:
        token_hash = _hash_token(refresh_token)
        await _db_revoke_session_by_token_hash(token_hash)
    response.delete_cookie(key="refresh_token", path="/auth/refresh")
    return MessageResponse(message="Logged out")


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh(
    response: Response,
    refresh_token: Optional[str] = Cookie(default=None),
) -> TokenRefreshResponse:
    """Rotate access token using refresh token from httpOnly cookie."""
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
        )
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong token type",
        )
    user_id = int(payload["sub"])
    # Verify session still exists and is not revoked
    token_hash = _hash_token(refresh_token)
    # (DB lookup omitted — wire to _db_get_session_by_hash in instantiation)
    access_token = create_access_token(user_id)
    return TokenRefreshResponse(access_token=access_token)


@router.post("/password/reset/request", response_model=MessageResponse)
async def password_reset_request(body: PasswordResetRequestRequest) -> MessageResponse:
    """Send a password reset email. Always returns 200 (no email enumeration)."""
    user = await _db_get_user_by_email(str(body.email))
    if user:
        raw_token = secrets.token_urlsafe(32)
        token_hash = _hash_token(raw_token)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        await _db_create_reset_token(user["id"], token_hash, expires_at)
        await _send_email(
            to=str(body.email),
            subject=f"Reset your {APP_NAME} password",
            body=f"Reset link: https://example.com/reset?token={raw_token}",
            # RECIPE_PARAM: replace with your domain + expiry language
        )
    # Always 200 — never confirm whether an email exists (anti-enumeration)
    return MessageResponse(message="If that email is registered, a reset link has been sent.")


@router.post("/password/reset/confirm", response_model=MessageResponse)
async def password_reset_confirm(body: PasswordResetConfirmRequest) -> MessageResponse:
    """Consume reset token and update password."""
    token_hash = _hash_token(body.token)
    user_id = await _db_consume_reset_token(token_hash)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid, expired, or already used reset token",
        )
    new_hash = hash_password(body.new_password)
    await _db_update_password(user_id, new_hash)
    return MessageResponse(message="Password updated successfully.")


@router.post("/email/verify", response_model=MessageResponse)
async def email_verify(body: EmailVerifyRequest) -> MessageResponse:
    """Consume email verification token and mark account as verified."""
    token_hash = _hash_token(body.token)
    user_id = await _db_consume_verify_token(token_hash)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid, expired, or already used verification token",
        )
    await _db_set_email_verified(user_id)
    return MessageResponse(message="Email verified. You can now log in.")
