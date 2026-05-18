"""Expo push notifications — FastAPI backend send stub.

RECIPE_PARAM markers:
  # RECIPE_PARAM:EXPO_PUSH_API=https://exp.host/--/api/v2/push/send
  # RECIPE_PARAM:PUSH_TOKEN_ENDPOINT=/push/register
  # RECIPE_PARAM:APP_NAME=MyApp

The template engine replaces RECIPE_PARAM defaults before writing the file
into the mission workspace. Leave the comment markers intact — they must
survive an ast.parse() call so the syntax checker passes before substitution.

This is a *stub*: token storage is an in-memory dict. Wire a real table
(`push_tokens(user_id, expo_push_token, platform, updated_at)`) in
production — see lessons.md.
"""
from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Expo's hosted push endpoint. The Expo push service fans the message out
# to APNs (iOS) and FCM (Android) — you never talk to APNs/FCM directly.
EXPO_PUSH_API = "<<EXPO_PUSH_API>>"
APP_NAME = "<<APP_NAME>>"

router = APIRouter(tags=["push"])

# STUB store — replace with a real DB table in production.
_TOKEN_STORE: dict[str, dict[str, str]] = {}


class RegisterTokenRequest(BaseModel):
    expo_push_token: str = Field(..., min_length=1)
    platform: str = Field(default="unknown")
    user_id: str | None = None


class SendPushRequest(BaseModel):
    expo_push_token: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    body: str = Field(default="")
    data: dict[str, Any] = Field(default_factory=dict)


# RECIPE_PARAM:PUSH_TOKEN_ENDPOINT — keep the route literal in sync with
# the client shim's PUSH_TOKEN_ENDPOINT marker.
@router.post("/push/register")
async def register_push_token(req: RegisterTokenRequest) -> dict[str, Any]:
    """Persist a device's Expo push token so the server can target it."""
    # Expo push tokens are always of the form ExponentPushToken[...] or
    # ExpoPushToken[...]. Reject anything else early — a raw FCM/APNs token
    # sent to the Expo API silently no-ops.
    if not _is_expo_push_token(req.expo_push_token):
        raise HTTPException(status_code=422, detail="not an Expo push token")
    _TOKEN_STORE[req.expo_push_token] = {
        "platform": req.platform,
        "user_id": req.user_id or "",
    }
    return {"ok": True, "stored": len(_TOKEN_STORE)}


async def send_push(req: SendPushRequest) -> dict[str, Any]:
    """Send one push message through the Expo push service.

    Returns the parsed Expo receipt envelope. Raises HTTPException on a
    transport error or a non-ok Expo status.
    """
    if not _is_expo_push_token(req.expo_push_token):
        raise HTTPException(status_code=422, detail="not an Expo push token")

    message = {
        "to": req.expo_push_token,
        "title": req.title,
        "body": req.body,
        "data": req.data,
        "sound": "default",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                EXPO_PUSH_API,
                json=message,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
    except httpx.HTTPError as exc:  # transport-level failure
        raise HTTPException(status_code=502, detail=f"expo push transport: {exc}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"expo push API returned {resp.status_code}",
        )

    payload = resp.json()
    # Expo wraps tickets under "data". A ticket with status "error" means
    # the token is dead (DeviceNotRegistered) — caller should evict it.
    ticket = (payload.get("data") or {})
    if isinstance(ticket, dict) and ticket.get("status") == "error":
        return {"ok": False, "ticket": ticket}
    return {"ok": True, "ticket": ticket}


@router.post("/push/send")
async def send_push_route(req: SendPushRequest) -> dict[str, Any]:
    return await send_push(req)


def _is_expo_push_token(token: str) -> bool:
    """True when ``token`` looks like an Expo push token."""
    return token.startswith(("ExponentPushToken[", "ExpoPushToken["))
