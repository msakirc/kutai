"""Smoke tests for the mobile_push/v1 backend send stub.

RECIPE_PARAM markers:
  # RECIPE_PARAM:EXPO_PUSH_API=https://exp.host/--/api/v2/push/send

These tests exercise the FastAPI router with httpx mocked — no network.
"""
from __future__ import annotations

import pytest

# The backend module is written next to this file as `backend.template.py`
# pre-instantiation; after `instantiate_recipe` it becomes `backend.py`.
# In a real mission the import path is the instantiated module name.

EXPO_PUSH_API = "<<EXPO_PUSH_API>>"


def test_is_expo_push_token_accepts_valid_forms():
    from backend import _is_expo_push_token
    assert _is_expo_push_token("ExponentPushToken[abc123]")
    assert _is_expo_push_token("ExpoPushToken[abc123]")


def test_is_expo_push_token_rejects_raw_fcm_token():
    from backend import _is_expo_push_token
    # A raw FCM token is NOT an Expo token — must be rejected.
    assert not _is_expo_push_token("fcm-raw-token-value")
    assert not _is_expo_push_token("")


@pytest.mark.asyncio
async def test_register_push_token_rejects_non_expo_token():
    from fastapi import HTTPException
    from backend import register_push_token, RegisterTokenRequest

    with pytest.raises(HTTPException) as exc:
        await register_push_token(
            RegisterTokenRequest(expo_push_token="bad-token", platform="ios")
        )
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_register_push_token_stores_valid_token():
    from backend import register_push_token, RegisterTokenRequest

    res = await register_push_token(
        RegisterTokenRequest(
            expo_push_token="ExponentPushToken[smoke]",
            platform="android",
        )
    )
    assert res["ok"] is True
    assert res["stored"] >= 1
