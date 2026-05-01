"""Tests for src.infra.db._retry_on_locked."""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from src.infra import db as db_mod


@pytest.mark.asyncio
async def test_retry_succeeds_after_one_lock_error():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    out = await db_mod._retry_on_locked(factory, label="test")
    assert out == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_retry_exhausts_then_raises():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):
        await db_mod._retry_on_locked(factory, label="test")
    assert calls["n"] == 4  # max_attempts


@pytest.mark.asyncio
async def test_retry_does_not_swallow_other_operational_errors():
    """Only 'database is locked' / 'database is busy' should retry —
    schema errors etc. must surface immediately."""
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        raise sqlite3.OperationalError("no such column: foo")

    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        await db_mod._retry_on_locked(factory, label="test")
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_handles_database_is_busy():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        if calls["n"] < 2:
            raise sqlite3.OperationalError("database is busy")
        return "ok"

    out = await db_mod._retry_on_locked(factory, label="test")
    assert out == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_retry_uses_fresh_coro_each_attempt():
    """Factory must be called fresh each time — re-awaiting a consumed
    coroutine raises RuntimeError."""
    n = 0

    async def factory():
        nonlocal n
        n += 1
        if n < 3:
            raise sqlite3.OperationalError("database is locked")
        return n

    out = await db_mod._retry_on_locked(factory, label="test")
    assert out == 3
