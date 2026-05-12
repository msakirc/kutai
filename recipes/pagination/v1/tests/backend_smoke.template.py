"""Pagination recipe — backend smoke tests.

Runs POST-instantiation against the instantiated recipe in the mission
workspace. Recipe sources are scaffolds (.template suffix).
"""
from __future__ import annotations

import pytest


async def test_cursor_first_page_returns_items():
    """GET /items with no cursor returns first page and next_cursor when has_more."""
    # T6 WILL FILL — build in-process FastAPI app, seed test rows, assert response shape
    pass


async def test_cursor_second_page_advances():
    """Passing next_cursor from page 1 returns the next batch without overlap."""
    # T6 WILL FILL — seed N > PAGE_SIZE_DEFAULT rows, walk two cursor pages, no dup ids
    pass


async def test_cursor_last_page_has_no_next():
    """Final page returns next_cursor=null and has_more=false."""
    # T6 WILL FILL — seed exactly PAGE_SIZE_DEFAULT rows, assert no next cursor
    pass


async def test_offset_page_returns_correct_slice():
    """GET /items/offset?page=2&limit=5 returns second slice of 5 items."""
    # T6 WILL FILL — seed 12 rows, page=2 limit=5 → items[5:10], total=12, pages=3
    pass


async def test_invalid_cursor_returns_400():
    """Malformed cursor token returns HTTP 400."""
    # T6 WILL FILL — GET /items?cursor=NOTBASE64 → assert status_code == 400
    pass


async def test_limit_above_max_clamped_or_rejected():
    """Limit exceeding MAX_PAGE_SIZE is rejected with 422."""
    # T6 WILL FILL — GET /items?limit=999 → assert status_code == 422
    pass
