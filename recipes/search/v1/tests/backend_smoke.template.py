"""Search recipe — backend smoke tests.

Runs POST-instantiation against the instantiated recipe in the mission
workspace. Recipe sources are scaffolds (.template suffix).
"""
from __future__ import annotations

import pytest


async def test_search_returns_matching_results():
    """GET /search?q=hello returns results containing 'hello' in title or body."""
    # T6 WILL FILL — seed documents, assert results non-empty and relevant
    pass


async def test_search_short_query_returns_400():
    """Query shorter than MIN_QUERY_LEN returns HTTP 400."""
    # T6 WILL FILL — GET /search?q=a → assert status_code == 400
    pass


async def test_search_no_results_returns_empty_list():
    """Query matching no documents returns empty results list (not 404)."""
    # T6 WILL FILL — GET /search?q=xyzzy_nonexistent → assert results == []
    pass


async def test_search_limit_respected():
    """limit param caps result count even when more matches exist."""
    # T6 WILL FILL — seed 20 matching docs, limit=5, assert len(results) <= 5
    pass


async def test_search_response_includes_backend_field():
    """Response always includes 'backend' field."""
    # T6 WILL FILL — assert "backend" in response JSON
    pass
