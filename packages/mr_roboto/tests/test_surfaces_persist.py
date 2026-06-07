"""Tests for surface_choice persistence (5.0b surfaces_lock).

parse_surface_choice turns founder-facing reply-keyboard labels
("mobile + web") into canonical surface tokens; write_surfaces_json
materialises the .charter/surfaces.json the verify_surfaces_shape
post-step asserts on.
"""
import json
import os

import pytest


@pytest.mark.parametrize(
    "label,expected",
    [
        ("mobile only", ["mobile"]),
        ("web only", ["web"]),
        ("mobile + web", ["mobile", "web"]),
        ("mobile + web + desktop", ["mobile", "web", "desktop"]),
        ("mobile + web + admin", ["mobile", "web", "admin"]),
        # dedupe + ignore unknown tokens
        ("mobile + mobile + tablet", ["mobile"]),
        ("", []),
    ],
)
def test_parse_surface_choice(label, expected):
    from mr_roboto.surfaces_persist import parse_surface_choice

    assert parse_surface_choice(label) == expected


@pytest.mark.asyncio
async def test_write_surfaces_json_shape(tmp_path):
    from mr_roboto.surfaces_persist import write_surfaces_json

    data = await write_surfaces_json(
        mission_id=7,
        option_label="mobile + web",
        confirmed_at="2026-06-07T10:00:00+00:00",
        workspace_path=str(tmp_path),
    )

    assert data["_schema_version"] == "1"
    assert data["mission_id"] == 7
    assert data["surfaces"] == ["mobile", "web"]
    assert data["primary_surface"] == "mobile"
    assert data["founder_confirmed_at"] == "2026-06-07T10:00:00+00:00"

    on_disk = os.path.join(str(tmp_path), ".charter", "surfaces.json")
    assert os.path.isfile(on_disk)
    with open(on_disk, encoding="utf-8") as f:
        assert json.load(f) == data


@pytest.mark.asyncio
async def test_write_surfaces_json_passes_verifier(tmp_path):
    """The file we write must satisfy verify_surfaces_shape (the 5.0b check)."""
    from mr_roboto.surfaces_persist import write_surfaces_json
    from mr_roboto.verify_surfaces_shape import verify_surfaces_shape

    await write_surfaces_json(
        mission_id=3,
        option_label="mobile + web + admin",
        workspace_path=str(tmp_path),
    )

    res = await verify_surfaces_shape(
        mission_id=3,
        path=".charter/surfaces.json",
        workspace_path=str(tmp_path),
    )
    assert res["ok"], res["errors"]
    assert res["surfaces"] == ["mobile", "web", "admin"]
    assert res["primary"] == "mobile"


@pytest.mark.asyncio
async def test_write_surfaces_json_rejects_empty(tmp_path):
    from mr_roboto.surfaces_persist import write_surfaces_json

    with pytest.raises(ValueError):
        await write_surfaces_json(
            mission_id=1,
            option_label="tablet only",  # no valid tokens
            workspace_path=str(tmp_path),
        )
