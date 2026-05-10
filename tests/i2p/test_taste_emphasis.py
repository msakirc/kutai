"""Tests for Z1 Tier 3 ``verify_taste_emphasis_shape`` mechanical action."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mr_roboto.verify_taste_emphasis_shape import (
    SCHEMA_VERSION,
    verify_payload,
    verify_taste_emphasis_shape,
)
from mr_roboto.derive_token_tag_signature import (
    derive_from_payload,
    derive_token_tag_signature,
)


_FIXTURE_DIR = Path(__file__).parent / "reviewer_regression" / "fixtures" / "v1" / "5_10"


def _load(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


# ---------- verify_payload ----------

def test_schema_version_is_string_one():
    assert SCHEMA_VERSION == "1"


def test_good_taste_fixture_passes():
    res = verify_payload(_load("good_taste_emphasis.json"))
    assert res["ok"] is True
    assert res["primary_content_type"] == "fact_primary"


def test_bad_taste_fixture_other_without_secondary_rejected():
    res = verify_payload(_load("bad_taste_emphasis.json"))
    assert res["ok"] is False
    assert "secondary_emphasis" in res["error"]


def test_missing_required_field_rejected():
    payload = _load("good_taste_emphasis.json")
    del payload["tone_keywords"]
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "tone_keywords" in res["error"]


def test_wrong_schema_version_rejected():
    payload = _load("good_taste_emphasis.json")
    payload["_schema_version"] = "2"
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "_schema_version" in res["error"]


def test_unknown_primary_content_type_rejected():
    payload = _load("good_taste_emphasis.json")
    payload["primary_content_type"] = "garbage_type"
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "primary_content_type" in res["error"]


def test_empty_tone_keywords_rejected():
    payload = _load("good_taste_emphasis.json")
    payload["tone_keywords"] = []
    res = verify_payload(payload)
    assert res["ok"] is False


def test_blank_rationale_rejected():
    payload = _load("good_taste_emphasis.json")
    payload["primary_content_type_rationale"] = "   "
    res = verify_payload(payload)
    assert res["ok"] is False


def test_non_dict_payload_rejected():
    res = verify_payload("not a dict")  # type: ignore[arg-type]
    assert res["ok"] is False


# ---------- verify_taste_emphasis_shape (filesystem) ----------

@pytest.mark.asyncio
async def test_filesystem_good(tmp_path):
    style = tmp_path / ".style"
    style.mkdir()
    (style / "taste_emphasis.json").write_text(
        json.dumps(_load("good_taste_emphasis.json")), encoding="utf-8"
    )
    res = await verify_taste_emphasis_shape(
        mission_id=None,
        path=".style/taste_emphasis.json",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_filesystem_missing(tmp_path):
    res = await verify_taste_emphasis_shape(
        mission_id=None,
        path=".style/taste_emphasis.json",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "not found" in res["error"]


@pytest.mark.asyncio
async def test_filesystem_invalid_json(tmp_path):
    style = tmp_path / ".style"
    style.mkdir()
    (style / "taste_emphasis.json").write_text("{not json", encoding="utf-8")
    res = await verify_taste_emphasis_shape(
        mission_id=None,
        path=".style/taste_emphasis.json",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "invalid JSON" in res["error"]


@pytest.mark.asyncio
async def test_filesystem_absolute_path_rejected(tmp_path):
    res = await verify_taste_emphasis_shape(
        mission_id=None,
        path=str(tmp_path / "foo.json"),
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False


# ---------- derive_token_tag_signature ----------

def test_derive_from_payload_canonical():
    payload = _load("good_taste_emphasis.json")
    assert derive_from_payload(payload) == "fact_primary"


def test_derive_from_payload_unknown_falls_back_to_other():
    assert derive_from_payload({"primary_content_type": "weird"}) == "other"


def test_derive_from_payload_each_canonical_type():
    for slug in (
        "fact_primary",
        "community_primary",
        "discovery_primary",
        "transactional_primary",
        "informational_primary",
        "other",
    ):
        assert derive_from_payload({"primary_content_type": slug}) == slug


def test_derive_from_payload_non_dict():
    assert derive_from_payload(None) == "other"  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_derive_from_payload_inline():
    res = await derive_token_tag_signature(
        mission_id=None,
        payload=_load("good_taste_emphasis.json"),
    )
    assert res["ok"] is True
    assert res["tag_signature"] == "fact_primary"


@pytest.mark.asyncio
async def test_derive_from_filesystem(tmp_path):
    style = tmp_path / ".style"
    style.mkdir()
    (style / "taste_emphasis.json").write_text(
        json.dumps(_load("good_taste_emphasis.json")), encoding="utf-8"
    )
    res = await derive_token_tag_signature(
        mission_id=None,
        path=".style/taste_emphasis.json",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["tag_signature"] == "fact_primary"
