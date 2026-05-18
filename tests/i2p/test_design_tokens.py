"""Tests for Z1 Tier 3 ``verify_design_tokens_shape`` mechanical action."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mr_roboto.verify_design_tokens_shape import (
    SCHEMA_VERSION,
    verify_payload,
    verify_design_tokens_shape,
)


_FIXTURE_DIR = Path(__file__).parent / "reviewer_regression" / "fixtures" / "v1" / "5_10"


def _load(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


# ---------- verify_payload ----------

def test_schema_version_is_string_one():
    assert SCHEMA_VERSION == "1"


def test_paraflow_shape_fixture_passes():
    res = verify_payload(_load("good_design_tokens.json"))
    assert res["ok"] is True
    assert "light" in res["variants"]
    assert "dark" in res["variants"]
    assert res["tag_signature"] == "fact_primary"


def test_bad_design_tokens_fixture_rejected():
    # Bad fixture is missing dark variant + has placeholder hex.
    res = verify_payload(_load("bad_design_tokens.json"))
    assert res["ok"] is False


def test_missing_dark_variant_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["variants"]["dark"]
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "dark" in res["error"]


def test_missing_light_variant_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["variants"]["light"]
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "light" in res["error"]


def test_missing_color_block_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["variants"]["light"]["color"]
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "color" in res["error"]


def test_missing_typography_block_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["variants"]["light"]["typography"]
    res = verify_payload(payload)
    assert res["ok"] is False


def test_missing_spacing_block_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["variants"]["light"]["spacing"]
    res = verify_payload(payload)
    assert res["ok"] is False


def test_missing_borders_block_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["variants"]["light"]["borders"]
    res = verify_payload(payload)
    assert res["ok"] is False


def test_missing_shadows_block_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["variants"]["light"]["shadows"]
    res = verify_payload(payload)
    assert res["ok"] is False


def test_missing_border_radius_block_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["variants"]["light"]["border_radius"]
    res = verify_payload(payload)
    assert res["ok"] is False


def test_placeholder_hex_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["variants"]["light"]["color"]["primary"]["base"] = "#XXXXXX"
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "placeholder" in res["error"]


def test_tbd_token_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["variants"]["light"]["color"]["accent"]["amber"] = "TBD"
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "TBD" in res["error"]


def test_malformed_hex_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["variants"]["light"]["color"]["accent"]["amber"] = "#GGHHII"
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "malformed hex" in res["error"]


def test_short_hex_3_digit_accepted():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["variants"]["light"]["color"]["accent"]["amber"] = "#FFF"
    res = verify_payload(payload)
    assert res["ok"] is True


def test_8_digit_hex_with_alpha_accepted():
    # Already present in fixture (#FFFFFFCC) — assert direct pass.
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["variants"]["light"]["color"]["accent"]["amber"] = "#12345678"
    res = verify_payload(payload)
    assert res["ok"] is True


def test_empty_font_family_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["variants"]["light"]["typography"]["font_family_base"] = ""
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "font_family_base" in res["error"]


def test_empty_typography_scale_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["variants"]["light"]["typography"]["scale"] = {}
    res = verify_payload(payload)
    assert res["ok"] is False


def test_wrong_schema_version_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["_schema_version"] = "2"
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "_schema_version" in res["error"]


def test_missing_tag_signature_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["tag_signature"]
    res = verify_payload(payload)
    assert res["ok"] is False
    assert "tag_signature" in res["error"]


def test_blank_tag_signature_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["tag_signature"] = "   "
    res = verify_payload(payload)
    assert res["ok"] is False


def test_missing_mission_id_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    del payload["mission_id"]
    res = verify_payload(payload)
    assert res["ok"] is False


def test_optional_density_variant_passes():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["variants"]["compact"] = copy.deepcopy(payload["variants"]["light"])
    res = verify_payload(payload)
    assert res["ok"] is True
    assert "compact" in res["variants"]


def test_optional_density_variant_with_bad_block_rejected():
    payload = copy.deepcopy(_load("good_design_tokens.json"))
    payload["variants"]["compact"] = copy.deepcopy(payload["variants"]["light"])
    del payload["variants"]["compact"]["color"]
    res = verify_payload(payload)
    assert res["ok"] is False


def test_non_dict_payload_rejected():
    res = verify_payload([1, 2, 3])  # type: ignore[arg-type]
    assert res["ok"] is False


# ---------- verify_design_tokens_shape (filesystem) ----------

@pytest.mark.asyncio
async def test_filesystem_good(tmp_path):
    style = tmp_path / ".style"
    style.mkdir()
    (style / "design_tokens.json").write_text(
        json.dumps(_load("good_design_tokens.json")), encoding="utf-8"
    )
    res = await verify_design_tokens_shape(
        mission_id=None,
        path=".style/design_tokens.json",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_filesystem_bad(tmp_path):
    style = tmp_path / ".style"
    style.mkdir()
    (style / "design_tokens.json").write_text(
        json.dumps(_load("bad_design_tokens.json")), encoding="utf-8"
    )
    res = await verify_design_tokens_shape(
        mission_id=None,
        path=".style/design_tokens.json",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False


@pytest.mark.asyncio
async def test_filesystem_missing(tmp_path):
    res = await verify_design_tokens_shape(
        mission_id=None,
        path=".style/design_tokens.json",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False


@pytest.mark.asyncio
async def test_filesystem_invalid_json(tmp_path):
    style = tmp_path / ".style"
    style.mkdir()
    (style / "design_tokens.json").write_text("garbage", encoding="utf-8")
    res = await verify_design_tokens_shape(
        mission_id=None,
        path=".style/design_tokens.json",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False


@pytest.mark.asyncio
async def test_mr_roboto_dispatch_taste(tmp_path):
    """Verify mechanical dispatcher routes to the new actions correctly."""
    import mr_roboto

    style = tmp_path / ".style"
    style.mkdir()
    (style / "taste_emphasis.json").write_text(
        json.dumps(_load("good_taste_emphasis.json")), encoding="utf-8"
    )
    task = {
        "id": 1,
        "mission_id": None,
        "payload": {
            "action": "verify_taste_emphasis_shape",
            "path": ".style/taste_emphasis.json",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"


@pytest.mark.asyncio
async def test_mr_roboto_dispatch_design_tokens(tmp_path):
    import mr_roboto

    style = tmp_path / ".style"
    style.mkdir()
    (style / "design_tokens.json").write_text(
        json.dumps(_load("good_design_tokens.json")), encoding="utf-8"
    )
    task = {
        "id": 1,
        "mission_id": None,
        "payload": {
            "action": "verify_design_tokens_shape",
            "path": ".style/design_tokens.json",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"


@pytest.mark.asyncio
async def test_mr_roboto_dispatch_design_tokens_failure(tmp_path):
    import mr_roboto

    style = tmp_path / ".style"
    style.mkdir()
    (style / "design_tokens.json").write_text(
        json.dumps(_load("bad_design_tokens.json")), encoding="utf-8"
    )
    task = {
        "id": 1,
        "mission_id": None,
        "payload": {
            "action": "verify_design_tokens_shape",
            "path": ".style/design_tokens.json",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "failed"


@pytest.mark.asyncio
async def test_mr_roboto_dispatch_derive_tag(tmp_path):
    import mr_roboto

    task = {
        "id": 1,
        "mission_id": None,
        "payload": {
            "action": "derive_token_tag_signature",
            "taste_payload": _load("good_taste_emphasis.json"),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["tag_signature"] == "fact_primary"
