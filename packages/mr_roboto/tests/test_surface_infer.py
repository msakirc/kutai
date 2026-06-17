"""Tests for surface inference (5.0b smart gate).

infer_surfaces reads freeform mission text and decides whether the 5.0b
surfaces_lock gate can answer itself (high/medium confidence) instead of
unconditionally pausing the pipeline to ask the founder.
"""
import pytest

from mr_roboto.surface_infer import infer_surfaces, surfaces_label


@pytest.mark.parametrize(
    "text,surfaces,primary,confidence",
    [
        # --- HIGH: explicit single surface --------------------------------
        ("Build an iOS app for tracking habits", ["mobile"], "mobile", "high"),
        ("A web sitesi for booking salons", ["web"], "web", "high"),
        ("Bir masaüstü uygulaması olsun", ["desktop"], "desktop", "high"),
        ("SaaS dashboard for invoicing", ["web"], "web", "high"),
        # --- HIGH: explicit multi-surface ---------------------------------
        ("mobile app plus a web application", ["mobile", "web"], "mobile", "high"),
        # "web app" must NOT add mobile from the bare "app" token
        ("just a web app", ["web"], "web", "high"),
        ("admin panel for the web platform", ["web", "admin"], "web", "high"),
        # --- MEDIUM: bare colloquial leaning ------------------------------
        ("I want an app for habit tracking", ["mobile"], "mobile", "medium"),
        ("bir uygulama yapalım alışveriş için", ["mobile"], "mobile", "medium"),
        # --- LOW: no surface signal ---------------------------------------
        ("A product that helps people save money", [], None, "low"),
        ("", [], None, "low"),
    ],
)
def test_infer_surfaces(text, surfaces, primary, confidence):
    res = infer_surfaces(text)
    assert res["surfaces"] == surfaces, res
    assert res["primary_surface"] == primary, res
    assert res["confidence"] == confidence, res


@pytest.mark.parametrize(
    "tp,expected",
    [
        ("web", {"surfaces": ["web"], "primary_surface": "web"}),
        ("mobile", {"surfaces": ["mobile"], "primary_surface": "mobile"}),
        ("both", {"surfaces": ["mobile", "web"], "primary_surface": "mobile"}),
        ("BOTH", {"surfaces": ["mobile", "web"], "primary_surface": "mobile"}),
        (" web ", {"surfaces": ["web"], "primary_surface": "web"}),
        ("desktop", None),   # target_platform can't express desktop
        ("garbage", None),
        ("", None),
        (None, None),
    ],
)
def test_surfaces_from_target_platform(tp, expected):
    from mr_roboto.surface_infer import surfaces_from_target_platform

    assert surfaces_from_target_platform(tp) == expected


@pytest.mark.parametrize(
    "surfaces,expected",
    [
        (["mobile"], "mobile"),
        (["web"], "web"),
        (["mobile", "web"], "both"),
        (["web", "mobile"], "both"),
        (["web", "admin"], "web"),        # non-mobile only → web
        (["desktop"], "web"),             # desktop folded into web (pre-Stage 2)
        (["mobile", "admin"], "both"),    # mobile + non-mobile → both
        ([], None),
    ],
)
def test_target_platform_from_surfaces(surfaces, expected):
    from mr_roboto.surface_infer import target_platform_from_surfaces

    assert target_platform_from_surfaces(surfaces) == expected


def test_target_platform_roundtrip_with_projection():
    """target_platform → surfaces → target_platform is stable for the 3 enums."""
    from mr_roboto.surface_infer import (
        surfaces_from_target_platform, target_platform_from_surfaces,
    )
    for tp in ("web", "mobile", "both"):
        surfaces = surfaces_from_target_platform(tp)["surfaces"]
        assert target_platform_from_surfaces(surfaces) == tp


@pytest.mark.parametrize(
    "tp,signal,surfaces,primary",
    [
        # desktop/admin layer onto the build base
        ("mobile", ["mobile", "desktop"], ["mobile", "desktop"], "mobile"),
        ("web", ["web", "admin"], ["web", "admin"], "web"),
        ("both", ["mobile", "web", "desktop", "admin"],
         ["mobile", "web", "desktop", "admin"], "mobile"),
        # signal has no design-only extras → just the build base
        ("mobile", ["mobile"], ["mobile"], "mobile"),
        # signal mobile/web are ignored (build axis owns those); only
        # desktop/admin are layered
        ("web", ["mobile", "web"], ["web"], "web"),
        # empty signal
        ("both", [], ["mobile", "web"], "mobile"),
        # canonical ordering regardless of signal order
        ("mobile", ["admin", "desktop", "mobile"],
         ["mobile", "desktop", "admin"], "mobile"),
    ],
)
def test_merge_surfaces(tp, signal, surfaces, primary):
    from mr_roboto.surface_infer import merge_surfaces

    res = merge_surfaces(tp, signal)
    assert res == {"surfaces": surfaces, "primary_surface": primary}


def test_merge_surfaces_none_target_returns_none():
    from mr_roboto.surface_infer import merge_surfaces

    assert merge_surfaces(None, ["desktop"]) is None
    assert merge_surfaces("garbage", ["mobile"]) is None


def test_surfaces_label_roundtrips():
    from mr_roboto.surfaces_persist import parse_surface_choice

    label = surfaces_label(["mobile", "web"])
    assert label == "mobile + web"
    assert parse_surface_choice(label) == ["mobile", "web"]
