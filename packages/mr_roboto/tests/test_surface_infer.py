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


def test_surfaces_label_roundtrips():
    from mr_roboto.surfaces_persist import parse_surface_choice

    label = surfaces_label(["mobile", "web"])
    assert label == "mobile + web"
    assert parse_surface_choice(label) == ["mobile", "web"]
