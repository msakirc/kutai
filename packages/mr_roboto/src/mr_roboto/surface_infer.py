"""Infer target surfaces from freeform mission text (5.0b smart gate).

The 5.0b ``surfaces_lock`` step used to *unconditionally* ask the founder
"which platforms does this product target?" — even when the mission
description already said "build an app" or "a website". That is the dumb
case: pausing the whole pipeline to ask what the founder already stated.

This module infers the surfaces from the mission text deterministically
(no LLM, Turkish + English) and reports a confidence so the clarify gate
can decide whether to ask at all:

- ``high``   — explicit signal ("ios app", "web sitesi", "mobile + web").
              Proceed silently.
- ``medium`` — a bare/colloquial signal ("app", "uygulama"). Proceed with
              the best guess but offer a non-blocking "tap to change".
- ``low``    — no surface signal at all. Genuinely ambiguous → ask the
              founder (the original blocking keyboard).

Kept LLM-free and dependency-free so it is unit-testable and callable
straight from the mechanical clarify executor.
"""
from __future__ import annotations

from typing import Any

# Canonical ordering — primary_surface defaults to the first present here,
# and multi-surface lists are emitted in this order. Mirrors
# verify_surfaces_shape.VALID_SURFACES.
SURFACE_ORDER = ("mobile", "web", "desktop", "admin")

# "Strong" phrases unambiguously name a surface → high confidence.
# "Weak" phrases are colloquial leanings ("app" usually means a mobile app
# in startup-speak, but could be a web app) → medium confidence, offer an
# easy correction instead of blocking.
_STRONG: dict[str, tuple[str, ...]] = {
    "mobile": (
        "mobile app", "mobile application", "ios app", "android app",
        "ios", "android", "app store", "play store", "react native",
        "flutter", "swiftui", "mobil uygulama", "ios uygulaması",
        "android uygulaması", "mobil app",
    ),
    "web": (
        "web app", "web application", "webapp", "website", "web site",
        "web sitesi", "internet sitesi", "saas", "landing page", "dashboard",
        "web uygulaması", "web platform", "web platformu", "browser-based",
        "tarayıcı tabanlı",
    ),
    "desktop": (
        "desktop app", "desktop application", "windows app", "macos app",
        "mac app", "electron app", "masaüstü uygulaması", "masaüstü app",
        "masaüstü programı",
    ),
    "admin": (
        "admin panel", "admin dashboard", "admin paneli", "back office",
        "yönetim paneli", "yönetici paneli",
    ),
}

_WEAK: dict[str, tuple[str, ...]] = {
    "mobile": ("app", "uygulama", "mobile", "mobil", "phone app"),
    "web": ("website", "web", "site", "portal"),
    "desktop": ("desktop", "electron", "masaüstü"),
    "admin": ("admin",),
}


def _hits(text: str, phrases: tuple[str, ...]) -> bool:
    return any(p in text for p in phrases)


def infer_surfaces(text: str) -> dict[str, Any]:
    """Infer surfaces + confidence from freeform mission text.

    Returns ``{"surfaces": [...], "primary_surface": str|None,
    "confidence": "high"|"medium"|"low", "signal": "strong"|"weak"|"none"}``.

    Rules:
    - Any strong signal → ``high`` confidence; surfaces = every family with a
      strong hit (so "mobile and web app" yields both). Weak hits are ignored
      once a strong hit exists, so "web app" (which contains the weak token
      "app") does not spuriously add "mobile".
    - No strong but ≥1 weak signal → ``medium``; surfaces = weak-hit families.
    - No signal at all → ``low``; surfaces = [].
    """
    t = (text or "").lower()

    strong = [s for s in SURFACE_ORDER if _hits(t, _STRONG[s])]
    if strong:
        return {
            "surfaces": strong,
            "primary_surface": strong[0],
            "confidence": "high",
            "signal": "strong",
        }

    weak = [s for s in SURFACE_ORDER if _hits(t, _WEAK[s])]
    if weak:
        return {
            "surfaces": weak,
            "primary_surface": weak[0],
            "confidence": "medium",
            "signal": "weak",
        }

    return {
        "surfaces": [],
        "primary_surface": None,
        "confidence": "low",
        "signal": "none",
    }


def surfaces_label(surfaces: list[str]) -> str:
    """Render surface tokens as a ``write_surfaces_json`` option label.

    ``["mobile", "web"]`` -> ``"mobile + web"`` (round-trips through
    ``parse_surface_choice``).
    """
    return " + ".join(surfaces)
