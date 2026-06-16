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


# target_platform (3.6, the canonical build signal) → surfaces projection.
# This is the single-source-of-truth map: surfaces is DERIVED from the same
# value the tech stack (4.2) and scaffold branch (7.5/7.5m) already honor, so
# the design lane can never contradict the build. See
# docs/superpowers/specs/2026-06-17-surface-single-source-design.md.
_TP_TO_SURFACES: dict[str, tuple[list[str], str]] = {
    "web": (["web"], "web"),
    "mobile": (["mobile"], "mobile"),
    "both": (["mobile", "web"], "mobile"),
}


def surfaces_from_target_platform(target_platform: str | None) -> dict[str, Any] | None:
    """Project ``platform_requirements.target_platform`` onto a surface set.

    Returns ``{"surfaces": [...], "primary_surface": str}`` for a recognized
    target_platform (``web`` | ``mobile`` | ``both``), else ``None`` (caller
    falls back to text inference). ``desktop``/``admin`` are intentionally NOT
    produced here — target_platform cannot express them (Stage 2/3 of the spec).
    """
    key = (target_platform or "").strip().lower()
    pair = _TP_TO_SURFACES.get(key)
    if pair is None:
        return None
    surfaces, primary = pair
    return {"surfaces": list(surfaces), "primary_surface": primary}


def target_platform_from_surfaces(surfaces: list[str]) -> str | None:
    """Project a surface set onto ``target_platform`` (the 3.6 build enum).

    Inverse of ``surfaces_from_target_platform``, for the Stage-3 intake
    signal that grounds 3.6 in the founder's own words:
    - mobile AND web present → ``both``
    - mobile only            → ``mobile``
    - any non-mobile surface (web/desktop/admin) without mobile → ``web``
      (matches 3.6's rule: "responsive web only counts as 'web'"; desktop is
      folded into 'web' until Stage 2 makes it first-class)
    - empty → ``None`` (no signal; 3.6 derives from the PRD itself)
    """
    s = set(surfaces or [])
    if not s:
        return None
    has_mobile = "mobile" in s
    has_nonmobile = bool(s - {"mobile"})
    if has_mobile and has_nonmobile:
        return "both"
    if has_mobile:
        return "mobile"
    return "web"


# Surfaces that target_platform (web/mobile/both) cannot express. They ride on
# the deterministic surface_signal (3.5z) and only affect the design lane
# (screens at 5.0c/5.0d) — NOT the build rail — until Stage 2's build track.
_DESIGN_ONLY_SURFACES = ("desktop", "admin")


def merge_surfaces(
    target_platform: str | None,
    signal_surfaces: list[str] | None,
) -> dict[str, Any] | None:
    """Reconstruct the full surface set for the design lane (Stage 2, safe half).

    web/mobile come from ``target_platform`` (the canonical build signal, so the
    design lane stays consistent with the tech stack at 4.2). desktop/admin —
    which target_platform cannot express — are layered on from the deterministic
    ``surface_signal``. Returns ``{"surfaces": [...], "primary_surface": str}``
    in canonical order, or ``None`` when target_platform is absent (caller falls
    back to text inference). ``primary_surface`` stays a build surface (never
    desktop/admin) so it aligns with what the stack was built for.
    """
    base = surfaces_from_target_platform(target_platform)
    if base is None:
        return None
    base_surfaces = base["surfaces"]
    extras = [
        s for s in (signal_surfaces or [])
        if s in _DESIGN_ONLY_SURFACES and s not in base_surfaces
    ]
    combined = set(base_surfaces) | set(extras)
    ordered = [s for s in SURFACE_ORDER if s in combined]
    return {"surfaces": ordered, "primary_surface": base["primary_surface"]}


def surfaces_label(surfaces: list[str]) -> str:
    """Render surface tokens as a ``write_surfaces_json`` option label.

    ``["mobile", "web"]`` -> ``"mobile + web"`` (round-trips through
    ``parse_surface_choice``).
    """
    return " + ".join(surfaces)
