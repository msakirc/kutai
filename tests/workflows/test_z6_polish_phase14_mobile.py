"""Z6 polish (P6) — Phase 14 mobile-track audit.

Verifies that every Phase 14 step which touches a mobile app-store
vendor (Apple App Store Connect / Google Play Console) carries the
``real_tool_kind`` tag so the Z6 admission gate can short-circuit when
credentials are absent. Pure local-build steps (xcodebuild,
gradle assembleRelease) must NOT be tagged — they don't need real tools.

Today the only mobile-vendor step in Phase 14 is 14.8
(app_store_submission). The other 14.x steps are planning, launch
execution against a web target, marketing, announcements, monitoring
and retrospective — none touch Apple/Google directly.
"""
from __future__ import annotations

import json
from pathlib import Path


WF_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "workflows" / "i2p" / "i2p_v3.json"
)

# Keywords whose presence in a step's name + instruction strongly
# suggests an Apple/Google app-store vendor interaction.
_MOBILE_VENDOR_HINTS = (
    "app store",
    "app store connect",
    "appstore",
    "play console",
    "play store",
    "google play",
    "testflight",
    "app store submission",
)


def _load_workflow():
    with open(WF_PATH, encoding="utf-8") as f:
        return json.load(f)


def _phase14_steps():
    wf = _load_workflow()
    return [s for s in wf["steps"] if str(s.get("id", "")).startswith("14.")]


def _looks_mobile_vendor(step: dict) -> bool:
    # Explicit opt-out: step declares it is a local step despite app-store
    # keywords in its instruction text (e.g. screenshot capture that runs
    # locally via Playwright / adb, not via an Apple/Google API).
    if step.get("vendor_interaction") is False:
        return False
    blob = " ".join((
        str(step.get("name", "")),
        str(step.get("instruction", "")),
    )).lower()
    return any(hint in blob for hint in _MOBILE_VENDOR_HINTS)


def test_phase14_mobile_vendor_steps_tagged():
    """Every Phase-14 step that names an app-store vendor must declare
    ``real_tool_kind``. Otherwise admission can't short-circuit and the
    agent will hallucinate a submission outcome."""
    mobile_steps = [s for s in _phase14_steps() if _looks_mobile_vendor(s)]
    assert mobile_steps, (
        "expected ≥1 Phase-14 step referencing Apple/Google app-store"
    )
    for s in mobile_steps:
        rtk = s.get("real_tool_kind")
        assert rtk, (
            f"Phase-14 mobile-vendor step {s['id']} ({s.get('name')}) "
            f"missing real_tool_kind"
        )
        assert s.get("needs_real_tools") is True, (
            f"Phase-14 mobile-vendor step {s['id']} should have "
            f"needs_real_tools=True"
        )
        # rtk should mention apple/google
        rtk_lower = rtk.lower()
        assert (
            "apple" in rtk_lower
            or "appstore" in rtk_lower
            or "google" in rtk_lower
            or "play" in rtk_lower
        ), f"step {s['id']} real_tool_kind={rtk!r} doesn't mention apple/google"


def test_phase14_pure_local_build_steps_not_tagged():
    """Steps that only run local toolchains (xcodebuild, gradle) without
    contacting a vendor must NOT be tagged real_tool_kind — tagging them
    would block missions that build mobile binaries for sideloading."""
    LOCAL_BUILD_NAMES = {
        # add any pure-local mobile build steps here if they exist; the
        # current Phase 14 has none, so the test asserts the empty
        # invariant.
    }
    for s in _phase14_steps():
        if s.get("name") in LOCAL_BUILD_NAMES:
            assert not s.get("needs_real_tools"), (
                f"pure local-build step {s['id']} must not be tagged"
            )


def test_phase14_step_14_8_canonical_shape():
    """Canonical regression — keeps the audit's expected tagging stable."""
    s = next((x for x in _phase14_steps() if x["id"] == "14.8"), None)
    assert s is not None
    assert s.get("needs_real_tools") is True
    assert s.get("real_tool_kind") == "apple_appstore|google_play"


def test_phase14_steps_dependent_on_app_store_submission_inherit():
    """Any Phase-14 step whose ``depends_on`` chain includes the mobile
    submission step should either also be tagged needs_real_tools or be
    explicitly local (no vendor interaction in its instruction). Soft
    check — only emits an assertion error when a downstream step
    advertises submission outcomes without a tag."""
    steps_by_id = {s["id"]: s for s in _phase14_steps()}
    submission_ids = {
        s["id"] for s in _phase14_steps() if _looks_mobile_vendor(s)
    }
    for s in _phase14_steps():
        deps = set(s.get("depends_on", []) or [])
        if not (deps & submission_ids):
            continue
        if _looks_mobile_vendor(s):
            assert s.get("real_tool_kind"), (
                f"downstream mobile-vendor step {s['id']} must be tagged"
            )
