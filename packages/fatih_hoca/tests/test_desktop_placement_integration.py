# packages/fatih_hoca/tests/test_desktop_placement_integration.py
"""Integration test: desktop-state signals flip local<->cloud placement.

S13 (user-presence) fires a FULLSCREEN_VETO (-10.0) for local when
foreground_fullscreen=True, amplified by M4 at load_mode="balanced" (×2.0).
The net negative pressure drives local's final composite below cloud's,
so the selector picks cloud.

When the user is away (user_idle_s ≥ PRESENT_IDLE_S=300s) and
load_mode="full" (M4=0.0 → desktop signals silenced), placement is
governed by cap score + stickiness alone.  Local (cap=55, loaded → ×1.10
stickiness) vs cloud (cap=70, free).  The absolute outcome is
weighting-sensitive, so the robust "away" assertion is the CONTRAST
invariant: gaming must be more cloud-ward than away.
"""
from nerd_herd.types import SystemSnapshot, LocalModelState
from fatih_hoca.selector import select_for_simulation

# Single free-tier cloud provider with cap_score_100=70.
PROVIDERS = {"groq": {"is_free": True, "models": {"groq/big": {"cap_score_100": 70}}}}


def _snap(**kw):
    """Build a SystemSnapshot with sensible defaults, overridable via **kw.

    Defaults represent an absent user on an idle machine in full mode —
    i.e. zero desktop pressure — so a test that doesn't supply desktop
    fields exercises the same baseline as the existing simulator scenarios.
    """
    base = dict(
        vram_available_mb=8000,
        local=LocalModelState(model_name="loaded-local", idle_seconds=120.0),
    )
    base.update(kw)
    return SystemSnapshot(**base)


def test_user_gaming_forces_cloud():
    """Fullscreen + balanced mode → S13 FULLSCREEN_VETO amplified by M4 (2.0×)
    → local composite deeply negative → cloud wins.  This is the core proof
    that desktop signals reach the ranking engine."""
    snap = _snap(user_idle_s=1.0, foreground_fullscreen=True, load_mode="balanced")
    pick = select_for_simulation(
        task_name="coder",
        difficulty=4,
        estimated_output_tokens=500,
        snapshot=snap,
        providers_cfg=PROVIDERS,
    )
    assert pick.pool != "local", (
        f"Expected cloud pick when user is gaming (fullscreen+heavy), "
        f"got pool={pick.pool!r} model={pick.model_name!r}. "
        "S13/M4 desktop signals are not reaching rank_candidates for the local stub."
    )


def test_away_keeps_local():
    # Binding: with no desktop pressure (away + full mode), the loaded local
    # stub must win an easy task on stickiness + cost. This is a real
    # assertion — it fails if the away path ever inverts to cloud.
    snap = _snap(user_idle_s=1e9, load_mode="full")
    pick = select_for_simulation(task_name="coder", difficulty=3,
                                 estimated_output_tokens=500, snapshot=snap,
                                 providers_cfg=PROVIDERS)
    assert pick.pool == "local"
