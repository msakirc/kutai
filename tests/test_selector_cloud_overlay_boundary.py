"""Selector must read cloud daily_exhausted / rpm_cooldown from the in-process
KDV, NOT from the (stale sidecar) NerdHerdClient snapshot.

Serialization-boundary regression — production triage 2026-06-15: task 1.4a
analyst re-picked gemini/gemini-flash-latest for hours and surfaced
"All models failed: Daily limit exhausted" on every ~15-min re-pend. KDV had
correctly marked the model daily-exhausted (pre_call refused every call) but
that flag never crossed into the selector's snapshot: the selector reads the
sidecar NerdHerdClient cache (client._overlay_local mirrors swap/queue/image
only), and configure_in_flight_push writes the in-process singleton, not the
client cache. So the model ranked #1 un-penalized, got admitted, and pre_call
refused it post-admission — forever.

The fix overlays snapshot.cloud from the live KDV (the SAME adapter Beckman
uses) right after self._nerd_herd.snapshot(), mirroring the existing in_flight
overlay. These tests exercise that seam — they inject exhaustion via the KDV
path, NOT via nh.snapshot(), which is exactly the boundary that was broken.
See feedback_test_serialization_boundary.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fatih_hoca.selector import Selector
from fatih_hoca.registry import ModelInfo, ModelRegistry
from nerd_herd.types import SystemSnapshot, CloudProviderState, CloudModelState


def _cloud_model(name: str, provider: str, litellm: str) -> ModelInfo:
    return ModelInfo(
        name=name,
        location="cloud",
        provider=provider,
        litellm_name=litellm,
        capabilities={
            "reasoning": 7.0,
            "code_generation": 7.0,
            "tool_use": 6.0,
            "instruction_adherence": 6.0,
        },
        tokens_per_second=20.0,
        context_length=32768,
        supports_function_calling=True,
        tier="free",
    )


class _StubKDV:
    """Minimal stand-in: the fix only reads `_providers.keys()` and hands the
    kdv to the (monkeypatched) build_cloud_provider_state."""

    def __init__(self, providers: dict) -> None:
        self._providers = providers


_FLASH = "gemini/gemini-flash-latest"
_ALT = "cerebras/zai-glm-4.7"


def _make_selector(monkeypatch, *, overlay: dict) -> Selector:
    """Build a selector over {flash, alt}. The nh snapshot is STALE (no cloud
    state at all — the pre-fix sidecar reality). Exhaustion is delivered only
    through the KDV seam via `overlay` (provider -> CloudProviderState | None).
    """
    flash = _cloud_model("gemini-flash-latest", "gemini", _FLASH)
    alt = _cloud_model("zai-glm-4.7", "cerebras", _ALT)
    reg = ModelRegistry()
    reg.register(flash)
    reg.register(alt)

    snap = SystemSnapshot(vram_available_mb=8192)  # stale sidecar: cloud empty
    nh = MagicMock()
    nh.snapshot.return_value = snap
    nh.can_swap.return_value = True
    nh.recent_swap_count.return_value = 0

    monkeypatch.setattr(
        "src.core.router.get_kdv",
        lambda: _StubKDV({"gemini": {_FLASH}, "cerebras": {_ALT}}),
    )
    monkeypatch.setattr(
        "kuleden_donen_var.nerd_herd_adapter.build_cloud_provider_state",
        lambda kdv, prov: overlay.get(prov),
    )
    return Selector(registry=reg, nerd_herd=nh)


def test_selector_excludes_daily_exhausted_sourced_from_kdv(monkeypatch):
    """KDV marks flash daily-exhausted (snapshot does NOT). Selector must
    rebuild cloud from KDV, exclude flash, and pick the live alternative."""
    overlay = {
        "gemini": CloudProviderState(
            provider="gemini",
            models={
                _FLASH: CloudModelState(model_id=_FLASH, daily_exhausted=True),
            },
        ),
        "cerebras": CloudProviderState(provider="cerebras", models={}),
    }
    sel = _make_selector(monkeypatch, overlay=overlay)

    diag: dict = {}
    pick = sel.select(task="analyst", difficulty=5, diag_out=diag)

    assert pick is not None, "expected a fallback pick, got pool-empty"
    assert pick.model.litellm_name == _ALT, (
        f"selector picked {pick.model.litellm_name!r}; flash should be excluded"
    )
    # The exclusion came from the eligibility gate, fed by the KDV overlay.
    assert any(
        str(r).startswith("daily_exhausted") for r in diag.get("filter_reasons", {})
    ), f"daily_exhausted filter never fired; reasons={diag.get('filter_reasons')}"


def test_control_without_kdv_overlay_flash_stays_eligible(monkeypatch):
    """Boundary control: when the KDV seam yields no data (build_cloud -> None),
    the selector keeps the stale snapshot — exactly the pre-fix world — and
    flash is NOT excluded. Proves the snapshot path alone cannot see daily
    exhaustion, so the KDV overlay is load-bearing."""
    sel = _make_selector(monkeypatch, overlay={})  # build_cloud -> None per prov

    diag: dict = {}
    pick = sel.select(task="analyst", difficulty=5, diag_out=diag)

    assert pick is not None
    assert diag.get("eligible_count") == 2, (
        "both models should be eligible without a KDV exhaustion overlay; "
        f"eligible_count={diag.get('eligible_count')}"
    )
    assert not any(
        str(r).startswith("daily_exhausted") for r in diag.get("filter_reasons", {})
    )
