import time

import pytest

from kuleden_donen_var.kdv import KuledenDonenVar
from kuleden_donen_var.config import KuledenConfig


def _kdv() -> KuledenDonenVar:
    return KuledenDonenVar(KuledenConfig())


def test_no_data_warning_fires_after_threshold():
    kdv = _kdv()
    kdv.register(model_id="groq/llama-3.3-70b", provider="groq", rpm=30, tpm=100000)
    kdv.mark_provider_enabled("groq", at_unix=time.time() - (25 * 3600))
    warnings = kdv.no_data_warnings(min_age_hours=24)
    assert any(w["provider"] == "groq" for w in warnings)


def test_no_data_warning_skips_recently_enabled():
    kdv = _kdv()
    kdv.register(model_id="groq/llama-3.3-70b", provider="groq", rpm=30, tpm=100000)
    kdv.mark_provider_enabled("groq", at_unix=time.time() - 3600)
    warnings = kdv.no_data_warnings(min_age_hours=24)
    assert warnings == []


def test_no_data_warning_skips_provider_with_attempts():
    """no_data_warnings now reads _provider_attempt_count, not call_count.
    A provider with at least one outgoing attempt — success OR failure — is
    "live", so the warning must NOT fire. This is the honest semantic for
    'cold-start defaults still in use': if traffic is going out at all,
    we have data.
    """
    kdv = _kdv()
    kdv.register(model_id="groq/llama-3.3-70b", provider="groq", rpm=30, tpm=100000)
    kdv.mark_provider_enabled("groq", at_unix=time.time() - (25 * 3600))
    # Real attempt — even if it never reaches post_call (failure path)
    # — should clear the warning.
    kdv.record_attempt("groq/llama-3.3-70b", "groq")
    warnings = kdv.no_data_warnings(min_age_hours=24)
    assert warnings == []


def test_no_data_warning_fires_for_post_call_only():
    """Conversely: if only post_call ran (e.g. legacy code path that bumps
    success count without recording an attempt), the warning still fires.
    record_attempt is the single source of truth for 'is the provider being
    used'.
    """
    kdv = _kdv()
    kdv.register(model_id="groq/llama-3.3-70b", provider="groq", rpm=30, tpm=100000)
    kdv.mark_provider_enabled("groq", at_unix=time.time() - (25 * 3600))
    kdv.post_call(
        model_id="groq/llama-3.3-70b", provider="groq",
        headers=None, token_count=100,
    )
    warnings = kdv.no_data_warnings(min_age_hours=24)
    # post_call alone doesn't increment attempt_count, so warning fires.
    assert any(w["provider"] == "groq" for w in warnings)


def test_mark_provider_enabled_idempotent():
    """Re-marking same provider must NOT reset the enabled-at timestamp.
    Otherwise periodic refresh resets the clock and the warning never fires."""
    kdv = _kdv()
    kdv.register(model_id="groq/m", provider="groq", rpm=30, tpm=100000)
    t0 = time.time() - (25 * 3600)
    kdv.mark_provider_enabled("groq", at_unix=t0)
    kdv.mark_provider_enabled("groq")  # default = now
    warnings = kdv.no_data_warnings(min_age_hours=24)
    # Initial enable was 25h ago, so warning still fires.
    assert any(w["provider"] == "groq" for w in warnings)


def test_age_hours_in_warning_payload():
    kdv = _kdv()
    kdv.register(model_id="groq/m", provider="groq", rpm=30, tpm=100000)
    enabled_at = time.time() - (30 * 3600)
    kdv.mark_provider_enabled("groq", at_unix=enabled_at)
    warnings = kdv.no_data_warnings(min_age_hours=24)
    w = next(x for x in warnings if x["provider"] == "groq")
    assert w["age_hours"] >= 30 - 0.1  # within 6min slack
    assert w["enabled_at_unix"] == enabled_at
