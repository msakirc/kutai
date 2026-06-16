"""While local inference is structurally down (llama-server cannot boot any
model), the selector must lay off ALL local at eligibility so tasks route to
cloud — instead of admitting every task against a dead server (live
2026-06-16: hours with not a single task processed).

_check_eligibility reads ``snapshot.local_inference_down``, which select()/
is_servable() overlay in-process from the nerd_herd client cache (the sidecar
has no load-outcome write path).
"""
from types import SimpleNamespace

from fatih_hoca.selector import Selector
from fatih_hoca.requirements import ModelRequirements


def _local_model():
    return SimpleNamespace(
        name="loc", litellm_name="loc", is_local=True, is_loaded=True,
        demoted=False, provider="", specialty=None, context_length=32000,
        supports_function_calling=True, supports_json_mode=True, has_vision=False,
        variant_flags=set(), max_input_tokens=None, rate_limit_tpm=0,
    )


def _cloud_model():
    m = _local_model()
    m.is_local = False
    m.provider = "gemini"
    return m


def _selector():
    reg = SimpleNamespace(is_dead=lambda *_: False, is_provider_dead=lambda *_: False)
    return Selector(registry=reg, nerd_herd=SimpleNamespace())


def _snap(down: bool):
    return SimpleNamespace(load_mode="full", vram_available_mb=8000, cloud={},
                           local_inference_down=down)


def test_local_rejected_when_inference_down():
    sel = _selector()
    reqs = ModelRequirements(task="coder", difficulty=5)
    reason = sel._check_eligibility(model=_local_model(), reqs=reqs,
                                    failed_models=set(), snapshot=_snap(True))
    assert reason == "local_server_down"


def test_local_allowed_when_inference_up():
    sel = _selector()
    reqs = ModelRequirements(task="coder", difficulty=5)
    reason = sel._check_eligibility(model=_local_model(), reqs=reqs,
                                    failed_models=set(), snapshot=_snap(False))
    assert reason is None


def test_cloud_unaffected_when_local_down():
    sel = _selector()
    reqs = ModelRequirements(task="coder", difficulty=5)
    # available_providers None → no api-key gate; cloud must NOT be rejected
    # for local_server_down.
    reason = sel._check_eligibility(model=_cloud_model(), reqs=reqs,
                                    failed_models=set(), snapshot=_snap(True))
    assert reason != "local_server_down"


def _servable_selector(down: bool):
    reg = SimpleNamespace(is_dead=lambda *_: False, is_provider_dead=lambda *_: False)
    nh = SimpleNamespace(
        snapshot=lambda: _snap(False),  # base snapshot; overlay sets the flag
        is_local_inference_down=lambda: down,
    )
    return Selector(registry=reg, nerd_herd=nh)


def test_is_servable_lays_off_unloaded_local_when_down():
    sel = _servable_selector(down=True)
    reqs = ModelRequirements(task="coder", difficulty=5)
    model = _local_model()
    model.is_loaded = False
    assert sel.is_servable(model=model, reqs=reqs) is False


def test_is_servable_keeps_held_loaded_local_when_down():
    # Down blocks STARTING new local load, not finishing in-flight work on an
    # already-resident model (mirrors the load_mode_minimal carve-out).
    sel = _servable_selector(down=True)
    reqs = ModelRequirements(task="coder", difficulty=5)
    model = _local_model()
    model.is_loaded = True
    assert sel.is_servable(model=model, reqs=reqs) is True
