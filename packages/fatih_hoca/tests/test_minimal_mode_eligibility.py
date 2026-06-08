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


def _selector():
    reg = SimpleNamespace(is_dead=lambda *_: False, is_provider_dead=lambda *_: False)
    return Selector(registry=reg, nerd_herd=SimpleNamespace())


def test_minimal_mode_rejects_local():
    sel = _selector()
    snap = SimpleNamespace(load_mode="minimal", vram_available_mb=8000, cloud={})
    reqs = ModelRequirements(task="coder", difficulty=5)
    reason = sel._check_eligibility(model=_local_model(), reqs=reqs,
                                    failed_models=set(), snapshot=snap)
    assert reason == "load_mode_minimal"


def test_full_mode_allows_local():
    sel = _selector()
    snap = SimpleNamespace(load_mode="full", vram_available_mb=8000, cloud={})
    reqs = ModelRequirements(task="coder", difficulty=5)
    reason = sel._check_eligibility(model=_local_model(), reqs=reqs,
                                    failed_models=set(), snapshot=snap)
    assert reason is None
