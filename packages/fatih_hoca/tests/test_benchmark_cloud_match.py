from pathlib import Path

import pytest

from fatih_hoca.benchmark_cloud_match import (
    apply_cloud_benchmarks,
    write_review_artifact,
    is_family_approved,
)
from fatih_hoca.registry import ModelInfo


def _model(name: str, provider: str, family: str) -> ModelInfo:
    return ModelInfo(
        name=name, location="cloud", provider=provider, litellm_name=name,
        capabilities={"reasoning": 6.0, "coding": 6.0},
        context_length=128000, max_tokens=4096,
        rate_limit_rpm=30, rate_limit_tpm=100000,
        family=family,
    )


def test_aa_hit_stored_but_not_active_until_approved(tmp_path: Path):
    aa_lookup = {"llama-3.3-70b": {"reasoning": 7.5, "coding": 7.0}}
    models = [_model("groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b")]
    apply_cloud_benchmarks(models, aa_lookup, approved_path=tmp_path / "approved.json")
    m = models[0]
    assert m.benchmark_scores == {"reasoning": 7.5, "coding": 7.0}
    # Not approved yet → active capabilities untouched.
    assert m.capabilities["reasoning"] == 6.0


def test_approved_family_promotes_aa_to_active(tmp_path: Path):
    approved = tmp_path / "approved.json"
    approved.write_text('["llama-3.3-70b"]')
    aa_lookup = {"llama-3.3-70b": {"reasoning": 7.5, "coding": 7.0}}
    models = [_model("groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b")]
    apply_cloud_benchmarks(models, aa_lookup, approved_path=approved)
    m = models[0]
    assert m.capabilities["reasoning"] == 7.5
    assert m.capabilities["coding"] == 7.0


def test_cross_provider_share_family_score(tmp_path: Path):
    approved = tmp_path / "approved.json"
    approved.write_text('["llama-3.3-70b"]')
    aa_lookup = {"llama-3.3-70b": {"reasoning": 7.5, "coding": 7.0}}
    groq_m = _model("groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b")
    cerebras_m = _model("cerebras/llama3.3-70b", "cerebras", "llama-3.3-70b")
    apply_cloud_benchmarks([groq_m, cerebras_m], aa_lookup, approved_path=approved)
    assert groq_m.capabilities["reasoning"] == 7.5
    assert cerebras_m.capabilities["reasoning"] == 7.5


def test_review_artifact_written(tmp_path: Path):
    aa_lookup = {"llama-3.3-70b": {"reasoning": 7.5}}
    models = [_model("groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b")]
    artifact = tmp_path / "review.json"
    apply_cloud_benchmarks(models, aa_lookup, approved_path=tmp_path / "approved.json")
    write_review_artifact(models, aa_lookup, output_path=artifact)
    assert artifact.exists()
    import json
    rows = json.loads(artifact.read_text())
    assert rows[0]["litellm_name"] == "groq/llama-3.3-70b-versatile"
    assert rows[0]["family"] == "llama-3.3-70b"
    assert rows[0]["matched_aa_entry"] == "llama-3.3-70b"
    assert rows[0]["source"] == "aa"


def test_no_aa_hit_keeps_profile_or_default(tmp_path: Path):
    models = [_model("groq/some-future", "groq", "some-future")]
    apply_cloud_benchmarks(models, aa_lookup={}, approved_path=tmp_path / "approved.json")
    assert not models[0].benchmark_scores
    assert models[0].capabilities["reasoning"] == 6.0


def test_is_family_approved_function(tmp_path: Path):
    approved = tmp_path / "approved.json"
    approved.write_text('["llama-3.3-70b", "claude-sonnet-4"]')
    assert is_family_approved("llama-3.3-70b", approved) is True
    assert is_family_approved("gpt-4o", approved) is False
    # Missing file → no families approved.
    assert is_family_approved("llama-3.3-70b", tmp_path / "nope.json") is False


def test_skips_local_and_familyless(tmp_path: Path):
    """Only cloud models with non-empty family are considered."""
    aa_lookup = {"llama-3.3-70b": {"reasoning": 7.5}}
    local = _model("local/model", "llama_cpp", "llama-3.3-70b")
    local.location = "local"
    no_family = _model("groq/x", "groq", "")
    cloud_with_family = _model("groq/y", "groq", "llama-3.3-70b")
    approved = tmp_path / "approved.json"
    approved.write_text('["llama-3.3-70b"]')
    apply_cloud_benchmarks([local, no_family, cloud_with_family], aa_lookup, approved_path=approved)
    # Local untouched.
    assert local.capabilities["reasoning"] == 6.0
    # No-family cloud untouched.
    assert no_family.capabilities["reasoning"] == 6.0
    # Cloud with approved family promoted.
    assert cloud_with_family.capabilities["reasoning"] == 7.5
