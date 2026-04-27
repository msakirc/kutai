import json
import time
from pathlib import Path

import pytest

from fatih_hoca.cloud.cache import (
    CACHE_TTL_SECONDS,
    EVICT_TTL_SECONDS,
    CachedSnapshot,
    load,
    save,
)
from fatih_hoca.cloud.types import DiscoveredModel


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cloud_models"


def _model() -> DiscoveredModel:
    return DiscoveredModel(litellm_name="groq/foo", raw_id="foo", context_length=8192)


def test_save_then_load_round_trip(cache_dir):
    save(cache_dir, "groq", [_model()], status="ok")
    snap = load(cache_dir, "groq")
    assert snap is not None
    assert snap.status == "ok"
    assert snap.is_fresh is True
    assert snap.is_evicted is False
    assert snap.models[0].litellm_name == "groq/foo"


def test_load_missing_returns_none(cache_dir):
    assert load(cache_dir, "groq") is None


def test_stale_but_not_evicted(cache_dir):
    save(cache_dir, "groq", [_model()], status="ok")
    snap_path = cache_dir / "groq.json"
    raw = json.loads(snap_path.read_text())
    raw["fetched_at_unix"] = time.time() - (CACHE_TTL_SECONDS + 60)
    snap_path.write_text(json.dumps(raw))
    snap = load(cache_dir, "groq")
    assert snap is not None
    assert snap.is_fresh is False
    assert snap.is_evicted is False


def test_evicted_returns_none(cache_dir):
    save(cache_dir, "groq", [_model()], status="ok")
    snap_path = cache_dir / "groq.json"
    raw = json.loads(snap_path.read_text())
    raw["fetched_at_unix"] = time.time() - (EVICT_TTL_SECONDS + 60)
    snap_path.write_text(json.dumps(raw))
    assert load(cache_dir, "groq") is None


def test_save_overwrites(cache_dir):
    save(cache_dir, "groq", [_model()], status="ok")
    new_model = DiscoveredModel(litellm_name="groq/bar", raw_id="bar")
    save(cache_dir, "groq", [new_model], status="ok")
    snap = load(cache_dir, "groq")
    assert snap is not None
    assert {m.litellm_name for m in snap.models} == {"groq/bar"}
