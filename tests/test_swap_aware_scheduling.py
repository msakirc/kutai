import json
from src.core.orchestrator import _should_defer_for_loaded_model


def test_defer_when_model_excluded():
    task = {"worker_attempts": 3, "context": json.dumps({"failed_models": ["m1"]})}
    assert _should_defer_for_loaded_model(task, "m1") is True

def test_no_defer_below_threshold():
    task = {"worker_attempts": 2, "context": json.dumps({"failed_models": ["m1"]})}
    assert _should_defer_for_loaded_model(task, "m1") is False

def test_no_defer_different_model():
    task = {"worker_attempts": 3, "context": json.dumps({"failed_models": ["m1"]})}
    assert _should_defer_for_loaded_model(task, "m2") is False

def test_no_defer_no_context():
    task = {"worker_attempts": 3, "context": "{}"}
    assert _should_defer_for_loaded_model(task, "m1") is False

def test_defer_with_dict_context():
    task = {"worker_attempts": 4, "context": {"failed_models": ["m1", "m2"]}}
    assert _should_defer_for_loaded_model(task, "m2") is True

def test_legacy_attempts_field():
    """Backwards compat: reads 'attempts' if 'worker_attempts' missing."""
    task = {"attempts": 3, "context": json.dumps({"failed_models": ["m1"]})}
    assert _should_defer_for_loaded_model(task, "m1") is True
