"""A local LOAD failure must let the worker re-select a DIFFERENT model
(→ cloud) instead of terminating the task.

Live regression (2026-06-16): llama-server could not bind its port (Metro
held it), so every local load returned ``CallError(category="loading",
retryable=False)``. The transport loop treated ``not retryable`` as
"give up entirely" and raised ``ModelCallFailed`` immediately — even though
the failed local was already appended to the failure list and a re-select
would have picked an abundant cloud model. Result: tasks pinned to dead local
while cloud sat idle, looping forever.

``retryable=False`` means "do not retry the SAME model", NOT "abandon the
task". For ``loading`` / ``circuit_breaker`` the model is unavailable; the
loop must re-select (the failed model is excluded via the appended Failure)
and only terminate when re-selection yields nothing or attempts are spent.
"""
from __future__ import annotations

from types import SimpleNamespace

from coulson.dispatch_helpers import _transport_should_terminate


def _err(category: str, retryable: bool):
    return SimpleNamespace(category=category, retryable=retryable, message="x")


MAX = 3


def test_loading_failure_does_not_terminate_when_attempts_remain():
    # The chosen local model failed to load → re-select (→ cloud), don't quit.
    assert _transport_should_terminate(_err("loading", False), 0, MAX) is False


def test_circuit_breaker_failure_does_not_terminate_when_attempts_remain():
    assert _transport_should_terminate(_err("circuit_breaker", False), 1, MAX) is False


def test_loading_failure_terminates_when_attempts_exhausted():
    assert _transport_should_terminate(_err("loading", False), MAX, MAX) is True


def test_non_retryable_non_loading_error_terminates_immediately():
    # e.g. a malformed request — no point re-selecting a different model.
    assert _transport_should_terminate(_err("bad_request", False), 0, MAX) is True


def test_retryable_transient_error_does_not_terminate():
    assert _transport_should_terminate(_err("timeout", True), 0, MAX) is False
