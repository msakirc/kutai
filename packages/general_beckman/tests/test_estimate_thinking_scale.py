"""Admission token-estimate must scale output for thinking tasks.

Regression: mission 81 ADR step 4.1 (architect, needs_thinking=True) DLQ'd
because the admission estimate read step_token_stats.out_p90 (~1579) WITHOUT
the THINKING_OUT_SCALE — even though `estimate_for` supports it. The flag was
never passed at the call site, so the KDV TPM gate (and the max_tokens budget
it feeds) under-counted thinking output and the artifact truncated.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from general_beckman import _estimate_task_tokens


_BTABLE = {
    ("architect", "4.1", "phase_4"): {
        "samples_n": 29, "in_p90": 35373, "out_p90": 1579, "iters_p90": 2,
    },
}


def test_thinking_task_scales_out_estimate():
    """needs_thinking=True -> out_p90 scaled by THINKING_OUT_SCALE (2.0)."""
    ctx = {"workflow_step_id": "4.1", "workflow_phase": "phase_4",
           "needs_thinking": True}
    _, out = _estimate_task_tokens("architect", ctx, _BTABLE)
    assert out == 3158  # 1579 * 2.0


def test_nonthinking_task_unscaled():
    """No needs_thinking -> raw out_p90, no scaling."""
    ctx = {"workflow_step_id": "4.1", "workflow_phase": "phase_4",
           "needs_thinking": False}
    _, out = _estimate_task_tokens("architect", ctx, _BTABLE)
    assert out == 1579


def test_missing_flag_unscaled():
    """Absent needs_thinking key behaves as non-thinking."""
    ctx = {"workflow_step_id": "4.1", "workflow_phase": "phase_4"}
    _, out = _estimate_task_tokens("architect", ctx, _BTABLE)
    assert out == 1579
