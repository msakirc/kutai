from fatih_hoca.estimates import Estimates


def test_estimates_total_tokens_computed():
    e = Estimates(in_tokens=1000, out_tokens=2000, iterations=5)
    assert e.total_tokens == (1000 + 2000) * 5


def test_estimates_per_call_tokens():
    e = Estimates(in_tokens=1000, out_tokens=2000, iterations=5)
    assert e.per_call_tokens == 3000


def test_step_token_overrides_known_step():
    from fatih_hoca.step_overrides import STEP_TOKEN_OVERRIDES
    e = STEP_TOKEN_OVERRIDES["4.5b"]
    assert e.out_tokens >= 100_000  # openapi_spec is heavy


def test_avg_iterations_by_agent_seeded_from_telemetry():
    from fatih_hoca.requirements import AVG_ITERATIONS_BY_AGENT
    # 2026-04-28 telemetry: analyst avg 7.1, architect 11.8, researcher 23.4
    assert 6 <= AVG_ITERATIONS_BY_AGENT["analyst"] <= 9
    assert 10 <= AVG_ITERATIONS_BY_AGENT["architect"] <= 14
    assert 20 <= AVG_ITERATIONS_BY_AGENT["researcher"] <= 28


def test_agent_requirements_calibrated_to_p90():
    from fatih_hoca.requirements import AGENT_REQUIREMENTS
    # Telemetry showed analyst p90 = 25k tokens; old default 3k under-reserves 8x
    analyst = AGENT_REQUIREMENTS["analyst"]
    assert analyst.estimated_output_tokens >= 15_000  # at least p75


import asyncio


class FakeTask:
    def __init__(self, agent_type, step_id=None, phase=None):
        self.agent_type = agent_type
        self.context = {}
        if step_id:
            self.context["workflow_step_id"] = step_id
        if phase:
            self.context["workflow_phase"] = phase


def test_estimate_for_uses_static_overrides_when_step_known():
    from fatih_hoca.estimates import estimate_for
    task = FakeTask("architect", step_id="4.5b")
    e = estimate_for(task, btable={})
    assert e.out_tokens >= 100_000


def test_estimate_for_falls_back_to_agent_requirements():
    from fatih_hoca.estimates import estimate_for
    task = FakeTask("analyst")
    e = estimate_for(task, btable={})
    assert e.out_tokens >= 15_000


def test_estimate_for_uses_btable_when_samples_sufficient():
    from fatih_hoca.estimates import estimate_for, Estimates
    task = FakeTask("analyst", step_id="2.6", phase="phase_2")
    btable = {
        ("analyst", "2.6", "phase_2"): {
            "samples_n": 10,
            "in_p90": 5000, "out_p90": 4000, "iters_p90": 7,
        }
    }
    e = estimate_for(task, btable=btable)
    assert e.in_tokens == 5000
    assert e.out_tokens == 4000
    assert e.iterations == 7


def test_estimate_for_skips_btable_when_samples_below_threshold():
    from fatih_hoca.estimates import estimate_for
    task = FakeTask("analyst", step_id="4.5b", phase="phase_4")
    btable = {
        ("analyst", "4.5b", "phase_4"): {
            "samples_n": 2,  # below MIN_SAMPLES=5
            "in_p90": 100, "out_p90": 100, "iters_p90": 1,
        }
    }
    e = estimate_for(task, btable=btable)
    assert e.out_tokens >= 100_000  # falls through to STEP_TOKEN_OVERRIDES
