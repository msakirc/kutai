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
