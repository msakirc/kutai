"""Unit tests for intersect.budget caps."""
from intersect import budget


def _app(artifact_type, exposure_class="tool", server="srv-a",
         confidence=0.6, name="x", artifact_id=1):
    return {
        "artifact_id": artifact_id,
        "name": name,
        "artifact_type": artifact_type,
        "exposure_class": exposure_class,
        "mcp_server": server,
        "confidence": confidence,
    }


def test_api_capped_at_three_per_step():
    apps = [_app("api", confidence=0.9 - i * 0.1, artifact_id=i)
            for i in range(6)]
    kept, dropped = budget.apply_caps(apps)
    api_kept = [a for a in kept if a["artifact_type"] == "api"]
    assert len(api_kept) == 3
    assert len(dropped) == 3
    # Highest-confidence ones survive.
    assert all(a["confidence"] >= 0.6 for a in api_kept)


def test_mcp_capped_three_per_server():
    apps = [_app("mcp", server="srv-a", confidence=0.9 - i * 0.05,
                 artifact_id=i) for i in range(5)]
    kept, dropped = budget.apply_caps(apps)
    assert len([a for a in kept if a["artifact_type"] == "mcp"]) == 3
    assert len(dropped) == 2


def test_mcp_total_capped_six_per_step():
    apps = []
    for srv in ("a", "b", "c"):
        apps += [_app("mcp", server=srv, confidence=0.8,
                      artifact_id=hash((srv, i)) & 0xFFFF) for i in range(3)]
    kept, _ = budget.apply_caps(apps)
    assert len([a for a in kept if a["artifact_type"] == "mcp"]) == 6


def test_inject_and_preempt_never_capped():
    apps = [_app("skill", exposure_class="inject", artifact_id=i)
            for i in range(10)]
    apps.append(_app("skill", exposure_class="preempt", artifact_id=99))
    kept, dropped = budget.apply_caps(apps)
    assert len(kept) == 11
    assert dropped == []
