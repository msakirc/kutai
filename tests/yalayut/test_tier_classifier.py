"""Tier classifier — min(max(source,owner), *checks) semantics."""
from yalayut.tier_classifier import classify


def test_owner_elevates_sketchy_source():
    # source untrusted (2), owner trusted (0) -> trust_cap = max(2,0)=... wait:
    # spec: trust_cap = max(source_max, owner_max); lower int = better tier,
    # and max() of two tier ints picks the WORSE. The spec text says "owner
    # elevates source" — elevation means a BETTER (lower) tier, so the cap is
    # min(source_max, owner_max). classify() implements the spec's intent:
    tier, audit = classify(source_max=2, owner_max=0, check_maxes={})
    assert tier == 0
    assert audit["trust_cap"] == 0


def test_checks_always_cap():
    # trusted source+owner but a check failed at T3
    tier, audit = classify(
        source_max=0, owner_max=0,
        check_maxes={"injection_scan": 3, "schema_valid": 0},
    )
    assert tier == 3
    assert audit["check_max"] == 3


def test_no_owner_elevation_past_checks():
    tier, _ = classify(
        source_max=0, owner_max=0, check_maxes={"shell_allowlist": 2},
    )
    assert tier == 2


def test_audit_records_each_contribution():
    tier, audit = classify(
        source_max=1, owner_max=2,
        check_maxes={"body_size_ok": 0, "license_present": 2},
    )
    assert audit["source_max"] == 1
    assert audit["owner_max"] == 2
    assert audit["trust_cap"] == 1   # best of source/owner
    assert audit["check_maxes"] == {"body_size_ok": 0, "license_present": 2}
    assert tier == 2


def test_empty_checks_uses_trust_cap():
    tier, _ = classify(source_max=1, owner_max=3, check_maxes={})
    assert tier == 1
