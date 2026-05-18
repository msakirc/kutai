"""Package-scaffold smoke test for intersect."""


def test_intersect_imports_and_exposes_flash():
    import intersect
    assert hasattr(intersect, "flash")
    assert callable(intersect.flash)


def test_intersect_exposes_thresholds():
    import intersect
    # Exposure thresholds must be importable for ops tuning.
    assert isinstance(intersect.THETA_PREEMPT, float)
    assert isinstance(intersect.THETA_INJECT, float)
    assert isinstance(intersect.THETA_TOOL, float)
    assert isinstance(intersect.THETA_MIN, float)
    assert intersect.THETA_PREEMPT > intersect.THETA_INJECT
    assert intersect.THETA_INJECT > intersect.THETA_TOOL
    assert intersect.THETA_TOOL > intersect.THETA_MIN
