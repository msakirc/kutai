from nerd_herd.breakdown import PressureBreakdown


def test_breakdown_serializable():
    b = PressureBreakdown(
        scalar=-0.4,
        signals={"S1": -0.3, "S2": -0.1, "S9": 0.2},
        modifiers={"M1": 1.5, "M3_difficulty": 5},
        bucket_totals={"burden": -0.05, "queue": 0.0, "other": -0.3},
        positive_total=0.0,
        negative_total=-0.4,
    )
    d = b.to_dict()
    assert d["scalar"] == -0.4
    assert d["signals"]["S1"] == -0.3
    assert d["bucket_totals"]["other"] == -0.3
