from nerd_herd.signals._curves import smoothstep


def test_smoothstep_endpoints():
    assert smoothstep(-1.0) == 0.0
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(2.0) == 1.0


def test_smoothstep_midpoint_and_shape():
    # Hermite 3x^2 - 2x^3: midpoint 0.5, quiet near 0, steep in the middle.
    assert smoothstep(0.5) == 0.5
    assert smoothstep(0.1) < 0.05          # near-zero stays quiet
    assert smoothstep(0.9) > 0.95          # saturates near 1
    # Strictly monotonic on (0, 1)
    xs = [i / 20 for i in range(21)]
    ys = [smoothstep(x) for x in xs]
    assert all(b > a for a, b in zip(ys, ys[1:]))


def test_s12_still_uses_shared_curve():
    # S12's private _smoothstep is now the shared one (no drift).
    from nerd_herd.signals import s12_pool_balance as s12
    assert s12._smoothstep(0.5) == 0.5
    assert s12._smoothstep(0.1) == smoothstep(0.1)
