from nerd_herd.nerd_herd import NerdHerd


def test_nerd_herd_exposes_swap_api():
    nh = NerdHerd(metrics_port=0)
    assert nh.recent_swap_count() == 0
    nh.record_swap("model_a")
    assert nh.recent_swap_count() == 1
    assert nh.can_swap() is True


def test_nerd_herd_swap_budget_configurable():
    nh = NerdHerd(metrics_port=0)
    for i in range(3):
        nh.record_swap(f"m{i}")
    assert nh.can_swap() is False
