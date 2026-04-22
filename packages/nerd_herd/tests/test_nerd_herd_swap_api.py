from nerd_herd.nerd_herd import NerdHerd


def test_nerd_herd_exposes_swap_api():
    nh = NerdHerd(metrics_port=0)
    assert nh.recent_swap_count() == 0
    nh.record_swap("model_a")
    assert nh.recent_swap_count() == 1


def test_nerd_herd_record_swap_is_cumulative():
    nh = NerdHerd(metrics_port=0)
    for i in range(5):
        nh.record_swap(f"m{i}")
    assert nh.recent_swap_count() == 5
