"""QueueProfile + push_queue_profile — Task 14."""
from nerd_herd.nerd_herd import NerdHerd
from nerd_herd.types import QueueProfile


def test_push_queue_profile_stored_and_exposed():
    nh = NerdHerd(metrics_port=0)
    nh.push_queue_profile(QueueProfile(hard_tasks_count=4, total_ready_count=12))
    snap = nh.snapshot()
    assert snap.queue_profile is not None
    assert snap.queue_profile.hard_tasks_count == 4
    assert snap.queue_profile.total_ready_count == 12


def test_queue_profile_none_by_default():
    nh = NerdHerd(metrics_port=0)
    assert nh.snapshot().queue_profile is None
