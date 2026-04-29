from nerd_herd.types import QueueProfile


def test_queue_profile_widened_fields():
    qp = QueueProfile(
        total_ready_count=10,
        hard_tasks_count=3,
        by_difficulty={3: 4, 7: 3, 9: 1},
        by_capability={"vision": 2, "thinking": 1, "function_calling": 8},
        projected_tokens=120_000,
        projected_calls=80,
    )
    assert qp.by_difficulty[7] == 3
    assert qp.by_capability["vision"] == 2
    assert qp.projected_tokens == 120_000
    assert qp.projected_calls == 80
