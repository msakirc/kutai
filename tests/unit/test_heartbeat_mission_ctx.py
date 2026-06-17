def test_current_mission_id_exists_and_defaults_none():
    from src.core import heartbeat as hb
    assert hb.current_mission_id.get() is None


def test_current_mission_id_set_get():
    from src.core import heartbeat as hb
    tok = hb.current_mission_id.set(99)
    try:
        assert hb.current_mission_id.get() == 99
    finally:
        hb.current_mission_id.reset(tok)
