from yasar_usta.status import build_dashboard_text
from yasar_usta.commands import build_dashboard_keyboard, build_hub_reply_keyboard
from yasar_usta.config import Messages


def test_hub_reply_keyboard_is_minimal():
    kb = build_hub_reply_keyboard(Messages(btn_status="Durum", btn_logs="Loglar",
                                           btn_remote="Claude"))
    flat = str(kb)
    assert "Durum" in flat and "Loglar" in flat and "Claude" in flat
    # No per-target Start/Restart/Stop on the persistent keyboard (spec R4)
    assert "Start" not in flat and "Restart" not in flat


def test_dashboard_lists_all_projects():
    projects = [
        {"project_id": "kutai", "name": "Kutay", "app_name": "Kutay",
         "running": True, "heartbeat_age": 3.0, "heartbeat_healthy_seconds": 90,
         "total_crashes": 0, "extra_processes": []},
        {"project_id": "foo", "name": "Foo", "app_name": "Foo",
         "running": False, "heartbeat_age": None, "heartbeat_healthy_seconds": 90,
         "total_crashes": 2, "extra_processes": []},
    ]
    text = build_dashboard_text("Yaşar Usta", projects, guard_start_time=0.0)
    assert "Kutay" in text and "Foo" in text
    assert "healthy" in text  # kutai
    assert "not running" in text  # foo


def test_dashboard_keyboard_has_per_project_callbacks():
    kb = build_dashboard_keyboard([
        {"project_id": "kutai", "name": "Kutay", "running": True},
        {"project_id": "foo", "name": "Foo", "running": False},
    ])
    flat = str(kb)
    assert "restart:kutai" in flat
    assert "start:foo" in flat
    assert "restart_hub" in flat
    assert "dashboard_refresh" in flat


def test_dashboard_shows_sidecar_health():
    from yasar_usta.status import build_project_section
    st = {"project_id": "kutai", "name": "Kutay", "app_name": "Kutay",
          "running": True, "heartbeat_age": 2.0, "heartbeat_healthy_seconds": 90,
          "total_crashes": 0, "extra_processes": [], "app_script": None,
          "sidecar_health": [
              {"name": "yazbunu", "http_alive": True, "pid": 111, "alive": True},
              {"name": "nerd_herd", "http_alive": False, "pid": None, "alive": False}]}
    text = build_project_section(st)
    assert "yazbunu" in text and "running" in text
    assert "nerd_herd" in text and "not running" in text


def test_dashboard_keyboard_has_sidecar_restart():
    from yasar_usta.commands import build_dashboard_keyboard
    kb = build_dashboard_keyboard([
        {"project_id": "kutai", "name": "Kutay", "running": True,
         "sidecar_health": [{"name": "yazbunu"}, {"name": "nerd_herd"}]}])
    flat = str(kb)
    assert "restart_sidecar:kutai:yazbunu" in flat
    assert "restart_sidecar:kutai:nerd_herd" in flat
