"""Tests for yasar_usta.config."""

from yasar_usta.config import GuardConfig, Messages, SidecarConfig


class TestMessages:
    def test_default_messages_have_no_empty_strings(self):
        msgs = Messages()
        for field_name in vars(msgs):
            val = getattr(msgs, field_name)
            assert isinstance(val, str) and len(val) > 0, f"Empty message: {field_name}"

    def test_messages_support_format_placeholders(self):
        msgs = Messages()
        result = msgs.started.format(app_name="MyApp")
        assert "MyApp" in result

    def test_custom_messages(self):
        msgs = Messages(started="✅ *{app_name} Başladı*")
        assert "Başladı" in msgs.started


class TestGuardConfig:
    def test_defaults(self):
        cfg = GuardConfig()
        assert cfg.backoff_steps == [5, 15, 60, 300]
        assert cfg.restart_exit_code == 42
        assert cfg.heartbeat_stale_seconds == 120
        assert cfg.auto_restart is True

    def test_custom_config(self):
        cfg = GuardConfig(
            name="Yaşar Usta",
            app_name="Kutay",
            command=["python", "run.py"],
            restart_exit_code=42,
            backoff_steps=[1, 5, 30],
        )
        assert cfg.app_name == "Kutay"
        assert cfg.backoff_steps == [1, 5, 30]

    def test_independent_list_defaults(self):
        a = GuardConfig()
        b = GuardConfig()
        a.backoff_steps.append(999)
        assert 999 not in b.backoff_steps


class TestSidecarConfig:
    def test_defaults(self):
        sc = SidecarConfig()
        assert sc.detached is True
        assert sc.auto_start is True
        assert sc.command == []

    def test_custom(self):
        sc = SidecarConfig(
            name="yazbunu",
            command=["python", "-m", "yazbunu.server"],
            health_url="http://127.0.0.1:9880/",
            pid_file="logs/yazbunu.pid",
        )
        assert sc.name == "yazbunu"
        assert sc.health_url == "http://127.0.0.1:9880/"
