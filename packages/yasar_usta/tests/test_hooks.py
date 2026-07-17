from yasar_usta.hooks import load_hook, run_pre_boot
from yasar_usta.config import GuardConfig, ProjectConfig


def test_load_hook_returns_none_for_missing():
    assert load_hook(None) is None
    assert load_hook("yasar_usta.projects.nope.hooks") is None


def test_pre_boot_invoked(tmp_path):
    calls = []
    class _Hook:
        @staticmethod
        def pre_boot(project):
            calls.append(project.id)
    proj = ProjectConfig(id="demo", name="Demo",
                         targets=[GuardConfig(name="t", command=["python"])])
    run_pre_boot(_Hook, proj)
    assert calls == ["demo"]


def test_kutai_hook_importable():
    hook = load_hook("yasar_usta.projects.kutai.hooks")
    assert hook is not None
    assert hasattr(hook, "pre_boot")
    assert hasattr(hook, "on_exit")
