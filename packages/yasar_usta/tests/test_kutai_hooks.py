"""KutAI orphan-orchestrator cleanup — precise script-path matching.

Replaces the bare ``"run.py"`` substring foot-gun (which force-killed library
run.py — torch/pexpect/openai/watchfiles — and other projects' orchestrators)
with a match on the target's absolute run-script path. Also drops the
deprecated ``wmic`` dependency (psutil enumeration).
"""

from types import SimpleNamespace

from yasar_usta.projects.kutai.hooks import (
    _norm, _orchestrator_script_paths, _stale_orchestrator_pids,
    _kill_stale_orchestrators,
)


def _project(*commands):
    return SimpleNamespace(targets=[SimpleNamespace(command=c) for c in commands])


class TestScriptPaths:
    def test_extracts_py_args_from_command(self):
        proj = _project(["C:/proj/.venv/python.exe", "C:/proj/src/app/run.py"])
        assert _orchestrator_script_paths(proj) == {_norm("C:/proj/src/app/run.py")}

    def test_ignores_non_py_args(self):
        proj = _project(["python", "-m", "yazbunu.server", "--port", "9880"])
        assert _orchestrator_script_paths(proj) == set()


class TestStaleMatch:
    def test_matches_exact_path_not_library_runpy(self):
        sp = {_norm("C:/proj/src/app/run.py")}
        procs = [
            (100, "python C:\\proj\\src\\app\\run.py"),
            (200, "python C:\\py\\site-packages\\torch\\distributed\\run.py"),
            (300, "python C:/proj/src/app/run.py --once"),
        ]
        assert _stale_orchestrator_pids(sp, procs, my_pid=999) == [100, 300]

    def test_excludes_my_pid(self):
        sp = {_norm("C:/proj/src/app/run.py")}
        procs = [(999, "python C:/proj/src/app/run.py")]
        assert _stale_orchestrator_pids(sp, procs, my_pid=999) == []

    def test_other_project_not_matched(self):
        sp = {_norm("C:/projA/src/app/run.py")}
        procs = [(400, "python C:/projB/src/app/run.py")]
        assert _stale_orchestrator_pids(sp, procs, my_pid=1) == []

    def test_slash_direction_and_case_insensitive(self):
        sp = {_norm("C:\\Proj\\src\\app\\run.py")}
        procs = [(500, "python c:/proj/SRC/app/run.py")]
        assert _stale_orchestrator_pids(sp, procs, my_pid=1) == [500]


class TestKillStaleOrchestrators:
    def test_kills_only_matching_pids(self):
        proj = _project(["py.exe", "C:/proj/src/app/run.py"])
        procs = [
            (100, "python C:/proj/src/app/run.py"),
            (200, "python C:/py/torch/run.py"),
        ]
        killed = []
        _kill_stale_orchestrators(proj, list_processes=lambda: procs, kill=killed.append)
        assert killed == [100]

    def test_noop_when_no_py_in_command(self):
        proj = _project(["python", "-m", "mod"])
        killed = []
        _kill_stale_orchestrators(
            proj, list_processes=lambda: [(1, "python -m mod")], kill=killed.append)
        assert killed == []
