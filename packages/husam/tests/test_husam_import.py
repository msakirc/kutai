def test_husam_imports_and_exposes_run():
    import husam
    assert callable(husam.run)

def test_husam_does_not_import_coulson():
    """Purity: husam must not drag coulson in at import time.

    Uses a subprocess so the check is immune to cross-test session pollution
    (another test in the same session may have already imported coulson before
    this test runs, producing a false positive when checking sys.modules
    in-process).
    """
    import os
    import pathlib
    import subprocess
    import sys

    worktree = pathlib.Path(__file__).parent.parent.parent.parent
    packages = [
        "fatih_hoca", "nerd_herd", "kuleden_donen_var", "general_beckman",
        "hallederiz_kadir", "dallama", "dogru_mu_samet", "vecihi",
        "yasar_usta", "yazbunu", "mr_roboto", "coulson", "sade_kalsin",
        "c21_paraflow_diff", "intersect", "yalayut", "safety_guard", "husam",
    ]
    paths = [str(worktree)]
    for pkg in packages:
        p = worktree / "packages" / pkg / "src"
        if p.is_dir():
            paths.append(str(p))
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    pythonpath = os.pathsep.join(paths)

    code = (
        "import husam, sys; "
        "bad = [m for m in sys.modules if m == 'coulson' or m.startswith('coulson.')]; "
        "assert not bad, f'husam import pulled in coulson modules: {bad}'"
    )
    env = {**os.environ, "PYTHONPATH": pythonpath}
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(worktree),
        env=env,
    )
    assert r.returncode == 0, (
        f"husam import pulled in coulson (subprocess check):\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}"
    )
