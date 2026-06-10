"""Task 16: dep-purity guardrail — the leaf imports nothing from src or feature pkgs."""
import subprocess
import sys
import pathlib

SRC = pathlib.Path(__file__).parent.parent / "src" / "prompt_foundry"

_FORBIDDEN = (
    "import src",
    "from src",
    "import coulson",
    "from coulson",
    "import husam",
    "from husam",
    "general_beckman",
    "fatih_hoca",
    "import yalayut",
    "from yalayut",
)


def test_no_src_or_feature_imports():
    """Scan all .py files under prompt_foundry/src for forbidden import markers."""
    bad = []
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for marker in _FORBIDDEN:
            if marker in text:
                bad.append((py.name, marker))
    assert not bad, f"Foundry leaf imports forbidden deps: {bad}"


def test_import_in_clean_subprocess():
    """Import prompt_foundry without src/ on path — proves leaf-ness."""
    r = subprocess.run(
        [sys.executable, "-c", "import prompt_foundry; print('ok')"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
