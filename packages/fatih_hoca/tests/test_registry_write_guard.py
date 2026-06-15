"""Guard: raw writes to model-registry tables must live in fatih_hoca (the
owner) or a small set of explicitly sanctioned writers. Locks in the Phase B
ownership move so new raw registry SQL elsewhere fails CI."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SQL = re.compile(
    r'(INSERT\s+INTO\s+(model_stats|model_pick_log|providers|models|registry_events)\b'
    r'|UPDATE\s+(model_stats|model_pick_log|providers|models|registry_events)\s+SET'
    r'|DELETE\s+FROM\s+(model_stats|model_pick_log|providers|models|registry_events)\b)',
    re.IGNORECASE,
)
ALLOWED = {
    ROOT / "packages/fatih_hoca/src/fatih_hoca/db.py",
    ROOT / "packages/fatih_hoca/src/fatih_hoca/schema.py",
    ROOT / "packages/fatih_hoca/src/fatih_hoca/registry_store.py",
    # sms_send writes a distinct scope='sms_send' registry_events row (sms cap
    # counter) — cannot use record_action_event (forces scope='action').
    ROOT / "packages/mr_roboto/src/mr_roboto/executors/sms_send.py",
}
ALLOWED = {p.resolve() for p in ALLOWED}


def test_no_registry_writes_outside_fatih():
    violations = []
    for base in ("src", "packages"):
        for p in (ROOT / base).rglob("*.py"):
            if "tests" in p.parts or p.resolve() in ALLOWED:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if SQL.search(line):
                    violations.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert violations == [], (
        "Raw registry-table writes found outside fatih_hoca — route through the "
        "fatih registry API or add a justified entry to ALLOWED:\n" + "\n".join(violations)
    )
