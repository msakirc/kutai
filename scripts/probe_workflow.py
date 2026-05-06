import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Force WAL read by opening with proper journaling
c = sqlite3.connect(r"C:\Users\sakir\ai\kutai\kutai.db", timeout=30)
c.row_factory = sqlite3.Row

print("journal_mode:", c.execute("PRAGMA journal_mode").fetchone()[0])
print("total tasks:", c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
print("total missions:", c.execute("SELECT COUNT(*) FROM missions").fetchone()[0])
print()

r = c.execute(
    "SELECT t.id, t.title, t.agent_type, t.mission_id, m.workflow, t.quality_score "
    "FROM tasks t LEFT JOIN missions m ON m.id=t.mission_id "
    "WHERE t.title LIKE '[%' AND t.status='completed' "
    "ORDER BY t.id DESC LIMIT 8"
).fetchall()
print(f"step-id completed tasks: {len(r)}")
for x in r:
    print(f"  task#{x['id']} mission={x['mission_id']} wf={x['workflow']} q={x['quality_score']} agent={x['agent_type']} title={x['title'][:55]}")
