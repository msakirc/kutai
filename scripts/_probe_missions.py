import sqlite3, sys, json
MID = int(sys.argv[1]) if len(sys.argv) > 1 else 71
p = "C:/Users/sakir/ai/kutai/kutai.db"
c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
c.row_factory = sqlite3.Row
cur = c.cursor()

def cols(t):
    return [r[1] for r in cur.execute("PRAGMA table_info(%s)" % t).fetchall()]

def q(s, a=()):
    try:
        return cur.execute(s, a).fetchall()
    except Exception as e:
        return [("ERR", str(e))]

print("== tasks cols ==", cols("tasks"))
print("== mission ==")
for r in q("select * from missions where id=?", (MID,)):
    print(dict(r))
print("== status counts (mission %d) ==" % MID)
for r in q("select status, count(*) from tasks where mission_id=? group by 1", (MID,)):
    print(" ", tuple(r))
print("== processing/pending head ==")
for r in q("select id, status, agent_type, substr(coalesce(title,''),0,50) t from tasks where mission_id=? and status in ('processing','pending') order by id limit 8", (MID,)):
    print(" ", tuple(r))
print("== last completed (10) ==")
for r in q("select id, agent_type, substr(coalesce(title,''),0,50) t from tasks where mission_id=? and status='completed' order by id desc limit 10", (MID,)):
    print(" ", tuple(r))
print("== dlq ==")
for r in q("select id, agent_type, substr(coalesce(title,''),0,60) t from tasks where mission_id=? and status in ('dlq','failed') order by id", (MID,)):
    print(" ", tuple(r))
print("== model_pick_log last 5 ==")
for r in q("select picked_model, call_category, round(picked_score,2) from model_pick_log order by rowid desc limit 5"):
    print(" ", tuple(r))
print("== mission_lessons count ==", q("select count(*) from mission_lessons"))
c.close()
