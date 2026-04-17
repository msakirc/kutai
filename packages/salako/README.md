# Salako — Mechanical Dispatcher

Turkish nickname for the steadfast worker who just does the grunt work without asking.

Salako is the non-LLM executor sibling to the LLM dispatcher. It runs mechanical,
deterministic tasks such as workspace snapshots and git commits — the kind of work
that must not burn model tokens or swap budget.

## Public API

```python
import salako

# Route a task dict based on its payload.action
action = await salako.run(task)   # -> salako.Action(status, result, error)

# Direct helpers (backwards-compatible)
snap = await salako.snapshot_workspace(mission_id, task_id, workspace_path)
await salako.auto_commit(task, result)
```

`Action.status` is one of `"completed" | "failed" | "skipped"`.

## Supported actions

| `payload.action`      | Behavior                                                 |
|-----------------------|----------------------------------------------------------|
| `workspace_snapshot`  | Hash files + record git SHA/branch into DB               |
| `git_commit`          | Stage & commit the mission workspace                     |

## Example task payload

```python
task = {
    "id": 42,
    "mission_id": 7,
    "executor": "mechanical",
    "payload": {
        "action": "workspace_snapshot",
        "workspace_path": "/path/to/ws",
        "repo_path": None,
    },
}
```

## Tests

```bash
pip install -e packages/salako
python -m pytest packages/salako/tests/
```

## TODO / notes

- `salako.workspace_snapshot` still imports from `src.infra.db` and `src.tools.*`.
  That is load-bearing within the KutAI repo (repo root is on `sys.path`). If this
  package is ever used standalone, those should be injected as dependencies.
