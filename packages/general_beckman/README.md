# General Beckman — Task Master

Named after General Beckman from *Chuck* — the NSA commander who hands out missions. This package owns the task queue and answers: **"what should we do next, and how many of it?"**

## Public API

```python
import general_beckman as beckman

task = await beckman.next_task()          # release one eligible task or None
await beckman.on_task_finished(tid, res)  # drain lifecycle handlers
await beckman.tick()                      # watchdog + scheduled jobs
```

## Tests

```
timeout 30 pytest packages/general_beckman/tests/ -v
```
