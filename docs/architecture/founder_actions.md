# founder_actions — architecture note

Real-world handoffs from the agent to the founder. When the agent has
prepared everything it can autonomously but the next step needs human
delegation (vendor enrollment, credential paste, cost ack, legal
counsel, KYC), it writes a row into the ``founder_actions`` table and
parks the dependent task/mission until the founder responds.

Shipped in Z6 T1B → T7B (2026-05-11).

## Table schema

```sql
CREATE TABLE founder_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id INTEGER NOT NULL,            -- 0 = system-scoped
    blocking_task_id INTEGER,               -- NULL = mission-wide
    blocking_step_id TEXT,                  -- e.g. "13.1"
    kind TEXT NOT NULL,                     -- see kinds below
    title TEXT NOT NULL,
    why TEXT NOT NULL,
    instructions_json TEXT NOT NULL,        -- list[str] of steps
    expected_output_kind TEXT,              -- credential | url | receipt | ack_only | free_text
    expected_output_schema_json TEXT,       -- JSON Schema if structured
    cost_estimate_usd REAL,
    reversibility TEXT,                     -- full | partial | irreversible
    status TEXT NOT NULL DEFAULT 'pending',
    response_payload_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT
);
```

## Kinds

| Kind | Use case |
|---|---|
| ``credential_paste`` | Founder generates a token in vendor UI, pastes back |
| ``vendor_enroll`` | KYC + signup + payment method at a vendor (Stripe, Apple, Google Play) |
| ``cost_ack`` | Confirm spend before irreversible operation |
| ``legal_counsel`` | Send doc to counsel; sign / file |
| ``kyc`` | Identity verification at a specific vendor |
| ``generic`` | Free-text fallback |

## Lifecycle

```
pending ──► in_progress ──► done       (terminal)
   │             │
   ├─► blocked ──┤                     (recoverable — founder can retry)
   └─► cancelled                       (terminal)
```

State machine is enforced by ``src/founder_actions/__init__.py::update_status``;
``done`` and ``cancelled`` have no outgoing edges. ``resolved_at`` is set
when the action transitions to ``done`` or ``cancelled``.

## Wake-up flow

1. ``src.founder_actions.create()`` inserts a ``pending`` row.
2. Best-effort: posts a Telegram inline card to the mission thread via
   the registered notifier (``register_notifier``).
3. ``block_mission_if_needed()`` flips the mission to
   ``status='blocked_on_founder_action'`` (or the Z0
   ``lifecycle_state`` column when present) if any pending/in_progress
   actions exist and the mission is currently ``active``.
4. Beckman's ``next_task()`` admission gate parks dependent tasks in
   ``status='blocked_on_founder_action'``.
5. Founder responds via inline button or ``/action_done <id> [payload]``.
6. ``resolve()`` calls ``unblock_mission_if_clear()`` — when no
   pending/in_progress actions remain, the mission flips back to
   ``active`` and the orchestrator's next pump cycle re-evaluates the
   parked tasks.

## Beckman admission gate (T1C)

``packages/general_beckman/src/general_beckman/z6_admission.py::check_z6_admission``
runs before dispatching any ``needs_real_tools=true`` task. It emits
one founder_action per missing prereq:

| Missing prereq | Emitted kind |
|---|---|
| ``real_tool_kind`` not declared | ``generic`` |
| no adapter registered for any kind | ``vendor_enroll`` |
| no credentials for the matched kind | ``credential_paste`` |
| irreversible + cost>0 with no prior ack | ``cost_ack`` |

De-dup: the gate checks for an existing pending/in_progress action with
the same ``(kind, step_id)`` pair before creating a new one.

## Coulson detect-and-bail (T7C)

``packages/coulson/src/coulson/__init__.py::_maybe_detect_and_bail``
re-runs the admission check inside the runtime's ``execute()`` entry
point. When prereqs are missing it short-circuits with
``status='blocked_on_founder_action'`` — the LLM is never called.
When prereqs are satisfied it injects the warning block from
``coulson.system_prompt_blocks.real_world_side_effects_warning`` into
``task["description"]`` so the model uses ``vendor_call`` rather than
fabricating API responses. Closes G9 in the v2 doc.

## Telegram surface

| Command | Action |
|---|---|
| ``/actions`` | List pending across all missions (or one) with inline cards |
| ``/action_done <id> [json]`` | Mark resolved, with optional structured payload |
| ``/missions`` | Active missions; appends ``· ⚠ N action(s) pending`` per mission with open actions |
| ``/mission <id>`` | Detail view; appends ``*Pending founder_actions: N*`` section with titles |

## System-wide actions

Some actions are not tied to a specific mission (credential rotation,
template staleness). Convention: ``mission_id=0`` is the system
sentinel. The Telegram surfaces render these with ``m=0`` and they
participate in the same lifecycle machinery.

## Cron-driven emitters

| Executor | Cadence | Kind | Source |
|---|---|---|---|
| ``compliance_template_staleness`` (T4D) | weekly | ``legal_counsel`` | template ``.meta.json`` > 180d |
| ``credential_rotation_reminder`` (T7A) | weekly | ``credential_paste`` | credentials expiring < 14d OR never rotated > 90d |
| ``stripe_dispute_check`` (T5D) | weekly | ``legal_counsel`` | new Stripe disputes |

All cron emitters are **idempotent**: they probe for an existing
pending/in_progress action with the same deterministic title before
creating a new one. Once the founder resolves the prior action, the
next scan re-emits.
