# vendor_call — architecture note

The bridge between the agent runtime and the IntegrationRegistry.
Exposed as **both** a mechanical post-hook (T3A) and an LLM tool
(T3B), so workflow steps and ReAct agents can both reach the
real world without duplicating retry / SSRF / auth code.

Shipped in Z6 T3A/T3B (2026-05-11).

## Two entry points, one backend

```
              ┌─────────────────────────────────────┐
              │   IntegrationRegistry               │
              │   (src/integrations/registry.py)    │
              │                                     │
              │   adapter.execute(action, params)   │
              └────────────────▲────────────────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
┌───────┴──────────┐                        ┌─────────┴────────┐
│ Mechanical post- │                        │ LLM-callable     │
│ hook (T3A)       │                        │ tool (T3B)       │
│ packages/        │                        │ src/tools/       │
│ mr_roboto/.../   │                        │ vendor_call.py   │
│ vendor_call.py   │                        │                  │
└──────────────────┘                        └──────────────────┘
```

Both paths share:

* SSRF-hardened HTTP layer (``HttpIntegration``)
* Per-vendor auth injection
* Retry + backoff
* Credential audit context (T2C)

## Mechanical post-hook (T3A)

Workflow step declares::

    {
      "id": "13.1",
      "needs_real_tools": true,
      "real_tool_kind": "vercel",
      "post_hook": {
        "kind": "vendor_call",
        "service": "vercel",
        "action": "deploy",
        "params_from_artifact": "production_compute_plan"
      }
    }

Mr. Roboto reads ``post_hook``, resolves params from the named artifact,
loads the adapter, executes. Failures emit a
``founder_action(kind='generic')`` with the error detail.

## LLM tool (T3B)

Agent calls::

    {"action": "tool_call",
     "tool": "vendor_call",
     "args": {
       "service": "vercel",
       "action": "list_projects",
       "params": {},
       "cost_estimate_usd": 0
     }}

Tool returns a JSON string with status/result.

### Per-agent allowlist

``src/tools/vendor_call.py::AGENT_ALLOWLIST`` declares which services
each agent type can touch. Default-deny — unlisted agents get
``"agent_not_allowed"`` back.

| Agent | Allowed services |
|---|---|
| ``executor`` | vercel, railway, supabase, cloudflare |
| ``implementer`` | stripe, sendgrid |
| ``reviewer`` | sentry |
| ``researcher``, ``coder``, ``planner``, shopping agents | (none) |

Tighten or loosen by editing the dict — every entry is a credential-
leakage / spend / blast-radius decision.

### Cost cap

``MAX_TOOL_CALL_COST_USD`` env (default ``5.0``). Calls with
``cost_estimate_usd`` over the cap return
``"cost_cap_exceeded"`` without hitting HTTP. Coarse safety net; the
mechanical path enforces mission-level budgets in addition.

## Audit log (T2C wired in T7D)

Every vendor call wraps execution in an audit context::

    async with audit_context(mission_id=…, task_id=…, agent=…):
        creds = await get_credential(service)        # writes credential_access_log
        result = await adapter.execute(action, …)

Because adapters pull credentials via ``get_credential()``, the
``credential_access_log`` row captures the right mission/task/agent
context for every vendor_call invocation. No separate
``vendor_call_audit_log`` table is needed.

Inspect via ``/credential log <service>`` (T2C).

## Allowlist matrix at a glance

Source of truth: ``src/tools/vendor_call.py::AGENT_ALLOWLIST``. Wave 1
configs live under ``src/integrations/configs/``: stripe, sendgrid,
cloudflare, sentry, supabase, vercel, railway, github,
apple_appstore, google_play.

## Cross-references

* Credentials live in ``src/security/credential_store.py`` (Fernet
  envelope, T2A-T2E).
* Per-vendor schemas at ``credential_schemas/<service>.json``
  validate paste payloads (T2B).
* ``real_tool_kind`` may be pipe-separated
  (``"vercel|railway|supabase"``); ``src/integrations/resolver.py``
  picks the first kind with adapter+creds (T3D).
* The G9 detect-and-bail in coulson (T7C) catches LLM agents that
  forget to use this tool when ``needs_real_tools=true``.
