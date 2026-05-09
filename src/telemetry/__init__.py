"""Cross-cutting telemetry helpers (B10+ rework metric, future signals).

Lives outside packages/ on purpose — telemetry composes from beckman,
coulson, workflow_engine, and telegram_bot. Putting it inside any one
package would create circular-import risk.
"""
