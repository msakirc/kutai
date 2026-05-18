"""Z6 T6C — adapter helpers for non-static auth flows.

Some vendor APIs need a credential transform before each call (JWT mint,
OAuth token exchange). Each helper here takes the stored credential dict
and returns a short-lived bearer token. ``HttpIntegration`` calls them
from its ``execute()`` based on ``auth_type``.
"""
