# Auth Recipe v1 — Known Lessons

Pitfalls captured from prior implementations. Seeded into `mission_lessons` on recipe instantiation.

- **CORS preflight on /auth/\***: Add `CORSMiddleware` before the auth router in FastAPI. Missing this causes OPTIONS requests to return 404, breaking browsers before any login attempt.
- **bcrypt cost factor in tests**: Default cost=12 adds ~300ms per hash. In test fixtures, override to cost=4: `CryptContext(schemes=["bcrypt"], bcrypt__rounds=4)`. Forget this and the test suite becomes painfully slow.
- **JWT `exp` claim must be int, not ISO string**: Pydantic v2 will reject `datetime` objects serialized as ISO strings in JWT payloads. Always use `int(datetime.timestamp())`. The template does this correctly — don't "fix" it to `isoformat()`.
- **alembic autogenerate misses token_hash indexes**: The indexes on `password_reset_tokens.token_hash` and `email_verify_tokens.token_hash` are frequently omitted by alembic autogenerate. Always diff the generated migration against `001_users.sql.template` and add missing indexes manually.
- **httpOnly cookie path scope**: The refresh token cookie is scoped to `path="/auth/refresh"`. Browsers will NOT send it to other endpoints. Widening to `path="/"` is a security regression — it exposes the refresh token to every request including static file fetches.
- **Anti-enumeration on /password/reset/request**: The endpoint always returns 200 regardless of whether the email exists. Never change this to a 404 — it leaks the user list.
- **Single-use token enforcement**: `password_reset_tokens.used` and `email_verify_tokens.used` must be checked AND set atomically. Use `UPDATE ... SET used=1 WHERE id=? AND used=0` + check rows_affected=1 in Postgres; SQLite's last_insert_rowid pattern works too. The stub implementation uses a dict flag — wire a real atomic update in production.
- **Refresh token is stored hashed, never raw**: Only the SHA-256 hash of the refresh JWT is stored in `auth_sessions`. If you accidentally store the raw token, session table leaks become a full account takeover vector.
- **Session GC is not automatic**: Expired + revoked rows accumulate in `auth_sessions`. Add a nightly cron or a beckman scheduled task: `DELETE FROM auth_sessions WHERE expires_at < datetime('now') OR revoked_at IS NOT NULL`.
- **SQLite datetime comparison uses space separator**: `strftime('%Y-%m-%d %H:%M:%S', 'now')` produces space-separated timestamps. Never store `datetime.isoformat()` (which uses `T`) in columns that are compared against SQLite's `datetime('now')` — string comparison fails silently.
- **`credentials: "include"` on client fetch**: The Next.js frontend must pass this flag or the browser will not attach the httpOnly refresh cookie to the `/auth/refresh` request. Cross-origin setups also need `allow_credentials=True` in CORSMiddleware.
- **Password min-length validation in two places**: Both the Pydantic model (`field_validator`) and the frontend form (`minLength={8}`) enforce 8-char minimum. Keep them in sync — a mismatch causes confusing UX where the UI accepts a password the API rejects.
- **`secrets.token_urlsafe` vs `uuid4` for reset tokens**: `secrets.token_urlsafe(32)` produces 43 chars of URL-safe base64 with 256 bits of entropy. `uuid4()` only provides 122 bits. Always use `secrets.token_urlsafe` for security tokens.
