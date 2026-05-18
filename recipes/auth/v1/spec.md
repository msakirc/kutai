# Auth Recipe v1 — Specification

## Scope

Email-password registration and session management for standard SaaS applications. Ships
complete end-to-end: registration, login, logout, access-token refresh, password reset via
emailed token, email verification, per-user session ledger, and brute-force protection on
the login endpoint.

## When to Pick This Recipe

**Use** `auth/v1` when:
- Your app uses email + password as the primary login method
- Stack is `fastapi+postgres+nextjs` or `fastapi+sqlite+nextjs`
- You want stateless access tokens (JWT) with a persistent refresh-token ledger
- B2B or B2C SaaS with self-serve registration

**Do NOT use** `auth/v1` when:
- Login is OAuth-only (Google/GitHub/etc) → use `oauth/v1` (TBD)
- Enterprise SSO / SAML / OIDC identity provider → out of scope for v1
- You need cookie-only sessions without JWTs (server-side session variant) → TBD
- Multi-tenant org-level auth (team invites, org roles) → use `org_auth/v1` (TBD)
- Mobile app where httpOnly cookies are not viable → set `token_transport: header_only` param

## Authentication Flow

```
Register:   POST /auth/register → creates user + sends verify email
Verify:     POST /auth/email/verify → activates account
Login:      POST /auth/login → access_token (15 min, Authorization header) +
                               refresh_token (7 days, httpOnly cookie)
Refresh:    POST /auth/refresh → rotates access_token (reads cookie)
Logout:     POST /auth/logout → revokes session row, clears cookie
PW Reset:   POST /auth/password/reset/request → mails token
            POST /auth/password/reset/confirm → consumes token, sets new password
```

## Security Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Access token TTL | 15 minutes | Short-lived; never stored in DB |
| Refresh token TTL | 7 days | Stored as bcrypt hash in `auth_sessions` |
| Password hash | bcrypt cost 12 | See trade-offs |
| Reset token TTL | 1 hour | Stored as SHA-256 hash |
| Email verify token TTL | 24 hours | Same table pattern |
| Login rate limit | 10 req/min per IP | SlowAPI / custom middleware |
| JWT algorithm | HS256 | See `RECIPE_PARAM:JWT_ALGO` |

## Database Tables

Four tables in `001_users.sql.template`:

1. **users** — identity ledger. `email_verified` flag gates access in some configurations.
2. **password_reset_tokens** — single-use; marked `used=1` on consumption.
3. **email_verify_tokens** — same single-use pattern.
4. **auth_sessions** — refresh token ledger. One row per active device/browser session.
   Revoked sessions are soft-deleted (`revoked_at` set). Expired + revoked rows are
   garbage-collected by a background job (add a cron step or call
   `DELETE FROM auth_sessions WHERE expires_at < datetime('now')`).

## Trade-offs

### JWT in cookie vs Authorization header

Default: **access token in `Authorization: Bearer` header + refresh token in httpOnly
`Set-Cookie`**. Rationale: access tokens are short-lived (15 min) and fine to expose to
JS for API calls; refresh tokens are long-lived and must be protected from XSS → httpOnly
cookie. Refresh endpoint reads the cookie server-side.

Alternative: put both tokens in cookies (fully opaque to JS). This hardens XSS but
requires CSRF mitigations (`SameSite=Strict` or double-submit cookie). To switch: set
`RECIPE_PARAM:TOKEN_TRANSPORT=cookie_only` and uncomment the CSRF middleware block in the
template. Not the default because it complicates mobile clients.

### bcrypt vs argon2id

Default: **bcrypt (cost 12)**. Rationale: `passlib[bcrypt]` is a single-package dependency
with wide deployment history; bcrypt is well-audited and universally supported in hosting
environments. Argon2id is stronger against GPU attacks but requires `argon2-cffi` and
careful memory/parallelism tuning (defaults vary by library version).

**Swap path**: replace `CryptContext(schemes=["bcrypt"])` with
`CryptContext(schemes=["argon2"])` and add `argon2-cffi` to requirements. Argon2id
parameters: `time_cost=3, memory_cost=65536, parallelism=4` is a reasonable starting
point. All other code (hash_password / verify_password) is unchanged because passlib
abstracts the scheme.

### Stateless access tokens

Access tokens are NOT stored in the database. This means they cannot be individually
revoked before expiry (15 min window). If you need instant revocation (compromised account,
admin force-logout), either shorten the TTL further or add a redis-backed blocklist. The
session ledger (`auth_sessions`) handles refresh-token revocation, which is the safer
path.

## RECIPE_PARAM Markers

The template engine substitutes these markers during mission instantiation:

| Marker | Default | Description |
|--------|---------|-------------|
| `JWT_ALGO` | `HS256` | JWT signing algorithm |
| `JWT_SECRET_ENV` | `AUTH_JWT_SECRET` | Name of env var holding the secret |
| `JWT_TTL_MIN` | `15` | Access token lifetime in minutes |
| `REFRESH_TTL_DAYS` | `7` | Refresh token lifetime in days |
| `BCRYPT_COST` | `12` | bcrypt cost factor (use 4 in tests) |
| `LOGIN_RATE_LIMIT` | `10/minute` | SlowAPI rate string for /login |
| `TOKEN_TRANSPORT` | `split` | `split` (default), `cookie_only`, `header_only` |
| `REDIRECT_AFTER_LOGIN` | `/dashboard` | Next.js: where to redirect after successful login |
| `APP_NAME` | `MyApp` | Used in email subject lines |

## Dependencies

**Backend (Python)**:
- `fastapi`
- `passlib[bcrypt]`
- `python-jose[cryptography]` — JWT encode/decode
- `aiosqlite` (SQLite) or `asyncpg` (Postgres) — one or the other per stack
- `slowapi` — rate limiting on /login (optional, swap with custom middleware)

**Frontend (npm)**:
- `zod` — form validation schemas
- `@tanstack/react-query` — optional, for client-side session invalidation

## Known Non-Goals (v1)

- No social login providers
- No admin "impersonate user" flow
- No hardware 2FA (TOTP / WebAuthn) — planned for `auth/v2`
- No account merging (social ↔ password)
- No audit log beyond the session ledger
