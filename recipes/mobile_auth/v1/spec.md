# Mobile Auth Recipe v1 — Specification

## Scope

Authentication for an **Expo / React Native** mobile app with a **FastAPI**
backend. Ships three sign-in methods end-to-end:

1. **Sign in with Apple** — `expo-apple-authentication`, nonce-bound, iOS-only.
2. **Sign in with Google** — `expo-auth-session` ID-token (PKCE) flow.
3. **Email + password** — classic bcrypt-hashed credentials.

All three converge on the same outcome: the backend verifies the proof
(provider ID token or password) and issues **the app's own JWT** — a 30-minute
access token plus a 30-day refresh token. The mobile client stores both in
**`expo-secure-store`** (iOS keychain / Android keystore).

## When to Pick This Recipe

**Use** `mobile_auth/v1` when:
- The frontend is **Expo** (`fastapi+sqlite+expo` or `fastapi+postgres+expo`).
- You want native Apple/Google sign-in alongside email/password.
- The session token must live in encrypted device storage, not a cookie.

**Do NOT use** `mobile_auth/v1` when:
- The frontend is a web app (Next.js/Vite) → use `auth/v1` (cookie-based).
- You need bare React Native without Expo → the `expo-*` modules will not
  resolve; out of scope for v1.
- You need enterprise SSO / SAML / OIDC-provider login → out of scope.
- You need account linking (same person via Apple *and* Google merged into
  one row) → v1 keeps them as separate users; planned for v2.

## Why a separate recipe from `auth/v1`

`auth/v1` puts the refresh token in an **httpOnly cookie** — there is no
browser cookie jar in a native app, so that mechanism does not exist here.
`mobile_auth` returns both tokens in the JSON body and the client owns
storage. The recipes **conflict** (`conflicts_with: [auth]`) — a mission picks
one or the other based on `target_platform`.

## Authentication Flows

```
Email register: POST /auth/register  -> {access_token, refresh_token, user_id, expires_in}
Email login:    POST /auth/login     -> same payload
Apple:          POST /auth/apple     {identity_token, nonce, full_name?, email?}
                  - verify token vs Apple JWKS, iss, aud=bundle id, nonce
Google:         POST /auth/google    {id_token, nonce?}
                  - verify token vs Google JWKS, iss, aud=OAuth client id
Refresh:        POST /auth/refresh   {refresh_token}  -> rotated pair
                  - old refresh row revoked, new pair issued
Logout:         POST /auth/logout    {refresh_token}  (Bearer access token)
Protected:      GET  /auth/me        (Bearer access token)
```

## Provider Token Verification

| Provider | JWKS endpoint | `iss` | `aud` must equal |
|----------|---------------|-------|------------------|
| Apple    | `appleid.apple.com/auth/keys` | `https://appleid.apple.com` | the app **bundle id** |
| Google   | `googleapis.com/oauth2/v3/certs` | `accounts.google.com` | the **OAuth client id** |

Both providers' tokens are RS256. The backend caches the JWKS in-process
(~1h) and refetches once on an unknown `kid` (handles daily key rotation).
Apple additionally requires the **nonce** to match the SHA-256 of the raw
nonce the client generated; Google's nonce is verified when present.

## Database Tables

Two tables in `001_mobile_users.sql.template`:

1. **users** — `provider ∈ {email, apple, google}`, `provider_subject` holds
   the provider `sub` claim, `password_hash` set only for email accounts.
   `UNIQUE(provider, provider_subject)` blocks duplicate social rows; a
   partial unique index enforces email uniqueness only for password accounts
   (Apple may withhold the email entirely).
2. **auth_sessions** — refresh-token ledger. Stores the **SHA-256 hash** of the
   refresh token, never the raw value. Refresh rotation revokes the old row.

## Security Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Access token TTL | 30 minutes | App JWT; never stored in DB |
| Refresh token TTL | 30 days | SHA-256 hash stored in `auth_sessions` |
| Password hash | bcrypt cost 12 | override to 4 in tests |
| JWT algorithm | HS256 | the app's own token; `RECIPE_PARAM:JWT_ALGO` |
| Provider JWKS cache | ~1 hour | refetch once on unknown `kid` |

## RECIPE_PARAM Markers

| Marker | Default | Description |
|--------|---------|-------------|
| `JWT_ALGO` | `HS256` | App JWT signing algorithm |
| `JWT_SECRET_ENV` | `MOBILE_AUTH_JWT_SECRET` | Env var holding the app JWT secret |
| `JWT_TTL_MIN` | `30` | Access token lifetime (minutes) |
| `REFRESH_TTL_DAYS` | `30` | Refresh token lifetime (days) |
| `BCRYPT_COST` | `12` | bcrypt cost factor (use 4 in tests) |
| `APPLE_CLIENT_ID_ENV` | `APPLE_BUNDLE_ID` | Env var with the Apple bundle id (expected `aud`) |
| `GOOGLE_CLIENT_ID_ENV` | `GOOGLE_OAUTH_CLIENT_ID` | Env var with the Google OAuth client id (expected `aud`) |
| `API_BASE_ENV` | `EXPO_PUBLIC_API_BASE` | Client env var pointing at the API |
| `SECURE_STORE_TOKEN_KEY` | `session_jwt` | `expo-secure-store` key for the access token |
| `AUTH_SCHEME` | `myapp` | URL scheme — must match `app.json` `scheme` |
| `APP_NAME` | `MyApp` | Display name in UI copy |
| `PK_TYPE` | `INTEGER PRIMARY KEY AUTOINCREMENT` | SQLite/Postgres primary-key spelling |

## Dependencies

**Backend (Python)**: `fastapi`, `python-jose[cryptography]` (JWT encode +
RS256 provider verification), `httpx` (JWKS fetch), `passlib[bcrypt]`,
`aiosqlite` (or `asyncpg` for Postgres).

**Frontend (npm / Expo)**: `expo`, `expo-router`, `expo-secure-store`,
`expo-auth-session`, `expo-apple-authentication`, `expo-crypto` (nonce
hashing), `expo-web-browser` (closes the OAuth in-app browser).

## Post-hooks

`imports_check`, `test_run`, `pattern_lint` — run after instantiation to
verify the templates import cleanly, the smoke tests pass, and no banned
patterns slipped in.

## Known Non-Goals (v1)

- No account linking across providers (Apple + Google + email = up to 3 rows).
- No biometric local re-auth (Face ID / Touch ID gate) — a `mobile_auth/v2`
  candidate via `expo-local-authentication`.
- No 2FA / TOTP / passkeys.
- No push-notification opt-in during onboarding (see the `mobile_push` recipe).
- No deep-link email-verification flow — social providers already vouch for
  the email; email/password accounts are usable immediately in v1.
