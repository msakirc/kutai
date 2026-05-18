# FastAPI — mobile_auth/v1 instantiation notes

When wiring the backend half of this recipe:

## Provider ID-token verification

- **Apple** (`/auth/apple`): verify the identity token against the JWKS at
  `https://appleid.apple.com/auth/keys`. Required checks: signature, `iss ==
  https://appleid.apple.com`, `aud ==` the app **bundle id**, `exp` in the
  future, and `nonce` equals the SHA-256 of the raw nonce the client sent.
- **Google** (`/auth/google`): verify against
  `https://www.googleapis.com/oauth2/v3/certs`. The `aud` claim **must equal
  the OAuth client id** registered for this app — a signature-valid token
  minted for a different client must be rejected. Also reject when
  `email_verified` is explicitly `false`.
- **No cookies.** There is no browser — both `/auth/*` endpoints return the
  access + refresh tokens in the JSON body. Do not set `Set-Cookie`.

## JWKS caching

- Cache provider JWKS documents in-process (~1h TTL). Apple/Google rotate
  signing keys roughly daily, so on an **unknown `kid`** drop the cache and
  refetch **once** before failing — the template already does this.

## Token issuance

- The app's own JWT secret comes from `RECIPE_PARAM:JWT_SECRET_ENV`
  (`MOBILE_AUTH_JWT_SECRET`). Raise at startup if missing — the template does.
- Access token TTL `30m`, refresh `30d` — longer than the web `auth` recipe
  because re-prompting a mobile user is higher friction.
- **Refresh-token rotation**: `/auth/refresh` revokes the presented refresh
  row and issues a new pair. Store only the SHA-256 hash of the refresh token
  in `auth_sessions`, never the raw value.

## CORS / hosts

- Native apps are not subject to browser CORS, but Expo Web and the dev client
  hit the API from `http://localhost:*` — add `CORSMiddleware` if a web target
  is in scope. The Expo dev server also reaches the API over the LAN IP, so
  bind the server to `0.0.0.0`, not `127.0.0.1`, during development.

## Email/password path

- `/auth/register` and `/auth/login` reuse the bcrypt pattern from the web
  `auth` recipe (`RECIPE_PARAM:BCRYPT_COST`, override to `4` in test fixtures).
- A social account has `password_hash = NULL`; `/auth/login` must reject those
  rows rather than crash on a `None` hash — the template guards this.
