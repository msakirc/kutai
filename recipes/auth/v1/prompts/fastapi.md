# FastAPI — auth/v1 instantiation notes

When wiring this recipe in a FastAPI project:

- **Route order**: add `CORSMiddleware` BEFORE including the auth router or OPTIONS pre-flights will return 404.
- **JWT secret**: `AUTH_JWT_SECRET` (or the `RECIPE_PARAM:JWT_SECRET_ENV` value) MUST come from an env var, never from the template literal. Raise on missing at startup — the template already does this.
- **Alembic index gap**: `alembic autogenerate` may miss `idx_prt_token_hash` and `idx_evt_token_hash` on the token tables. Always check the generated migration against `001_users.sql.template` after `alembic revision --autogenerate`.
- **Rate limiting**: wire `slowapi.Limiter` to the `/auth/login` route. The `RECIPE_PARAM:LOGIN_RATE_LIMIT` value (`10/minute` default) maps directly to the `@limiter.limit(...)` decorator.
- **bcrypt cost in tests**: override `auth.pwd_context` with `CryptContext(schemes=["bcrypt"], bcrypt__rounds=4)` in conftest — default cost 12 adds ~300ms per test.
- **Cookie path scope**: the refresh cookie is set with `path="/auth/refresh"` so browsers only send it to that exact endpoint. Do not widen to `/` unless you understand the CSRF surface.
