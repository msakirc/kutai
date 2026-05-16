# Mobile Auth Recipe v1 — Known Lessons

Pitfalls captured from prior Expo-auth implementations. Seeded into
`mission_lessons` (domain `mobile_auth`) on recipe instantiation.

- **`expo-secure-store` has a ~2KB value limit**: A value larger than roughly 2KB throws on iOS (`SecItemAdd` failure) and silently truncates on some Android keystores. Store the access and refresh JWTs as two separate keys — never pack a JSON blob with user profile data into one secure-store entry.

- **Never use AsyncStorage for tokens**: `AsyncStorage` is unencrypted plain text on disk. A rooted/jailbroken device or a backup extraction leaks every token. Session tokens go in `expo-secure-store` only (iOS keychain / Android keystore).

- **Sign in with Apple needs a physical device or a signed-in simulator**: `AppleAuthentication.signInAsync` does nothing on Android, does nothing on Expo Web, and fails on an iOS Simulator that is not signed into an Apple ID. Gate the button with `Platform.OS === 'ios'` AND `await AppleAuthentication.isAvailableAsync()`.

- **Apple's nonce is mandatory and must be hashed**: Generate a random nonce, pass the **SHA-256 hash** to `signInAsync({ nonce })`, and send the **raw** nonce to the backend. Apple embeds the hash in the identity token; the backend re-hashes the raw value and compares. Skip this and the token is replayable — a stolen identity token works forever.

- **Apple only returns the email and full name on the FIRST sign-in**: For a given Apple ID + app pair, `credential.email` and `credential.fullName` are populated exactly once. Every subsequent sign-in returns `null` for both. Persist them to the `users` row on first contact — you cannot ask Apple again.

- **Apple private-relay emails and email-withheld accounts**: Users may hide their email (Apple issues a `@privaterelay.appleid.com` relay address) or, on re-auth, you simply get no email at all. The `users.email` column must be nullable and email uniqueness must be a *partial* index scoped to password accounts only.

- **Google `redirect_uri_mismatch`**: The redirect URI from `AuthSession.makeRedirectUri({ scheme })` is derived from the app's URL scheme. That `scheme` must exactly match the `"scheme"` field in `app.json` AND be registered in the Google Cloud console OAuth client. A mismatch (or a missing `scheme` in `app.json`) fails the sign-in with `redirect_uri_mismatch`.

- **Google ID-token audience check is not optional**: Verify that the token's `aud` claim equals *your* OAuth client id. A signature-valid Google ID token minted for a different app is still cryptographically valid — accepting it without the `aud` check lets any Google app impersonate your users.

- **`WebBrowser.maybeCompleteAuthSession()` must run at module load**: Without this call the in-app browser opened by `expo-auth-session` never closes after the OAuth redirect, leaving the user staring at a blank web view. Call it once at the top of the module, not inside a component.

- **`EXPO_PUBLIC_` prefix is required for client env vars**: Only env vars prefixed `EXPO_PUBLIC_` are inlined into the JS bundle and readable at runtime via `process.env`. `API_BASE` or `GOOGLE_CLIENT_ID` without the prefix is `undefined` in the app — sign-in then silently posts to `undefined/auth/...`.

- **Provider JWKS keys rotate — cache, but refetch on unknown `kid`**: Apple and Google rotate their RS256 signing keys roughly daily. Cache the JWKS (~1h TTL) but on a token whose header `kid` is not in the cache, drop the cache and refetch once before rejecting. A hard-cached JWKS starts failing all logins the day a key rotates.

- **Store only the hash of the refresh token**: `auth_sessions` stores the SHA-256 hash of the refresh token, never the raw JWT. A leaked session table with raw refresh tokens is a full account-takeover vector. On `/auth/refresh`, revoke the presented row and issue a new pair (rotation) so a captured-and-reused token is detected.

- **`identityToken` can be `null`**: If the user backs out of the Apple sheet, `signInAsync` may still resolve with a credential whose `identityToken` is `null`. Null-check before posting to the backend or you send `"null"` as the token and get a confusing 401.

- **JWT `exp` must be an int, not an ISO string**: The app's own JWT `exp` claim must be `int(datetime.timestamp())`. python-jose and most clients reject ISO-string expiries. The backend template does this correctly — do not "fix" it to `isoformat()`.
