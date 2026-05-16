# Expo — mobile_auth/v1 instantiation notes

Recipe-specific guidance. The generic `expo` reminders (Expo Router, StyleSheet,
EAS Build, Platform guards) live in `STACK_BLOCKS` — this file does **not**
repeat them. It covers what is unique to mobile auth.

## Token storage

- Store the session JWTs in **`expo-secure-store`** (iOS keychain / Android
  keystore). **Never `AsyncStorage`** — it is unencrypted plain text.
- `expo-secure-store` rejects values larger than **~2KB**. A JWT is well under
  that, so store the access and refresh tokens as **two separate keys** —
  do not pack a JSON blob with user profile data into one value.
- `SecureStore.deleteItemAsync` on a missing key is a no-op — safe to call on
  sign-out without a guard.

## Sign in with Apple

- Requires `expo-apple-authentication` and a **physical iOS device** or a
  signed-in iOS Simulator — it does **not** work on Android or Expo Web.
  Gate the button with `Platform.OS === 'ios'` **and**
  `AppleAuthentication.isAvailableAsync()`.
- **Nonce is mandatory.** Generate a random nonce, SHA-256 it (`expo-crypto`),
  pass the **hash** to `signInAsync({ nonce })`, and send the **raw** nonce to
  the backend so it can re-hash and compare. Skipping this allows token replay.
- Apple returns `fullName` and `email` **only on the first-ever sign-in** for
  that Apple ID + app pair. Persist them immediately on first contact — they
  are gone on every subsequent sign-in.
- `identityToken` can be `null` if the user backs out — always null-check
  before posting to the backend.

## Sign in with Google

- Use `expo-auth-session` with `responseType: ResponseType.IdToken`.
- The **redirect URI** is built from the app's URL scheme:
  `AuthSession.makeRedirectUri({ scheme })`. That `scheme` **must exactly
  match** the `"scheme"` field in `app.json` or Google returns
  `redirect_uri_mismatch`. Also register the redirect URI in the Google Cloud
  console OAuth client.
- Call `WebBrowser.maybeCompleteAuthSession()` once at module load or the
  in-app browser will not close after the redirect.
- Pass a `nonce` in `extraParams` and forward it to the backend's `aud` check.

## Environment variables

- Client-side config must be prefixed `EXPO_PUBLIC_` to be inlined into the
  bundle (`EXPO_PUBLIC_API_BASE`, `EXPO_PUBLIC_GOOGLE_CLIENT_ID`). Anything
  unprefixed is **not** available at runtime in the app.
- The Google OAuth **client secret** is never used in the app — the ID-token
  flow needs only the public client id.

## Wiring into Expo Router

- Wrap the root `Stack` in `<AuthProvider>` inside `app/_layout.tsx`.
- `SignInScreen` is the route `app/sign-in.tsx`; redirect unauthenticated
  users there from a layout guard, and `router.replace('/')` once `session`
  is set.
