# Expo — mobile_deep_links/v1 instantiation notes

When wiring this recipe into an Expo / React Native project:

- **`scheme` is a single source of truth**: set `scheme` once in
  `app.json` / `app.config.ts` (the `app.config.template.ts` fragment
  does). The linking config reuses the same `RECIPE_PARAM:APP_SCHEME`
  value. A mismatch silently breaks custom-scheme links.
- **AASA must be served correctly**: host
  `apple-app-site-association` (NO `.json` extension on the served path)
  at `https://<domain>/.well-known/apple-app-site-association`, with
  `Content-Type: application/json`, over HTTPS, with NO redirect. Apple's
  CDN caches it — changes can take ~24h to propagate. The template ships
  the JSON body; serving it is a backend/hosting task.
- **`appID` format is `TEAMID.BUNDLEID`**: e.g. `ABCDE12345.com.example.app`.
  A wrong team id is the single most common universal-links failure.
- **Android needs `assetlinks.json` too**: host
  `https://<domain>/.well-known/assetlinks.json` with the app's package
  name and the SHA-256 signing-cert fingerprint. `autoVerify:true` in the
  intent filter makes Android verify it. The fingerprint comes from the
  *upload/release* key — debug builds use a different cert and won't verify.
- **`associatedDomains` value has no scheme/path**: it is `applinks:` +
  bare host (`applinks:links.example.com`). Adding `https://` or a path
  makes iOS reject it silently.
- **Custom-scheme links always work; verified links need the well-known
  files**: during development, test with `myapp://...` (no verification
  needed). Universal/app links (`https://...`) only resolve to the app
  after AASA/assetlinks are live AND the app has been (re)installed.
- **expo-router infers routes from `app/`**: the explicit `screens` map in
  `linking.template.ts` is the fallback for paths needing custom parsing —
  a pure file-based app may only need `prefixes`.
- **Test with `npx uri-scheme open`**: `npx uri-scheme open myapp://item/42
  --ios` (or `--android`) drives a deep link without a QR code. The Maestro
  smoke flow's `openLink` command does the same in CI.
