# Expo — mobile_push/v1 instantiation notes

When wiring this recipe into an Expo / React Native project:

- **Physical device required**: `getExpoPushTokenAsync` returns a token only
  on a real device. Simulators/emulators have no APNs/FCM registration —
  the client shim returns `{token:null, error:"...physical device"}` there,
  which is expected, not a bug.
- **`projectId` is mandatory**: SDK 49+ requires `projectId` passed to
  `getExpoPushTokenAsync`. It comes from `app.json` →
  `extra.eas.projectId`. Run `eas init` once so this is populated, or the
  token call throws.
- **Permission rationale before the prompt**: iOS shows the system
  notification prompt exactly once, ever. Surface
  `RECIPE_PARAM:NOTIFICATION_PERMISSION_RATIONALE` in your own UI *before*
  calling `requestPermissionsAsync` — if the user declines the system
  prompt you cannot re-ask.
- **Android channel first**: On Android 8+ a notification with no channel
  is silently dropped. `setNotificationChannelAsync` must run before the
  first notification — the shim does this in `registerForPushNotificationsAsync`.
- **Foreground handler**: without `setNotificationHandler`, notifications
  received while the app is foregrounded are silent. The shim sets a
  banner+list handler at module load.
- **Backend talks to Expo, not APNs/FCM**: the FastAPI stub POSTs to the
  Expo push service (`RECIPE_PARAM:EXPO_PUSH_API`). Expo fans out to APNs
  and FCM. Never wire APNs/FCM credentials into the backend for the hosted
  path.
- **Token storage is a stub**: `backend.template.py` keeps tokens in an
  in-memory dict. Replace with a `push_tokens` table keyed by user before
  shipping — see `lessons.md`.
- **Evict dead tokens**: an Expo receipt with `status:"error"` /
  `DeviceNotRegistered` means the token is dead. Delete it so you don't
  keep paying to push into the void.
