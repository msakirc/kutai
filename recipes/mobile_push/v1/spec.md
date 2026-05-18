# mobile_push Recipe v1 — Specification

## Scope

Expo hosted push notifications for an Expo / React Native app with a FastAPI
backend. Ships end-to-end: the `expo-notifications` permission flow, Expo
push-token acquisition, an Android notification channel, a foreground
notification handler, token sync to the backend, and a FastAPI send stub
that POSTs through the Expo push service.

## When to Pick This Recipe

**Use** `mobile_push/v1` when:
- The app is built with Expo (managed or prebuild workflow).
- Stack is `fastapi+sqlite+expo`, `fastapi+postgres+expo`, or `expo`.
- You want Expo *hosted* push (Expo fans out to APNs + FCM) — not a
  self-managed APNs/FCM integration.

**Do not use** when:
- The app is bare React Native without Expo modules.
- You need provider-direct delivery receipts at APNs/FCM granularity.

## What It Generates

| File | Role |
|------|------|
| `client.template.ts` | `registerForPushNotificationsAsync`, `syncPushToken`, `usePushNotifications` hook |
| `backend.template.py` | FastAPI router: `/push/register`, `/push/send`, `send_push` |
| `flows/push_smoke.flow.yaml` | Maestro smoke flow (sign in → onboard → grant permission → sign out) |
| `tests/backend_smoke.template.py` | Token-validation + register-route smoke tests |

## Mobile QA

The `flows/push_smoke.flow.yaml` Maestro flow is the input to the
`mobile_smoke` post-hook: it drives the canonical sign in → onboard → core
action (grant notification permission) → sign out path against a running
build. Declare `post_hooks: ["mobile_smoke"]` plus
`maestro_flows: ["flows/push_smoke.flow.yaml"]` on the step that builds the
mobile app to gate it on the flow.

## Parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `EXPO_PUSH_API` | `https://exp.host/--/api/v2/push/send` | Expo hosted push endpoint |
| `ANDROID_CHANNEL_ID` | `default` | Android notification channel id |
| `NOTIFICATION_PERMISSION_RATIONALE` | (see yaml) | Copy shown before the system prompt |
| `PUSH_TOKEN_ENDPOINT` | `/push/register` | Backend route for token sync |
| `APP_NAME` | `MyApp` | Display name (channel name) |

## Out of Scope

- Rich/interactive notification categories and action buttons.
- Scheduled local notifications.
- A real `push_tokens` table — the backend store is an in-memory stub.
- Receipt-polling cron for delayed delivery errors.
