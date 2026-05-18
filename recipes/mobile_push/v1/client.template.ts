/**
 * Expo push notifications — client registration shim.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:ANDROID_CHANNEL_ID=default
 *   // RECIPE_PARAM:NOTIFICATION_PERMISSION_RATIONALE=Enable notifications to get order and activity updates.
 *   // RECIPE_PARAM:PUSH_TOKEN_ENDPOINT=/push/register
 *   // RECIPE_PARAM:APP_NAME=MyApp
 *
 * The template engine replaces `KEY` tokens before this file is written
 * into the mission workspace. The RECIPE_PARAM comment markers are left
 * intact so the generated file stays self-documenting.
 *
 * Wire `registerForPushNotificationsAsync` at app startup (e.g. in the
 * root layout effect) and POST the returned Expo push token to the
 * backend so the server can target the device later.
 */
import { useEffect, useRef, useState } from "react";
import { Platform } from "react-native";
import * as Notifications from "expo-notifications";
import * as Device from "expo-device";
import Constants from "expo-constants";

const ANDROID_CHANNEL_ID = "<<ANDROID_CHANNEL_ID>>";
const PERMISSION_RATIONALE = "<<NOTIFICATION_PERMISSION_RATIONALE>>";
const PUSH_TOKEN_ENDPOINT = "<<PUSH_TOKEN_ENDPOINT>>";
const APP_NAME = "<<APP_NAME>>";

// Foreground presentation: Expo SDK 50+ — banner + list, no legacy
// shouldShowAlert. Without a handler, foreground notifications are silent.
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldPlaySound: true,
    shouldSetBadge: false,
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

export type PushRegistration = {
  token: string | null;
  error: string | null;
};

/**
 * Request permission, fetch the Expo push token, and (Android) create the
 * notification channel. Returns {token, error}; token is null when the
 * user declines or the call runs on a simulator.
 */
export async function registerForPushNotificationsAsync(): Promise<PushRegistration> {
  if (!Device.isDevice) {
    // Expo push tokens require a physical device — simulators have no
    // APNs/FCM registration. Not an error, just unavailable.
    return { token: null, error: "push tokens require a physical device" };
  }

  if (Platform.OS === "android") {
    await Notifications.setNotificationChannelAsync(ANDROID_CHANNEL_ID, {
      name: APP_NAME,
      importance: Notifications.AndroidImportance.DEFAULT,
    });
  }

  const existing = await Notifications.getPermissionsAsync();
  let status = existing.status;
  if (status !== "granted") {
    // PERMISSION_RATIONALE should be surfaced in UI BEFORE this call —
    // iOS shows the system prompt only once, ever.
    const req = await Notifications.requestPermissionsAsync();
    status = req.status;
  }
  if (status !== "granted") {
    return { token: null, error: "notification permission not granted" };
  }

  // projectId is required by getExpoPushTokenAsync on SDK 49+.
  const projectId =
    Constants.expoConfig?.extra?.eas?.projectId ??
    Constants.easConfig?.projectId;
  if (!projectId) {
    return { token: null, error: "missing EAS projectId in app config" };
  }

  const tokenResponse = await Notifications.getExpoPushTokenAsync({ projectId });
  return { token: tokenResponse.data, error: null };
}

/** POST the Expo push token to the backend so it can target this device. */
export async function syncPushToken(
  apiBase: string,
  token: string,
  authHeader?: Record<string, string>,
): Promise<boolean> {
  const res = await fetch(`${apiBase}${PUSH_TOKEN_ENDPOINT}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(authHeader ?? {}) },
    body: JSON.stringify({ expo_push_token: token, platform: Platform.OS }),
  });
  return res.ok;
}

/** Hook: registers on mount, syncs the token, exposes the latest notification. */
export function usePushNotifications(apiBase: string) {
  const [registration, setRegistration] = useState<PushRegistration>({
    token: null,
    error: null,
  });
  const lastNotification = useRef<Notifications.Notification | null>(null);

  useEffect(() => {
    let cancelled = false;
    registerForPushNotificationsAsync().then((reg) => {
      if (cancelled) return;
      setRegistration(reg);
      if (reg.token) {
        void syncPushToken(apiBase, reg.token);
      }
    });

    const receivedSub = Notifications.addNotificationReceivedListener((n) => {
      lastNotification.current = n;
    });
    return () => {
      cancelled = true;
      receivedSub.remove();
    };
  }, [apiBase]);

  return { registration, lastNotification };
}

export { PERMISSION_RATIONALE };
