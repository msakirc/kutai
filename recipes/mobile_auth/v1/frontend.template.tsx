/**
 * Mobile auth UI — Expo / React Native scaffold (Expo Router).
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:APP_NAME=MyApp
 *   // RECIPE_PARAM:API_BASE_ENV=EXPO_PUBLIC_API_BASE
 *   // RECIPE_PARAM:SECURE_STORE_TOKEN_KEY=session_jwt
 *   // RECIPE_PARAM:GOOGLE_CLIENT_ID_ENV=GOOGLE_OAUTH_CLIENT_ID
 *   // RECIPE_PARAM:AUTH_SCHEME=myapp
 *
 * Exports:
 *   AuthProvider / useAuth   — session context backed by expo-secure-store
 *   SignInScreen             — Apple + Google + email/password screen
 *
 * IMPORTANT — this is React Native, NOT the DOM. There is no <div>, no
 * <form>, no localStorage. Components are <View>/<Text>/<TextInput>/
 * <Pressable>. The session token lives in expo-secure-store (keychain on
 * iOS, keystore on Android) — never AsyncStorage, which is plain-text.
 *
 * In the instantiated project this single file is split:
 *   context/auth.tsx   — AuthProvider / useAuth
 *   app/sign-in.tsx    — SignInScreen (an expo-router route)
 *   app/_layout.tsx    — wraps the Stack in <AuthProvider>
 */
import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import * as SecureStore from "expo-secure-store";
import * as AppleAuthentication from "expo-apple-authentication";
import * as AuthSession from "expo-auth-session";
import * as Crypto from "expo-crypto";
import * as WebBrowser from "expo-web-browser";
import { useRouter } from "expo-router";

// expo-auth-session needs this to close the in-app browser after redirect.
WebBrowser.maybeCompleteAuthSession();

// RECIPE_PARAM:API_BASE_ENV=EXPO_PUBLIC_API_BASE
// EXPO_PUBLIC_-prefixed env vars are inlined into the JS bundle at build time.
const API_BASE = process.env.EXPO_PUBLIC_API_BASE ?? "http://localhost:8000";
// RECIPE_PARAM:GOOGLE_CLIENT_ID_ENV=GOOGLE_OAUTH_CLIENT_ID
const GOOGLE_CLIENT_ID = process.env.EXPO_PUBLIC_GOOGLE_CLIENT_ID ?? "";
// RECIPE_PARAM:SECURE_STORE_TOKEN_KEY=session_jwt
// expo-secure-store rejects values larger than ~2KB — store only the JWTs.
const ACCESS_KEY = "session_jwt";
const REFRESH_KEY = "session_refresh";
// RECIPE_PARAM:AUTH_SCHEME=myapp
// MUST match the "scheme" field in app.json or the OAuth redirect drops.
const APP_SCHEME = "myapp";
// RECIPE_PARAM:APP_NAME=MyApp
const APP_NAME = "MyApp";

// ---------------------------------------------------------------------------
// Session type
// ---------------------------------------------------------------------------

interface Session {
  accessToken: string;
  refreshToken: string;
  userId: number;
}

interface AuthContextValue {
  session: Session | null;
  loading: boolean;
  signInWithEmail: (email: string, password: string, register: boolean) => Promise<void>;
  signInWithApple: () => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

// ---------------------------------------------------------------------------
// Secure-store helpers
// ---------------------------------------------------------------------------

async function persistSession(s: Session): Promise<void> {
  // Two small writes — each value is a JWT, comfortably under the 2KB limit.
  await SecureStore.setItemAsync(ACCESS_KEY, s.accessToken);
  await SecureStore.setItemAsync(REFRESH_KEY, s.refreshToken);
}

async function clearSession(): Promise<void> {
  await SecureStore.deleteItemAsync(ACCESS_KEY);
  await SecureStore.deleteItemAsync(REFRESH_KEY);
}

async function readPersistedTokens(): Promise<{ access: string; refresh: string } | null> {
  const access = await SecureStore.getItemAsync(ACCESS_KEY);
  const refresh = await SecureStore.getItemAsync(REFRESH_KEY);
  if (!access || !refresh) return null;
  return { access, refresh };
}

// ---------------------------------------------------------------------------
// Backend calls
// ---------------------------------------------------------------------------

interface TokenResponse {
  access_token: string;
  refresh_token: string;
  user_id: number;
  expires_in: number;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

function toSession(t: TokenResponse): Session {
  return { accessToken: t.access_token, refreshToken: t.refresh_token, userId: t.user_id };
}

// ---------------------------------------------------------------------------
// AuthProvider
// ---------------------------------------------------------------------------

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  // On cold start: read tokens from secure-store and refresh the access token.
  useEffect(() => {
    (async () => {
      try {
        const tokens = await readPersistedTokens();
        if (tokens) {
          const fresh = await postJson<TokenResponse>("/auth/refresh", {
            refresh_token: tokens.refresh,
          });
          const s = toSession(fresh);
          await persistSession(s);
          setSession(s);
        }
      } catch {
        // Refresh token expired/revoked — drop it and show the sign-in screen.
        await clearSession();
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const adopt = useCallback(async (t: TokenResponse) => {
    const s = toSession(t);
    await persistSession(s);
    setSession(s);
  }, []);

  const signInWithEmail = useCallback(
    async (email: string, password: string, register: boolean) => {
      const path = register ? "/auth/register" : "/auth/login";
      await adopt(await postJson<TokenResponse>(path, { email, password }));
    },
    [adopt],
  );

  const signInWithApple = useCallback(async () => {
    // Generate a nonce, hash it with SHA-256, and pass the HASH to Apple.
    // The backend re-hashes the raw nonce and compares — binds the token to
    // this attempt and blocks replay.
    const rawNonce = Crypto.randomUUID();
    const hashedNonce = await Crypto.digestStringAsync(
      Crypto.CryptoDigestAlgorithm.SHA256,
      rawNonce,
    );
    const credential = await AppleAuthentication.signInAsync({
      requestedScopes: [
        AppleAuthentication.AppleAuthenticationScope.FULL_NAME,
        AppleAuthentication.AppleAuthenticationScope.EMAIL,
      ],
      nonce: hashedNonce,
    });
    if (!credential.identityToken) {
      throw new Error("Apple did not return an identity token");
    }
    // fullName + email are populated ONLY on the first-ever sign-in.
    const fullName = credential.fullName
      ? [credential.fullName.givenName, credential.fullName.familyName]
          .filter(Boolean)
          .join(" ")
      : null;
    await adopt(
      await postJson<TokenResponse>("/auth/apple", {
        identity_token: credential.identityToken,
        nonce: rawNonce,
        full_name: fullName,
        email: credential.email ?? null,
      }),
    );
  }, [adopt]);

  const signInWithGoogle = useCallback(async () => {
    // Discovery doc + PKCE flow via expo-auth-session.
    const discovery = {
      authorizationEndpoint: "https://accounts.google.com/o/oauth2/v2/auth",
      tokenEndpoint: "https://oauth2.googleapis.com/token",
    };
    // The redirect URI MUST use the app.json "scheme" — Google rejects a
    // mismatch with redirect_uri_mismatch.
    const redirectUri = AuthSession.makeRedirectUri({ scheme: APP_SCHEME });
    const rawNonce = Crypto.randomUUID();
    const request = new AuthSession.AuthRequest({
      clientId: GOOGLE_CLIENT_ID,
      scopes: ["openid", "email", "profile"],
      redirectUri,
      responseType: AuthSession.ResponseType.IdToken,
      extraParams: { nonce: rawNonce },
    });
    const result = await request.promptAsync(discovery);
    if (result.type !== "success" || !result.params.id_token) {
      throw new Error("Google sign-in was cancelled or failed");
    }
    await adopt(
      await postJson<TokenResponse>("/auth/google", {
        id_token: result.params.id_token,
        nonce: rawNonce,
      }),
    );
  }, [adopt]);

  const signOut = useCallback(async () => {
    if (session) {
      // Best-effort server-side revoke — local state is cleared regardless.
      try {
        await fetch(`${API_BASE}/auth/logout`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${session.accessToken}`,
          },
          body: JSON.stringify({ refresh_token: session.refreshToken }),
        });
      } catch {
        // ignore network failure on logout
      }
    }
    await clearSession();
    setSession(null);
  }, [session]);

  const value: AuthContextValue = {
    session,
    loading,
    signInWithEmail,
    signInWithApple,
    signInWithGoogle,
    signOut,
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// ---------------------------------------------------------------------------
// SignInScreen — an expo-router route (app/sign-in.tsx)
// ---------------------------------------------------------------------------

export default function SignInScreen() {
  const { signInWithEmail, signInWithApple, signInWithGoogle, session, loading } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [appleAvailable, setAppleAvailable] = useState(false);

  // Sign in with Apple is iOS-only; the button is hidden elsewhere.
  useEffect(() => {
    if (Platform.OS === "ios") {
      AppleAuthentication.isAvailableAsync().then(setAppleAvailable);
    }
  }, []);

  // Already signed in → leave the auth route.
  useEffect(() => {
    if (session) router.replace("/");
  }, [session, router]);

  async function run(action: () => Promise<void>) {
    setBusy(true);
    try {
      await action();
    } catch (e) {
      Alert.alert("Sign-in failed", e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{register ? `Join ${APP_NAME}` : `Sign in to ${APP_NAME}`}</Text>

      <TextInput
        style={styles.input}
        placeholder="Email"
        autoCapitalize="none"
        autoComplete="email"
        keyboardType="email-address"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        secureTextEntry
        autoComplete={register ? "new-password" : "current-password"}
        value={password}
        onChangeText={setPassword}
      />

      <Pressable
        style={[styles.primaryButton, busy && styles.disabled]}
        disabled={busy}
        onPress={() => run(() => signInWithEmail(email, password, register))}
      >
        <Text style={styles.primaryButtonText}>
          {register ? "Create account" : "Sign in"}
        </Text>
      </Pressable>

      <Pressable onPress={() => setRegister((r) => !r)}>
        <Text style={styles.link}>
          {register ? "Have an account? Sign in" : "No account? Register"}
        </Text>
      </Pressable>

      <View style={styles.divider} />

      {appleAvailable && (
        <AppleAuthentication.AppleAuthenticationButton
          buttonType={AppleAuthentication.AppleAuthenticationButtonType.SIGN_IN}
          buttonStyle={AppleAuthentication.AppleAuthenticationButtonStyle.BLACK}
          cornerRadius={8}
          style={styles.appleButton}
          onPress={() => run(signInWithApple)}
        />
      )}

      <Pressable
        style={[styles.googleButton, busy && styles.disabled]}
        disabled={busy}
        onPress={() => run(signInWithGoogle)}
      >
        <Text style={styles.googleButtonText}>Continue with Google</Text>
      </Pressable>

      {busy && <ActivityIndicator style={styles.spinner} />}
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles — StyleSheet.create, not CSS. (NativeWind is an alternative.)
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", padding: 24, gap: 12 },
  centered: { flex: 1, justifyContent: "center", alignItems: "center" },
  title: { fontSize: 22, fontWeight: "600", marginBottom: 12, textAlign: "center" },
  input: {
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 15,
  },
  primaryButton: {
    backgroundColor: "#2563eb",
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: "center",
  },
  primaryButtonText: { color: "#ffffff", fontWeight: "600", fontSize: 15 },
  link: { color: "#2563eb", textAlign: "center", fontSize: 14 },
  divider: { height: 1, backgroundColor: "#e2e8f0", marginVertical: 8 },
  appleButton: { height: 44, width: "100%" },
  googleButton: {
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: "center",
  },
  googleButtonText: { color: "#1e293b", fontWeight: "600", fontSize: 15 },
  disabled: { opacity: 0.5 },
  spinner: { marginTop: 8 },
});
