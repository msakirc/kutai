/**
 * Session state hook — minimal in-memory stub.
 *
 * `mobile_nav` is a NAVIGATION recipe; it only needs to KNOW whether a user
 * is signed in so the root-layout auth gate can redirect. Real credential
 * handling (Sign in with Apple/Google, token storage in `expo-secure-store`)
 * is the `mobile_auth` recipe's job — wire that in and replace this stub.
 *
 * `isReady` exists so the auth gate waits for the persisted session to load
 * before deciding to redirect: redirecting while `isReady === false` would
 * flash the sign-in screen on every cold start.
 */
import { createContext, useContext } from "react";

export interface Session {
  /** True when a session token is present. */
  isSignedIn: boolean;
  /** True once the persisted session has finished loading. */
  isReady: boolean;
  /** Persist a session token and flip `isSignedIn`. */
  signIn: (token: string) => Promise<void>;
  /** Clear the session token and flip `isSignedIn`. */
  signOut: () => Promise<void>;
}

const noop = async () => {};

/**
 * Default session — signed out, ready immediately.
 *
 * Replace the provider with a real one (SecureStore-backed) from the
 * `mobile_auth` recipe. The default here keeps `mobile_nav` runnable and
 * testable standalone.
 */
export const SessionContext = createContext<Session>({
  isSignedIn: false,
  isReady: true,
  signIn: noop,
  signOut: noop,
});

/** Read the current session. Defaults to signed-out when no provider wraps. */
export function useSession(): Session {
  return useContext(SessionContext);
}
