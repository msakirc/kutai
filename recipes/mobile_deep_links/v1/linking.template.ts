/**
 * Expo Router linking configuration — universal/app links + custom scheme.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:APP_SCHEME=myapp
 *   // RECIPE_PARAM:ASSOCIATED_DOMAIN=links.example.com
 *   // RECIPE_PARAM:DEEP_LINK_PREFIX_PATH=/app
 *
 * The template engine replaces `KEY` tokens before this file is written
 * into the mission workspace. RECIPE_PARAM comment markers are left intact.
 *
 * expo-router auto-derives routes from the `app/` directory, so the screen
 * map below is the explicit fallback used by `getStateFromPath` for paths
 * that need custom parsing. For a pure file-based app you can pass just
 * `prefixes` and let the router infer the rest.
 */
import * as Linking from "expo-linking";
import type { LinkingOptions } from "@react-navigation/native";

const APP_SCHEME = "<<APP_SCHEME>>";
const ASSOCIATED_DOMAIN = "<<ASSOCIATED_DOMAIN>>";
const DEEP_LINK_PREFIX_PATH = "<<DEEP_LINK_PREFIX_PATH>>";

/**
 * Prefixes the app will resolve as deep links:
 *  - the custom scheme  (myapp://...)            — always works, no verification
 *  - the https universal/app link domain         — requires AASA + assetlinks
 *  - Linking.createURL("") for the Expo Go dev shell (exp://...)
 */
export const linkingPrefixes: string[] = [
  `${APP_SCHEME}://`,
  `https://${ASSOCIATED_DOMAIN}`,
  Linking.createURL("/"),
];

export const linking: LinkingOptions<Record<string, unknown>> = {
  prefixes: linkingPrefixes,
  config: {
    screens: {
      // expo-router maps these to app/index.tsx, app/item/[id].tsx, etc.
      index: "",
      "item/[id]": `${DEEP_LINK_PREFIX_PATH}/item/:id`,
      profile: `${DEEP_LINK_PREFIX_PATH}/profile`,
      "not-found": "*",
    },
  },
};

/** Build a canonical https deep link for sharing (universal/app link). */
export function buildShareLink(path: string): string {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return `https://${ASSOCIATED_DOMAIN}${DEEP_LINK_PREFIX_PATH}${clean}`;
}

/** Build a custom-scheme link (works without domain verification). */
export function buildSchemeLink(path: string): string {
  const clean = path.startsWith("/") ? path.slice(1) : path;
  return `${APP_SCHEME}://${clean}`;
}

/**
 * Parse an inbound deep-link URL into {screen, params}. Accepts both the
 * https and custom-scheme forms. Returns null for an unrecognised URL so
 * the caller can route to the not-found screen.
 */
export function parseDeepLink(
  url: string,
): { path: string; queryParams: Record<string, string> } | null {
  try {
    const parsed = Linking.parse(url);
    const path = parsed.path ?? "";
    const queryParams: Record<string, string> = {};
    for (const [k, v] of Object.entries(parsed.queryParams ?? {})) {
      if (typeof v === "string") queryParams[k] = v;
    }
    return { path, queryParams };
  } catch {
    return null;
  }
}

export { APP_SCHEME, ASSOCIATED_DOMAIN, DEEP_LINK_PREFIX_PATH };
