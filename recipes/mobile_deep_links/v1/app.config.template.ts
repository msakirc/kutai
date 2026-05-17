/**
 * Expo app config fragment — scheme + associated-domains + intent filters.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:APP_SCHEME=myapp
 *   // RECIPE_PARAM:ASSOCIATED_DOMAIN=links.example.com
 *   // RECIPE_PARAM:ANDROID_PACKAGE=com.example.app
 *   // RECIPE_PARAM:IOS_BUNDLE_ID=com.example.app
 *
 * Merge the returned object into your `app.config.ts` (or hand-port the
 * fields into `app.json`). The `scheme`, `ios.associatedDomains`, and
 * `android.intentFilters` keys are what make deep links resolve.
 *
 * The template engine replaces `KEY` tokens before this file is written.
 */
import type { ExpoConfig } from "expo/config";

const APP_SCHEME = "<<APP_SCHEME>>";
const ASSOCIATED_DOMAIN = "<<ASSOCIATED_DOMAIN>>";
const ANDROID_PACKAGE = "<<ANDROID_PACKAGE>>";
const IOS_BUNDLE_ID = "<<IOS_BUNDLE_ID>>";

/** Deep-link-relevant Expo config fragment. Shallow-merge into ExpoConfig. */
export const deepLinkConfig: Partial<ExpoConfig> = {
  // Custom URL scheme — `myapp://...`. Single source of truth; the
  // linking config reuses the same RECIPE_PARAM:APP_SCHEME value.
  scheme: APP_SCHEME,
  ios: {
    bundleIdentifier: IOS_BUNDLE_ID,
    // Universal Links: the `applinks:` entry must match the AASA-hosting
    // domain EXACTLY (no scheme, no trailing slash, no path).
    associatedDomains: [`applinks:${ASSOCIATED_DOMAIN}`],
  },
  android: {
    package: ANDROID_PACKAGE,
    // App Links: autoVerify makes Android verify the assetlinks.json so
    // links open the app directly instead of the disambiguation dialog.
    intentFilters: [
      {
        action: "VIEW",
        autoVerify: true,
        data: [{ scheme: "https", host: ASSOCIATED_DOMAIN }],
        category: ["BROWSABLE", "DEFAULT"],
      },
    ],
  },
};

export default deepLinkConfig;
