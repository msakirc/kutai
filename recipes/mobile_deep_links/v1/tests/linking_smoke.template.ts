/**
 * Smoke tests for the mobile_deep_links/v1 linking config.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:APP_SCHEME=myapp
 *   // RECIPE_PARAM:ASSOCIATED_DOMAIN=links.example.com
 *
 * Run with the project's TS test runner (jest or vitest). expo-linking is
 * mocked because it touches native modules.
 */
import { describe, it, expect } from "@jest/globals";
import {
  buildShareLink,
  buildSchemeLink,
  linkingPrefixes,
} from "../linking";

const APP_SCHEME = "<<APP_SCHEME>>";
const ASSOCIATED_DOMAIN = "<<ASSOCIATED_DOMAIN>>";

describe("mobile_deep_links/v1 linking config", () => {
  it("includes the custom scheme prefix", () => {
    expect(linkingPrefixes.some((p) => p.startsWith(`${APP_SCHEME}://`))).toBe(
      true,
    );
  });

  it("includes the https universal-link domain prefix", () => {
    expect(
      linkingPrefixes.some((p) => p === `https://${ASSOCIATED_DOMAIN}`),
    ).toBe(true);
  });

  it("buildShareLink produces an absolute https URL", () => {
    const link = buildShareLink("/item/42");
    expect(link.startsWith(`https://${ASSOCIATED_DOMAIN}`)).toBe(true);
    expect(link).toContain("/item/42");
  });

  it("buildSchemeLink produces a custom-scheme URL", () => {
    const link = buildSchemeLink("item/42");
    expect(link).toBe(`${APP_SCHEME}://item/42`);
  });

  it("buildShareLink normalises a path with no leading slash", () => {
    const link = buildShareLink("profile");
    expect(link).toContain("/profile");
  });
});
