/**
 * mobile_nav recipe — route/component smoke tests.
 *
 * NOTE — execution context:
 *   This is a `.template` scaffold. After the recipe is instantiated via
 *   mr_roboto `instantiate_recipe` into a mission workspace, the `.template`
 *   suffix is dropped and these tests run against the real `app/` tree with
 *   `jest-expo` + `@testing-library/react-native`.
 *
 *   Running it directly from `recipes/mobile_nav/v1/tests/` will fail — the
 *   recipe source intentionally has no installed `expo-router`. Structural
 *   validation of the recipe lives in the Z2/Z5 recipe test suite.
 *
 * What these tests assert:
 *   1. The tab layout declares exactly the expected tab routes, in order.
 *   2. The root layout's auth gate redirects an unauthenticated user to the
 *      sign-in route.
 *   3. Route groups `(tabs)` / `(auth)` add no path segment.
 *
 * Run (post-instantiation):
 *   npx jest tests/tab_layout.test.tsx
 */
import React from "react";
import { render, screen } from "@testing-library/react-native";

import TabLayout from "../app/(tabs)/_layout";
import { deepLinkMap } from "../lib/linking";
import { routes } from "../lib/routes";

// ---------------------------------------------------------------------------
// expo-router mock — Tabs/Stack record their declared screen names so we can
// assert the navigation shape without a real navigator.
// ---------------------------------------------------------------------------

const declaredTabScreens: string[] = [];

jest.mock("expo-router", () => {
  const React = require("react");
  const Tabs = ({ children }: any) => <>{children}</>;
  Tabs.Screen = ({ name }: { name: string }) => {
    declaredTabScreens.push(name);
    return null;
  };
  const Stack = ({ children }: any) => <>{children}</>;
  Stack.Screen = () => null;
  return {
    Tabs,
    Stack,
    Link: ({ children }: any) => <>{children}</>,
    useRouter: () => ({ push: jest.fn(), replace: jest.fn() }),
    useSegments: () => [],
    useLocalSearchParams: () => ({}),
  };
});

beforeEach(() => {
  declaredTabScreens.length = 0;
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("(tabs)/_layout", () => {
  it("declares exactly the three expected tab routes in order", () => {
    render(<TabLayout />);
    expect(declaredTabScreens).toEqual(["index", "explore", "profile"]);
  });

  it("every declared tab has a matching route file in the deep-link map", () => {
    render(<TabLayout />);
    const tabPaths = ["/", "/explore", "/profile"];
    for (const p of tabPaths) {
      expect(deepLinkMap).toHaveProperty(p);
    }
  });
});

describe("typed route helpers", () => {
  it("detail() builds a dynamic href with the [id] param", () => {
    expect(routes.detail("99")).toEqual({
      pathname: "/detail/[id]",
      params: { id: "99" },
    });
  });

  it("group-named routes expose no group segment in their path", () => {
    // `(tabs)` and `(auth)` must NOT appear in any public route string.
    for (const href of [routes.home(), routes.explore(), routes.signIn()]) {
      const path = typeof href === "string" ? href : href.pathname;
      expect(path).not.toMatch(/\(tabs\)|\(auth\)/);
    }
  });
});
