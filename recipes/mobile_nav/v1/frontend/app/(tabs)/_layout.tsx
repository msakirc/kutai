/**
 * Bottom-tab navigator — nested inside the root Stack.
 *
 * Lives at `app/(tabs)/_layout.tsx`. Every directory under `app/` that should
 * own a navigator needs its own `_layout.tsx`; this one renders the bottom
 * tab bar. The `(tabs)` group name adds NO path segment, so `index.tsx` here
 * is the app root route `/`.
 *
 * Each `<Tabs.Screen name>` MUST match a sibling file's basename (minus the
 * `.tsx` extension). A `name` with no matching file silently renders nothing;
 * a file with no `Tabs.Screen` still appears as a tab using its filename.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:HOME_TAB_TITLE=Home
 *   // RECIPE_PARAM:EXPLORE_TAB_TITLE=Explore
 *   // RECIPE_PARAM:PROFILE_TAB_TITLE=Profile
 *   // RECIPE_PARAM:INITIAL_TAB=index
 */
import { Tabs } from "expo-router";
import { Text } from "react-native";

// RECIPE_PARAM:HOME_TAB_TITLE=Home
const HOME_TAB_TITLE = "<<HOME_TAB_TITLE>>";
// RECIPE_PARAM:EXPLORE_TAB_TITLE=Explore
const EXPLORE_TAB_TITLE = "<<EXPLORE_TAB_TITLE>>";
// RECIPE_PARAM:PROFILE_TAB_TITLE=Profile
const PROFILE_TAB_TITLE = "<<PROFILE_TAB_TITLE>>";

/**
 * TabIcon — placeholder glyph icon.
 *
 * Swap for `@expo/vector-icons` (`<Ionicons name="home" .../>`) in a real
 * project; emoji keeps the recipe dependency-free and testable.
 */
function TabIcon({ glyph, color }: { glyph: string; color: string }) {
  return <Text style={{ fontSize: 20, color }}>{glyph}</Text>;
}

export default function TabLayout() {
  return (
    <Tabs
      // RECIPE_PARAM:INITIAL_TAB=index
      initialRouteName="<<INITIAL_TAB>>"
      screenOptions={{
        headerShown: true,
        tabBarActiveTintColor: "#2563eb",
        tabBarInactiveTintColor: "#94a3b8",
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: HOME_TAB_TITLE,
          tabBarIcon: ({ color }) => <TabIcon glyph="🏠" color={color} />,
        }}
      />
      <Tabs.Screen
        name="explore"
        options={{
          title: EXPLORE_TAB_TITLE,
          tabBarIcon: ({ color }) => <TabIcon glyph="🔍" color={color} />,
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: PROFILE_TAB_TITLE,
          tabBarIcon: ({ color }) => <TabIcon glyph="👤" color={color} />,
        }}
      />
    </Tabs>
  );
}
