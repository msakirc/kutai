# Next.js — auth/v1 instantiation notes

When wiring this recipe in a Next.js 14 App Router project:

- **Server actions vs client fetch**: the template uses `useTransition` + client-side `fetch` for simplicity. For production, move the fetch calls into `app/auth/actions.ts` with `'use server'` and call them from the form's `action` prop — this enables progressive enhancement and avoids exposing API_BASE to the client bundle.
- **Refresh cookie visibility**: `next/headers` `cookies()` can read the httpOnly `refresh_token` cookie in Server Components and Route Handlers. The `credentials: "include"` on client fetch ensures the cookie is forwarded to the FastAPI backend via the Next.js API proxy.
- **RECIPE_PARAM:REDIRECT_AFTER_LOGIN**: replace `/dashboard` with your post-login landing page. If the destination is behind a layout that checks auth, add a `middleware.ts` matcher so unauthenticated users are redirected back to `/auth/login`.
- **Tailwind tokens**: `bg-primary`, `text-secondary`, `border-input`, `bg-destructive` must be defined in your `tailwind.config.ts` theme extension. If using shadcn/ui these are already present via CSS variables.
- **Access token in sessionStorage**: the template stores the access token in `sessionStorage` (not `localStorage`) — it is cleared on tab close. For persistent sessions across browser restarts, the refresh flow re-issues a new access token on every page load via the `/auth/refresh` route.
