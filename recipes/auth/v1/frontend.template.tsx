/**
 * Auth UI components — Next.js 14 App Router scaffold
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:REDIRECT_AFTER_LOGIN=/dashboard
 *   // RECIPE_PARAM:APP_NAME=MyApp
 *   // RECIPE_PARAM:API_BASE=/api
 *
 * Components exported:
 *   LoginForm           — email/password login with server action
 *   RegisterForm        — registration with server action
 *   PasswordResetForm   — request reset link by email
 *   EmailVerifyView     — consume verify token from URL query param
 *
 * Design tokens assumed: bg-primary, text-primary, text-secondary, border-input,
 * bg-destructive, text-destructive-foreground (shadcn/ui or equivalent Tailwind layer).
 */
"use client";

import { useState, useTransition } from "react";

// ---------------------------------------------------------------------------
// Server actions  (these files live in app/auth/actions.ts in the
// instantiated project — kept inline here as 'use server' stubs)
// ---------------------------------------------------------------------------

// RECIPE_PARAM:REDIRECT_AFTER_LOGIN=/dashboard
const REDIRECT_AFTER_LOGIN = "/dashboard";
// RECIPE_PARAM:API_BASE=/api
const API_BASE = "/api";

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

interface FieldError {
  field: string;
  message: string;
}

interface ActionResult {
  ok: boolean;
  message?: string;
  errors?: FieldError[];
  redirectTo?: string;
}

function FieldMessage({ error }: { error?: string }) {
  if (!error) return null;
  return <p className="text-sm text-destructive-foreground mt-1">{error}</p>;
}

function FormAlert({ message, variant }: { message: string; variant: "error" | "success" }) {
  const cls =
    variant === "error"
      ? "bg-destructive text-destructive-foreground"
      : "bg-green-100 text-green-800";
  return <div className={`rounded px-3 py-2 text-sm ${cls}`}>{message}</div>;
}

function SubmitButton({ loading, label }: { loading: boolean; label: string }) {
  return (
    <button
      type="submit"
      disabled={loading}
      className="w-full bg-primary text-primary px-4 py-2 rounded font-medium disabled:opacity-50"
    >
      {loading ? "Please wait…" : label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Login form
// ---------------------------------------------------------------------------

/**
 * LoginForm — submits to /auth/login via server action.
 *
 * RECIPE_PARAM: REDIRECT_AFTER_LOGIN controls where the user lands after success.
 */
export function LoginForm() {
  const [pending, startTransition] = useTransition();
  const [result, setResult] = useState<ActionResult | null>(null);

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = new FormData(e.currentTarget);
    const email = data.get("email") as string;
    const password = data.get("password") as string;

    startTransition(async () => {
      // 'use server' action — move to app/auth/actions.ts during instantiation
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
        credentials: "include", // required for httpOnly refresh cookie
      });
      if (res.ok) {
        const json = await res.json();
        // Store access token in memory / session storage (NOT localStorage)
        if (typeof window !== "undefined" && json.access_token) {
          sessionStorage.setItem("access_token", json.access_token);
        }
        // RECIPE_PARAM:REDIRECT_AFTER_LOGIN=/dashboard
        window.location.href = REDIRECT_AFTER_LOGIN;
      } else {
        const err = await res.json().catch(() => ({ detail: "Login failed" }));
        setResult({ ok: false, message: err.detail ?? "Login failed" });
      }
    });
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full max-w-sm">
      <h1 className="text-xl font-semibold text-primary">Sign in</h1>

      {result && !result.ok && (
        <FormAlert message={result.message ?? "Login failed"} variant="error" />
      )}

      <div>
        <label className="block text-sm font-medium text-secondary mb-1" htmlFor="login-email">
          Email
        </label>
        <input
          id="login-email"
          name="email"
          type="email"
          required
          autoComplete="email"
          className="w-full border border-input rounded px-3 py-2 text-sm focus:outline-none focus:ring-2"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-secondary mb-1" htmlFor="login-password">
          Password
        </label>
        <input
          id="login-password"
          name="password"
          type="password"
          required
          autoComplete="current-password"
          className="w-full border border-input rounded px-3 py-2 text-sm focus:outline-none focus:ring-2"
        />
      </div>

      <SubmitButton loading={pending} label="Sign in" />

      <p className="text-sm text-secondary text-center">
        <a href="/auth/reset" className="underline">Forgot password?</a>
      </p>
      <p className="text-sm text-secondary text-center">
        No account?{" "}
        <a href="/auth/register" className="underline">Register</a>
      </p>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Register form
// ---------------------------------------------------------------------------

/**
 * RegisterForm — creates account; on success shows "check your email" message.
 */
export function RegisterForm() {
  const [pending, startTransition] = useTransition();
  const [result, setResult] = useState<ActionResult | null>(null);

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = new FormData(e.currentTarget);
    const email = data.get("email") as string;
    const password = data.get("password") as string;
    const confirm = data.get("confirm_password") as string;

    if (password !== confirm) {
      setResult({ ok: false, message: "Passwords do not match" });
      return;
    }

    startTransition(async () => {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const json = await res.json().catch(() => ({}));
      if (res.ok) {
        setResult({ ok: true, message: json.message ?? "Check your email to verify your account." });
      } else {
        setResult({ ok: false, message: json.detail ?? "Registration failed" });
      }
    });
  }

  if (result?.ok) {
    return <FormAlert message={result.message ?? "Registered!"} variant="success" />;
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full max-w-sm">
      <h1 className="text-xl font-semibold text-primary">Create account</h1>

      {result && !result.ok && (
        <FormAlert message={result.message ?? "Registration failed"} variant="error" />
      )}

      <div>
        <label className="block text-sm font-medium text-secondary mb-1" htmlFor="reg-email">
          Email
        </label>
        <input
          id="reg-email"
          name="email"
          type="email"
          required
          autoComplete="email"
          className="w-full border border-input rounded px-3 py-2 text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-secondary mb-1" htmlFor="reg-password">
          Password
          <span className="ml-1 font-normal text-secondary">(min 8 characters)</span>
        </label>
        <input
          id="reg-password"
          name="password"
          type="password"
          required
          minLength={8}
          autoComplete="new-password"
          className="w-full border border-input rounded px-3 py-2 text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-secondary mb-1" htmlFor="reg-confirm">
          Confirm password
        </label>
        <input
          id="reg-confirm"
          name="confirm_password"
          type="password"
          required
          minLength={8}
          autoComplete="new-password"
          className="w-full border border-input rounded px-3 py-2 text-sm"
        />
      </div>

      <SubmitButton loading={pending} label="Create account" />

      <p className="text-sm text-secondary text-center">
        Already have an account?{" "}
        <a href="/auth/login" className="underline">Sign in</a>
      </p>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Password reset form
// ---------------------------------------------------------------------------

/**
 * PasswordResetForm — two-step: request link (by email) or confirm (token from URL).
 *
 * Render the request step when no `token` prop; render confirm step with token.
 */
export function PasswordResetForm({ token }: { token?: string }) {
  const [pending, startTransition] = useTransition();
  const [result, setResult] = useState<ActionResult | null>(null);

  function handleRequest(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const email = new FormData(e.currentTarget).get("email") as string;
    startTransition(async () => {
      const res = await fetch(`${API_BASE}/auth/password/reset/request`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      const json = await res.json().catch(() => ({}));
      setResult({ ok: res.ok, message: json.message ?? json.detail });
    });
  }

  function handleConfirm(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = new FormData(e.currentTarget);
    const new_password = data.get("new_password") as string;
    startTransition(async () => {
      const res = await fetch(`${API_BASE}/auth/password/reset/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, new_password }),
      });
      const json = await res.json().catch(() => ({}));
      if (res.ok) {
        setResult({ ok: true, message: "Password updated. You can now log in.", redirectTo: "/auth/login" });
      } else {
        setResult({ ok: false, message: json.detail ?? "Reset failed" });
      }
    });
  }

  if (result?.ok) {
    return (
      <div className="flex flex-col gap-3 w-full max-w-sm">
        <FormAlert message={result.message ?? "Done!"} variant="success" />
        {result.redirectTo && (
          <a href={result.redirectTo} className="text-sm underline text-secondary text-center">
            Back to login
          </a>
        )}
      </div>
    );
  }

  if (token) {
    // Confirm step
    return (
      <form onSubmit={handleConfirm} className="flex flex-col gap-4 w-full max-w-sm">
        <h1 className="text-xl font-semibold text-primary">Set new password</h1>
        {result && !result.ok && <FormAlert message={result.message ?? "Error"} variant="error" />}
        <div>
          <label className="block text-sm font-medium text-secondary mb-1" htmlFor="reset-pw">
            New password
          </label>
          <input
            id="reset-pw"
            name="new_password"
            type="password"
            required
            minLength={8}
            className="w-full border border-input rounded px-3 py-2 text-sm"
          />
        </div>
        <SubmitButton loading={pending} label="Update password" />
      </form>
    );
  }

  // Request step
  return (
    <form onSubmit={handleRequest} className="flex flex-col gap-4 w-full max-w-sm">
      <h1 className="text-xl font-semibold text-primary">Reset password</h1>
      {result && !result.ok && <FormAlert message={result.message ?? "Error"} variant="error" />}
      <p className="text-sm text-secondary text-center">
        Enter your email and we will send a reset link.
      </p>
      <div>
        <label className="block text-sm font-medium text-secondary mb-1" htmlFor="reset-email">
          Email
        </label>
        <input
          id="reset-email"
          name="email"
          type="email"
          required
          autoComplete="email"
          className="w-full border border-input rounded px-3 py-2 text-sm"
        />
      </div>
      <SubmitButton loading={pending} label="Send reset link" />
    </form>
  );
}

// ---------------------------------------------------------------------------
// Email verify view
// ---------------------------------------------------------------------------

/**
 * EmailVerifyView — reads `token` query param and POSTs to /auth/email/verify.
 *
 * Typically rendered server-side on first load; this client component shows
 * feedback after the POST.
 *
 * RECIPE_PARAM: wire the token prop from `searchParams.token` in the Page component.
 */
export function EmailVerifyView({ token }: { token?: string }) {
  const [pending, startTransition] = useTransition();
  const [result, setResult] = useState<ActionResult | null>(null);

  function handleVerify() {
    if (!token) return;
    startTransition(async () => {
      const res = await fetch(`${API_BASE}/auth/email/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });
      const json = await res.json().catch(() => ({}));
      setResult({ ok: res.ok, message: json.message ?? json.detail });
    });
  }

  if (!token) {
    return <FormAlert message="Missing verification token in URL." variant="error" />;
  }

  if (result) {
    return (
      <div className="flex flex-col gap-3 w-full max-w-sm">
        <FormAlert
          message={result.message ?? (result.ok ? "Verified!" : "Verification failed")}
          variant={result.ok ? "success" : "error"}
        />
        {result.ok && (
          <a href="/auth/login" className="text-sm underline text-secondary text-center">
            Proceed to login
          </a>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 w-full max-w-sm items-center">
      <h1 className="text-xl font-semibold text-primary">Verify your email</h1>
      <p className="text-sm text-secondary text-center">
        Click the button below to confirm your email address.
      </p>
      <button
        onClick={handleVerify}
        disabled={pending}
        className="bg-primary text-primary px-6 py-2 rounded font-medium disabled:opacity-50"
      >
        {pending ? "Verifying…" : "Verify email"}
      </button>
    </div>
  );
}
