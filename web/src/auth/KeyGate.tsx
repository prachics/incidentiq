import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { api, describeError } from "../api/client";
import type { Me } from "../api/types";
import { SessionContext } from "./SessionContext";
import { getApiKey, setApiKey } from "./session";

/** Asks for an API key, checks it with GET /me, and provides the session to the app. */
export function KeyGate({ children }: { children: ReactNode }) {
  const [me, setMe] = useState<Me | null>(null);
  const [checking, setChecking] = useState(() => getApiKey() !== null);
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const stored = getApiKey();
    if (!stored) return;
    api
      .me(stored)
      .then(setMe)
      .catch(() => setApiKey(null))
      .finally(() => setChecking(false));
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const who = await api.me(key.trim());
      setApiKey(key.trim());
      setMe(who);
    } catch (e) {
      setError(describeError(e));
    }
  }

  if (checking) return null;
  if (me) {
    const signOut = () => {
      setApiKey(null);
      setMe(null);
      setKey("");
    };
    return <SessionContext.Provider value={{ me, signOut }}>{children}</SessionContext.Provider>;
  }

  return (
    <main className="mx-auto mt-24 max-w-sm px-4">
      <h1 className="text-xl font-semibold">IncidentIQ</h1>
      <p className="mt-1 text-sm text-slate-600">Sign in with an API key from API_KEYS.</p>
      <form onSubmit={submit} className="mt-6 space-y-3">
        <label className="block text-sm font-medium" htmlFor="api-key">
          API key
        </label>
        <input
          id="api-key"
          type="password"
          autoComplete="off"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
        {error && <p className="text-sm text-red-700">{error}</p>}
        <button
          type="submit"
          disabled={!key.trim()}
          className="w-full rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          Sign in
        </button>
      </form>
    </main>
  );
}
