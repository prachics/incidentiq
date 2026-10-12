import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { KeyGate } from "./auth/KeyGate";
import { useSession } from "./auth/SessionContext";
import { InvestigationList } from "./pages/InvestigationList";
import { InvestigationPage } from "./pages/InvestigationPage";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1 } },
});

function Shell() {
  const { me, signOut } = useSession();
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <Link to="/" className="text-sm font-semibold tracking-tight">
            IncidentIQ
          </Link>
          <div className="flex items-center gap-3 text-xs text-slate-600">
            <span>
              {me.user} · <span className="font-medium">{me.role}</span>
            </span>
            <button type="button" onClick={signOut} className="underline-offset-2 hover:underline">
              Sign out
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/" element={<InvestigationList />} />
          <Route path="/investigations/:id" element={<InvestigationPage />} />
        </Routes>
      </main>
    </div>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <KeyGate>
          <Shell />
        </KeyGate>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
