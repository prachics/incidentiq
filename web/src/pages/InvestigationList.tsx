import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, describeError } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";
import { ACTIVE, formatTime } from "../lib/format";

const EXAMPLES = [
  "order-service looks unhealthy in the dashboard but I can't tell what changed",
  "getting paged for web-frontend, latency is way up. what's going on?",
];

function NewInvestigation() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const start = useMutation({
    mutationFn: (q: string) => api.start(q),
    onSuccess: ({ id }) => {
      void queryClient.invalidateQueries({ queryKey: ["investigations"] });
      navigate(`/investigations/${id}`);
    },
  });

  function submit(event: FormEvent) {
    event.preventDefault();
    start.mutate(query.trim());
  }

  return (
    <form onSubmit={submit} className="rounded-lg border border-slate-200 bg-white p-4">
      <label htmlFor="report" className="block text-sm font-medium">
        What is the on-call engineer seeing?
      </label>
      <textarea
        id="report"
        rows={2}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={EXAMPLES[0]}
        className="mt-2 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-slate-400"
      />
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          type="submit"
          disabled={query.trim().length < 10 || start.isPending}
          className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
        >
          Investigate
        </button>
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => setQuery(example)}
            className="truncate rounded-full border border-slate-200 px-2.5 py-1 text-xs text-slate-600 hover:bg-slate-50"
          >
            {example.slice(0, 48)}…
          </button>
        ))}
      </div>
      {start.isError && <p className="mt-2 text-sm text-red-700">{describeError(start.error)}</p>}
    </form>
  );
}

export function InvestigationList() {
  const list = useQuery({
    queryKey: ["investigations"],
    queryFn: api.investigations,
    // Fast while something runs; slowly otherwise, since other people start
    // investigations and decide approvals too.
    refetchInterval: (q) => (q.state.data?.some((i) => ACTIVE.has(i.status)) ? 4000 : 15000),
  });

  return (
    <div className="space-y-6">
      <NewInvestigation />
      <section>
        <h2 className="text-sm font-semibold text-slate-700">Investigations</h2>
        {list.isError && <p className="mt-2 text-sm text-red-700">{describeError(list.error)}</p>}
        <ul className="mt-2 divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white">
          {list.data?.map((inv) => (
            <li key={inv.id}>
              <Link to={`/investigations/${inv.id}`} className="flex items-center gap-4 px-4 py-3 hover:bg-slate-50">
                <StatusBadge status={inv.status} />
                <span className="min-w-0 flex-1 truncate text-sm">{inv.user_query}</span>
                <span className="shrink-0 font-mono text-xs text-slate-400">{inv.id}</span>
                <span className="hidden shrink-0 text-xs text-slate-400 sm:inline">{formatTime(inv.created_at)}</span>
              </Link>
            </li>
          ))}
          {list.data?.length === 0 && <li className="px-4 py-6 text-sm text-slate-500">None yet.</li>}
        </ul>
      </section>
    </div>
  );
}
