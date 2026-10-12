import type { AuditEvent, Proposal, RetrievedDoc, ToolCall } from "../api/types";
import { formatArgs, formatMs, formatTime } from "../lib/format";

export function ProposalCard({ proposal }: { proposal: Proposal }) {
  if (proposal.abstained) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-4" aria-label="Conclusion">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Abstained</p>
        <p className="mt-1 text-sm">{proposal.abstention_reason ?? "The evidence did not support a conclusion."}</p>
      </section>
    );
  }
  const pct = Math.round(proposal.confidence * 100);
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4" aria-label="Conclusion">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Root cause</p>
        <span className="text-xs text-slate-500" title="The model's own estimate, not a calibrated probability">
          confidence {pct}%
        </span>
      </div>
      <p className="mt-1 text-sm">{proposal.root_cause || "—"}</p>
      {proposal.remediation && (
        <>
          <p className="mt-3 text-xs font-medium uppercase tracking-wide text-slate-500">Remediation</p>
          <p className="mt-1 text-sm">{proposal.remediation}</p>
        </>
      )}
      {proposal.evidence_citations.length > 0 && (
        <p className="mt-3 text-xs text-slate-500">Cites {proposal.evidence_citations.join(", ")}</p>
      )}
    </section>
  );
}

const CALL_STYLE: Record<ToolCall["status"], string> = {
  success: "text-emerald-700",
  partial: "text-amber-700",
  failed: "text-red-700",
  timeout: "text-red-700",
  malformed: "text-red-700",
};

export function ToolCalls({ calls }: { calls: ToolCall[] }) {
  if (!calls.length) return <p className="text-sm text-slate-500">No tool calls yet.</p>;
  return (
    <ul className="divide-y divide-slate-100">
      {calls.map((c, i) => (
        <li key={i} className="py-2">
          <div className="flex items-baseline justify-between gap-3">
            <p className="font-mono text-sm">
              {c.tool_name}
              <span className="text-slate-500">({formatArgs(c.arguments)})</span>
            </p>
            <span className={`shrink-0 text-xs font-medium ${CALL_STYLE[c.status]}`}>
              {c.status}
              {c.attempt > 1 && ` · attempt ${c.attempt}`}
              {c.injected_failure && " · injected"}
              {c.latency_ms !== null && ` · ${formatMs(c.latency_ms)}`}
            </span>
          </div>
          {c.error && <p className="mt-1 text-xs text-red-700">{c.error}</p>}
          {c.result && (
            <details className="mt-1">
              <summary className="cursor-pointer text-xs text-slate-500">Result</summary>
              <pre className="mt-1 max-h-64 overflow-auto rounded bg-slate-50 p-2 text-[11px]">
                {JSON.stringify(c.result, null, 2)}
              </pre>
            </details>
          )}
        </li>
      ))}
    </ul>
  );
}

export function RetrievedDocs({ docs, cited }: { docs: RetrievedDoc[]; cited: string[] }) {
  if (!docs.length) return <p className="text-sm text-slate-500">Nothing retrieved yet.</p>;
  return (
    <ul className="space-y-3">
      {docs.map((d) => (
        <li key={d.chunk_id} className="rounded-md border border-slate-200 bg-white p-3">
          <div className="flex items-baseline justify-between gap-3">
            <p className="text-sm font-medium">
              <span className="font-mono text-xs text-slate-500">{d.parent_doc_id}</span> {d.title}
              {cited.includes(d.parent_doc_id) && (
                <span className="ml-2 rounded bg-slate-900 px-1.5 py-0.5 text-[10px] font-medium text-white">cited</span>
              )}
            </p>
            <span
              className="shrink-0 font-mono text-[11px] text-slate-500"
              title="Rank in vector search and in keyword search, fused with reciprocal rank fusion"
            >
              vec #{d.vector_rank ?? "–"} · kw #{d.keyword_rank ?? "–"}
            </span>
          </div>
          <p className="mt-1 line-clamp-3 text-xs text-slate-600">{d.excerpt}</p>
        </li>
      ))}
    </ul>
  );
}

export function AuditLog({ events }: { events: AuditEvent[] }) {
  if (!events.length) return <p className="text-sm text-slate-500">No events.</p>;
  return (
    <ul className="space-y-1 font-mono text-xs">
      {events.map((e, i) => (
        <li key={i}>
          <details>
            <summary className="flex cursor-pointer gap-3">
              <span className="w-44 shrink-0 text-slate-400">{formatTime(e.occurred_at)}</span>
              <span className="w-56 shrink-0">{e.event_type}</span>
              <span className="text-slate-600">{e.actor}</span>
            </summary>
            <pre className="ml-4 mt-1 max-h-64 overflow-auto rounded bg-slate-50 p-2 text-[11px]">
              {JSON.stringify(e.payload, null, 2)}
            </pre>
          </details>
        </li>
      ))}
    </ul>
  );
}
