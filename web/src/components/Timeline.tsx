import type { NodeEvent } from "../api/types";
import { formatMs } from "../lib/format";

const LABEL: Record<string, string> = {
  intake: "Read the report",
  retrieve: "Retrieved similar incidents and runbooks",
  plan: "Chose a diagnostic",
  execute_tool: "Ran the diagnostic",
  reflect: "Checked whether the evidence is enough",
  propose: "Proposed a cause and remediation",
  await_approval: "Asked for approval",
  act: "Acted on the decision",
  summarize: "Wrote the summary",
};

function detail(event: NodeEvent): string | null {
  const parts: string[] = [];
  if (typeof event.tool === "string") parts.push(event.tool);
  if (typeof event.service === "string") parts.push(event.service);
  if (typeof event.decision === "string") parts.push(event.decision);
  if (typeof event.attempts === "number" && event.attempts > 1) parts.push(`${event.attempts} attempts`);
  if (event.sufficient === false) parts.push("needs more evidence");
  if (event.abstained === true) parts.push("abstained");
  return parts.length ? parts.join(" · ") : null;
}

export function Timeline({ events, running }: { events: NodeEvent[]; running: boolean }) {
  if (!events.length) {
    return <p className="text-sm text-slate-500">{running ? "Starting…" : "No steps recorded."}</p>;
  }
  return (
    <ol className="relative space-y-3 border-l border-slate-200 pl-5" aria-label="Investigation steps">
      {events.map((event, i) => {
        const failed = event.ok === false;
        const extra = detail(event);
        return (
          <li key={`${event.node}-${i}`} className="relative">
            <span
              className={`absolute -left-[25px] top-1.5 h-2.5 w-2.5 rounded-full ring-4 ring-white ${
                failed ? "bg-red-500" : "bg-slate-400"
              }`}
            />
            <div className="flex items-baseline justify-between gap-4">
              <p className={`text-sm ${failed ? "text-red-800" : "text-slate-800"}`}>
                {LABEL[event.node] ?? event.node}
                {extra && <span className="ml-2 font-mono text-xs text-slate-500">{extra}</span>}
              </p>
              <span className="shrink-0 font-mono text-xs text-slate-400">{formatMs(event.duration_ms)}</span>
            </div>
          </li>
        );
      })}
      {running && (
        <li className="relative text-sm text-blue-700">
          <span className="absolute -left-[25px] top-1.5 h-2.5 w-2.5 animate-pulse rounded-full bg-blue-500 ring-4 ring-white" />
          Working…
        </li>
      )}
    </ol>
  );
}
