import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, describeError } from "../api/client";
import type { DecisionRequest } from "../api/types";
import { useInvestigation } from "../api/useInvestigation";
import { useSession } from "../auth/SessionContext";
import { ApprovalPanel } from "../components/ApprovalPanel";
import { AuditLog, ProposalCard, RetrievedDocs, ToolCalls } from "../components/Evidence";
import { QueryState } from "../components/QueryState";
import { StatusBadge } from "../components/StatusBadge";
import { Timeline } from "../components/Timeline";
import { ACTIVE, formatArgs } from "../lib/format";

type Tab = "tools" | "evidence" | "audit";

export function InvestigationPage() {
  const { id = "" } = useParams();
  const { me } = useSession();
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<Tab>("tools");
  const inv = useInvestigation(id);
  const data = inv.data;

  const pending = data?.approvals.find((a) => a.decision === null) ?? data?.approvals.at(-1);
  const approval = useQuery({
    queryKey: ["approval", pending?.id],
    queryFn: () => api.approval(pending!.id),
    enabled: Boolean(pending),
  });
  const toolCalls = useQuery({ queryKey: ["tool-calls", id], queryFn: () => api.toolCalls(id), enabled: tab === "tools" });
  const audit = useQuery({ queryKey: ["audit", id], queryFn: () => api.audit(id), enabled: tab === "audit" });

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ["investigation", id] });
    void queryClient.invalidateQueries({ queryKey: ["approval"] });
    void queryClient.invalidateQueries({ queryKey: ["investigations"] });
  };
  const decide = (body: DecisionRequest) => api.decide(pending!.id, body).finally(refresh);
  const resume = useMutation({ mutationFn: () => api.resume(id), onSettled: refresh });

  if (inv.isError) return <p className="text-sm text-red-700">{describeError(inv.error)}</p>;
  if (!data) return <p className="text-sm text-slate-500">Loading…</p>;

  const running = ACTIVE.has(data.status);
  const lastAction = data.actions.at(-1);

  return (
    <div className="space-y-6">
      <div>
        <Link to="/" className="text-xs text-slate-500 hover:underline">
          ← Investigations
        </Link>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <h1 className="font-mono text-lg font-semibold">{data.id}</h1>
          <StatusBadge status={data.status} />
        </div>
        <p className="mt-1 text-sm text-slate-700">“{data.user_query}”</p>
      </div>

      {data.failure_reason && (
        <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          <p>{data.failure_reason}</p>
          {me.role === "approver" && data.status === "failed" && (
            <button
              type="button"
              onClick={() => resume.mutate()}
              disabled={resume.isPending}
              className="mt-2 rounded-md border border-red-300 bg-white px-3 py-1 text-sm disabled:opacity-40"
            >
              Resume
            </button>
          )}
          {resume.isError && <p className="mt-1">{describeError(resume.error)}</p>}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-5">
        <section className="lg:col-span-2" aria-label="Timeline">
          <h2 className="mb-3 text-sm font-semibold text-slate-700">What the agent did</h2>
          <Timeline events={data.events ?? []} running={running} />
        </section>

        <div className="space-y-4 lg:col-span-3">
          {pending && (
            <QueryState query={approval} what="the approval">
              {(detail) => <ApprovalPanel approval={detail} role={me.role} onDecide={decide} />}
            </QueryState>
          )}
          {data.proposal && <ProposalCard proposal={data.proposal} />}
          {lastAction && (
            <section className="rounded-lg border border-slate-200 bg-white p-4" aria-label="Action">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Action</p>
              <p className="mt-1 font-mono text-sm">
                {lastAction.tool_name}({formatArgs(lastAction.arguments)})
              </p>
              <p className="mt-1 text-sm">
                {lastAction.outcome === "executed" ? "Executed" : "Failed"} after {lastAction.attempts} attempt
                {lastAction.attempts === 1 ? "" : "s"}
                {typeof lastAction.result?.operation_id === "string" && (
                  <span className="ml-2 font-mono text-xs text-slate-500">{lastAction.result.operation_id}</span>
                )}
              </p>
            </section>
          )}
          {data.final_summary && (
            <section className="rounded-lg border border-slate-200 bg-white p-4" aria-label="Summary">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Summary</p>
              <p className="mt-1 whitespace-pre-line text-sm">{data.final_summary}</p>
            </section>
          )}

          <section className="rounded-lg border border-slate-200 bg-white">
            <div role="tablist" className="flex gap-1 border-b border-slate-200 px-2">
              {(
                [
                  ["tools", `Tool calls (${data.tool_call_count})`],
                  ["evidence", `Retrieved (${data.retrieved_docs.length})`],
                  ["audit", "Audit log"],
                ] as const
              ).map(([key, label]) => (
                <button
                  key={key}
                  role="tab"
                  aria-selected={tab === key}
                  onClick={() => setTab(key)}
                  className={`px-3 py-2 text-sm ${
                    tab === key ? "border-b-2 border-slate-900 font-medium" : "text-slate-500"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="p-4">
              {tab === "tools" && (
                <QueryState query={toolCalls} what="tool calls">
                  {(calls) => <ToolCalls calls={calls} />}
                </QueryState>
              )}
              {tab === "evidence" && (
                <RetrievedDocs docs={data.retrieved_docs} cited={data.proposal?.evidence_citations ?? []} />
              )}
              {tab === "audit" && (
                <QueryState query={audit} what="the audit log">
                  {(events) => <AuditLog events={events} />}
                </QueryState>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
