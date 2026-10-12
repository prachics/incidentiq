import { useMemo, useState } from "react";
import { ApiError, describeError } from "../api/client";
import type { ApprovalDetail, DecisionRequest, Role } from "../api/types";
import { formatArgs, formatTime } from "../lib/format";
import { fieldsFromSchema, toArguments, toFormValues, validate, type FormValues } from "../lib/schemaForm";
import { SchemaForm } from "./SchemaForm";

type Mode = "idle" | "edit" | "reject";

interface Props {
  approval: ApprovalDetail;
  role: Role;
  onDecide: (body: DecisionRequest) => Promise<unknown>;
}

/**
 * Approve as proposed, approve with edited arguments, or reject with a reason.
 * Buttons are not optimistic: another approver may have decided first, and the
 * server's 409 is shown rather than a decision that did not happen.
 */
export function ApprovalPanel({ approval, role, onDecide }: Props) {
  const fields = useMemo(() => fieldsFromSchema(approval.arguments_schema), [approval.arguments_schema]);
  const [mode, setMode] = useState<Mode>("idle");
  const [values, setValues] = useState<FormValues>(() => toFormValues(fields, approval.proposed_arguments));
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fieldErrors = mode === "edit" ? validate(fields, values) : {};
  // The server refuses to approve arguments that cannot execute; only an edit can.
  const blocked = (approval.argument_problems?.length ?? 0) > 0;
  const reasonOk = reason.trim().length >= 5;

  async function send(body: DecisionRequest) {
    setBusy(true);
    setError(null);
    try {
      await onDecide(body);
      setMode("idle");
    } catch (e) {
      setError(e instanceof ApiError && e.status === 409 ? `${describeError(e)}.` : describeError(e));
    } finally {
      setBusy(false);
    }
  }

  if (approval.decision) {
    const tone = approval.decision === "rejected" ? "border-red-200 bg-red-50" : "border-emerald-200 bg-emerald-50";
    return (
      <section className={`rounded-lg border p-4 ${tone}`} aria-label="Decision">
        <p className="text-sm">
          <span className="font-medium capitalize">{approval.decision}</span> by {approval.decided_by} ·{" "}
          {formatTime(approval.decided_at)}
        </p>
        {approval.rejection_reason && <p className="mt-1 text-sm text-slate-700">“{approval.rejection_reason}”</p>}
        {approval.modified_arguments && (
          <p className="mt-1 font-mono text-xs text-slate-700">
            approved arguments: {formatArgs(approval.modified_arguments)}
          </p>
        )}
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-amber-300 bg-amber-50/60 p-4" aria-label="Approval">
      <p className="text-xs font-medium uppercase tracking-wide text-amber-900">Approval needed</p>
      <p className="mt-1 font-mono text-sm">
        {approval.proposed_tool}({formatArgs(approval.proposed_arguments)})
      </p>
      {approval.reasoning && (
        <p className="mt-2 whitespace-pre-line text-sm text-slate-700">{approval.reasoning.trim()}</p>
      )}
      {(approval.evidence?.documents?.length ?? 0) > 0 && (
        <p className="mt-2 text-xs text-slate-500">
          Evidence: {approval.evidence.documents!.map((d) => d.parent_doc_id).filter((v, i, all) => all.indexOf(v) === i).join(", ")}
        </p>
      )}

      {(approval.argument_problems?.length ?? 0) > 0 && (
        <div className="mt-3 rounded border border-red-200 bg-red-50 p-2 text-xs text-red-800">
          <p className="font-medium">These arguments cannot execute as proposed:</p>
          <ul className="mt-1 list-disc pl-4">
            {approval.argument_problems!.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      {role !== "approver" ? (
        <p className="mt-3 text-sm text-slate-600">Only an approver can decide. You are signed in as a viewer.</p>
      ) : (
        <>
          {mode === "edit" && (
            <div className="mt-4">
              <SchemaForm fields={fields} values={values} errors={fieldErrors} onChange={setValues} />
            </div>
          )}
          {mode === "reject" && (
            <div className="mt-4">
              <label htmlFor="reject-reason" className="block text-xs font-medium text-slate-700">
                Why not? The agent reads this when it revises its diagnosis.
              </label>
              <textarea
                id="reject-reason"
                rows={3}
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
              />
            </div>
          )}
          {error && (
            <p role="alert" className="mt-3 text-sm text-red-700">
              {error}
            </p>
          )}
          <div className="mt-4 flex flex-wrap gap-2">
            {mode === "idle" && (
              <>
                {!blocked && (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => send({ decision: "approve" })}
                    className="rounded-md bg-emerald-700 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
                  >
                    Approve
                  </button>
                )}
                <button
                  type="button"
                  disabled={busy || !fields.length}
                  onClick={() => setMode("edit")}
                  className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm disabled:opacity-40"
                >
                  Edit and approve
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => setMode("reject")}
                  className="rounded-md border border-red-300 bg-white px-3 py-1.5 text-sm text-red-800 disabled:opacity-40"
                >
                  Reject
                </button>
              </>
            )}
            {mode === "edit" && (
              <button
                type="button"
                disabled={busy || Object.keys(fieldErrors).length > 0}
                onClick={() => send({ decision: "approve", arguments: toArguments(fields, values) })}
                className="rounded-md bg-emerald-700 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
              >
                Approve with these arguments
              </button>
            )}
            {mode === "reject" && (
              <button
                type="button"
                disabled={busy || !reasonOk}
                onClick={() => send({ decision: "reject", reason: reason.trim() })}
                className="rounded-md bg-red-700 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
              >
                Reject
              </button>
            )}
            {mode !== "idle" && (
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  setMode("idle");
                  setError(null);
                }}
                className="px-3 py-1.5 text-sm text-slate-600"
              >
                Cancel
              </button>
            )}
          </div>
        </>
      )}
    </section>
  );
}
