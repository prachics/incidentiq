import type { Status } from "../api/types";
import { STATUS_LABEL } from "../lib/format";

const STYLE: Record<Status, string> = {
  queued: "bg-slate-100 text-slate-700 ring-slate-300",
  running: "bg-blue-50 text-blue-800 ring-blue-200",
  awaiting_approval: "bg-amber-50 text-amber-900 ring-amber-300",
  complete: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  failed: "bg-red-50 text-red-800 ring-red-200",
};

export function StatusBadge({ status }: { status: Status }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${STYLE[status]}`}
    >
      {status === "running" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-600" />}
      {STATUS_LABEL[status]}
    </span>
  );
}
