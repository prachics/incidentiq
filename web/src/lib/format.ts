import type { Status } from "../api/types";

export const STATUS_LABEL: Record<Status, string> = {
  queued: "Queued",
  running: "Running",
  awaiting_approval: "Awaiting approval",
  complete: "Complete",
  failed: "Failed",
};

/** Statuses during which the investigation is still changing. */
export const ACTIVE: ReadonlySet<Status> = new Set<Status>(["queued", "running"]);

export function formatMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  return s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export function formatArgs(args: Record<string, unknown> | null | undefined): string {
  if (!args) return "";
  return Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join(", ");
}
