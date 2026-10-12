import { getApiKey } from "../auth/session";
import type {
  ApprovalDetail,
  AuditEvent,
  DecisionRequest,
  Investigation,
  InvestigationSummary,
  Me,
  ToolCall,
} from "./types";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
  }
}

export function authHeaders(key = getApiKey()): Record<string, string> {
  return key ? { Authorization: `Bearer ${key}` } : {};
}

async function request<T>(path: string, init: RequestInit = {}, key?: string): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...authHeaders(key),
      ...init.headers,
    },
  });
  const text = await response.text();
  const body: unknown = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail ?? body;
    throw new ApiError(response.status, detail);
  }
  return body as T;
}

export const api = {
  me: (key?: string) => request<Me>("/me", {}, key),
  investigations: () => request<InvestigationSummary[]>("/investigations"),
  investigation: (id: string) => request<Investigation>(`/investigations/${encodeURIComponent(id)}`),
  toolCalls: (id: string) => request<ToolCall[]>(`/investigations/${encodeURIComponent(id)}/tool-calls`),
  audit: (id: string) => request<AuditEvent[]>(`/investigations/${encodeURIComponent(id)}/audit`),
  approval: (id: string) => request<ApprovalDetail>(`/approvals/${encodeURIComponent(id)}`),
  start: (query: string) =>
    request<{ id: string }>("/investigations", { method: "POST", body: JSON.stringify({ query }) }),
  resume: (id: string) =>
    request<{ id: string }>(`/investigations/${encodeURIComponent(id)}/resume`, { method: "POST" }),
  decide: (approvalId: string, body: DecisionRequest) =>
    request<{ decision: string; decided_by: string }>(
      `/approvals/${encodeURIComponent(approvalId)}/decision`,
      { method: "POST", body: JSON.stringify(body) },
    ),
};

/** A readable message for any error the API returns, including 422 field lists. */
export function describeError(error: unknown): string {
  if (!(error instanceof ApiError)) return error instanceof Error ? error.message : String(error);
  const detail = error.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`)
      .join("; ");
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    const d = detail as { message: string; errors?: { loc: unknown[]; msg: string }[] };
    const fields = (d.errors ?? []).map((e) => `${e.loc.join(".")}: ${e.msg}`).join("; ");
    return fields ? `${d.message} (${fields})` : d.message;
  }
  return `HTTP ${error.status}`;
}
