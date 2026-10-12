// Shapes returned by the IncidentIQ API. Field names match the backend exactly.

export type Status = "queued" | "running" | "awaiting_approval" | "complete" | "failed";
export type Role = "viewer" | "approver";
export type Decision = "approved" | "modified" | "rejected";

export interface Me {
  user: string;
  role: Role;
}

export interface InvestigationSummary {
  id: string;
  user_query: string;
  status: Status;
  iteration: number;
  failure_reason: string | null;
  created_at: string;
  updated_at: string;
}

export interface NodeEvent {
  node: string;
  duration_ms: number;
  at: number;
  ok?: boolean;
  [extra: string]: unknown;
}

export interface RetrievedDoc {
  chunk_id: number;
  doc_type: string;
  parent_doc_id: string;
  title: string | null;
  service: string | null;
  excerpt: string;
  rrf_score: number;
  vector_rank: number | null;
  keyword_rank: number | null;
}

export interface Proposal {
  root_cause: string;
  confidence: number;
  evidence_citations: string[];
  remediation: string;
  remediation_tool: string | null;
  remediation_arguments: Record<string, unknown> | null;
  requires_approval: boolean;
  abstained: boolean;
  abstention_reason: string | null;
}

export interface ApprovalSummary {
  id: string;
  proposed_tool: string;
  proposed_arguments: Record<string, unknown>;
  reasoning?: string;
  decision: Decision | null;
  decided_by: string | null;
  decided_at: string | null;
  rejection_reason: string | null;
  modified_arguments: Record<string, unknown> | null;
  created_at: string;
}

export interface ActionRecord {
  approval_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  outcome: "executed" | "failed" | "recorded" | "simulated_failure";
  attempts: number;
  result: Record<string, unknown> | null;
  error: string | null;
  executed_at: string;
}

export interface Investigation extends InvestigationSummary {
  entities: Record<string, unknown>;
  final_summary: string | null;
  scratchpad: string[];
  retrieved_docs: RetrievedDoc[];
  events: NodeEvent[];
  proposal: Proposal | null;
  approvals: ApprovalSummary[];
  actions: ActionRecord[];
  tool_call_count: number;
}

export interface JsonSchema {
  type?: string;
  title?: string;
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  anyOf?: JsonSchema[];
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  default?: unknown;
}

export interface ApprovalDetail extends ApprovalSummary {
  investigation_id: string;
  reasoning: string;
  evidence: { documents?: RetrievedDoc[]; tool_calls?: ToolCall[] };
  arguments_schema: JsonSchema | null;
  /** Why the proposed arguments could not execute as they stand; empty if they can. */
  argument_problems?: string[];
}

export interface ToolCall {
  tool_name: string;
  arguments: Record<string, unknown>;
  attempt: number;
  status: "success" | "failed" | "timeout" | "malformed" | "partial";
  result: Record<string, unknown> | null;
  error: string | null;
  latency_ms: number | null;
  injected_failure: boolean;
  started_at?: string;
}

export interface AuditEvent {
  event_type: string;
  actor: string;
  payload: Record<string, unknown>;
  occurred_at: string;
}

export interface DecisionRequest {
  decision: "approve" | "reject";
  reason?: string;
  arguments?: Record<string, unknown>;
}
