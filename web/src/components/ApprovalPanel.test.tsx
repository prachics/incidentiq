import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ApiError } from "../api/client";
import type { ApprovalDetail } from "../api/types";
import { ApprovalPanel } from "./ApprovalPanel";

const approval: ApprovalDetail = {
  id: "APR-1",
  investigation_id: "IQ-1",
  proposed_tool: "scale_service",
  proposed_arguments: { service: "checkout-service", replica_count: 6, reason: "pool saturation under load" },
  reasoning: "connection pool exhausted",
  evidence: {},
  decision: null,
  decided_by: null,
  decided_at: null,
  rejection_reason: null,
  modified_arguments: null,
  created_at: "2026-10-12T00:00:00Z",
  arguments_schema: {
    type: "object",
    properties: {
      service: { type: "string" },
      replica_count: { type: "integer", minimum: 1, maximum: 50 },
      reason: { type: "string", minLength: 10 },
    },
    required: ["service", "replica_count", "reason"],
  },
};

describe("ApprovalPanel", () => {
  it("shows the agent's reasoning and the evidence it cites", () => {
    render(
      <ApprovalPanel
        approval={{
          ...approval,
          reasoning: "connection pool exhausted\n\nscale out",
          evidence: {
            documents: [
              { chunk_id: 1, doc_type: "runbook", parent_doc_id: "RB-027", title: null, service: null, excerpt: "",
                rrf_score: 1, vector_rank: 1, keyword_rank: 1 },
              { chunk_id: 2, doc_type: "runbook", parent_doc_id: "RB-027", title: null, service: null, excerpt: "",
                rrf_score: 1, vector_rank: 2, keyword_rank: 2 },
            ],
          },
        }}
        role="viewer"
        onDecide={vi.fn()}
      />,
    );
    expect(screen.getByText(/connection pool exhausted/)).toBeInTheDocument();
    expect(screen.getByText("Evidence: RB-027")).toBeInTheDocument();
  });

  it("lets a viewer see the request but not decide", () => {
    render(<ApprovalPanel approval={approval} role="viewer" onDecide={vi.fn()} />);
    expect(screen.getByText(/only an approver can decide/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
  });

  it("approves as proposed", async () => {
    const onDecide = vi.fn().mockResolvedValue({});
    render(<ApprovalPanel approval={approval} role="approver" onDecide={onDecide} />);
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(onDecide).toHaveBeenCalledWith({ decision: "approve" });
  });

  it("sends edited arguments, and blocks values outside the schema", async () => {
    const onDecide = vi.fn().mockResolvedValue({});
    render(<ApprovalPanel approval={approval} role="approver" onDecide={onDecide} />);
    await userEvent.click(screen.getByRole("button", { name: "Edit and approve" }));
    const replicas = screen.getByLabelText(/replica/i);
    await userEvent.clear(replicas);
    await userEvent.type(replicas, "500");
    const send = screen.getByRole("button", { name: "Approve with these arguments" });
    expect(screen.getByText("Must be at most 50")).toBeInTheDocument();
    expect(send).toBeDisabled();
    await userEvent.clear(replicas);
    await userEvent.type(replicas, "4");
    await userEvent.click(send);
    expect(onDecide).toHaveBeenCalledWith({
      decision: "approve",
      arguments: { service: "checkout-service", replica_count: 4, reason: "pool saturation under load" },
    });
  });

  it("needs a reason to reject", async () => {
    const onDecide = vi.fn().mockResolvedValue({});
    render(<ApprovalPanel approval={approval} role="approver" onDecide={onDecide} />);
    await userEvent.click(screen.getByRole("button", { name: "Reject" }));
    const confirm = screen.getByRole("button", { name: "Reject" });
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/why not/i), "checkout is a victim of the database");
    await userEvent.click(confirm);
    expect(onDecide).toHaveBeenCalledWith({ decision: "reject", reason: "checkout is a victim of the database" });
  });

  it("shows who won when another approver decided first", async () => {
    const onDecide = vi
      .fn()
      .mockRejectedValue(new ApiError(409, "approval APR-1 was already approved by alice at 2026-10-12"));
    render(<ApprovalPanel approval={approval} role="approver" onDecide={onDecide} />);
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("already approved by alice");
  });

  it("offers only an edit when the proposed arguments cannot execute", () => {
    render(
      <ApprovalPanel
        approval={{ ...approval, argument_problems: ["'v1.6.9' was never deployed to checkout-service"] }}
        role="approver"
        onDecide={vi.fn()}
      />,
    );
    expect(screen.getByText(/never deployed/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit and approve" })).toBeInTheDocument();
  });

  it("shows a decision once made", () => {
    render(
      <ApprovalPanel
        approval={{ ...approval, decision: "rejected", decided_by: "alice", rejection_reason: "wrong service" }}
        role="approver"
        onDecide={vi.fn()}
      />,
    );
    expect(screen.getByText(/rejected/i)).toBeInTheDocument();
    expect(screen.getByText(/wrong service/)).toBeInTheDocument();
  });
});
