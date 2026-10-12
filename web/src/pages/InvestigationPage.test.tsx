import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { SessionContext } from "../auth/SessionContext";
import { InvestigationPage } from "./InvestigationPage";

const waiting = {
  id: "IQ-1",
  user_query: "checkout-service is timing out",
  status: "awaiting_approval",
  iteration: 1,
  failure_reason: null,
  created_at: "2026-10-12T00:00:00Z",
  updated_at: "2026-10-12T00:00:00Z",
  entities: {},
  final_summary: null,
  scratchpad: [],
  retrieved_docs: [],
  events: [],
  proposal: null,
  approvals: [
    {
      id: "APR-1",
      proposed_tool: "scale_service",
      proposed_arguments: { service: "checkout-service", replica_count: 6, reason: "pool saturation" },
      decision: null,
      decided_by: null,
      decided_at: null,
      rejection_reason: null,
      modified_arguments: null,
      created_at: "2026-10-12T00:00:00Z",
    },
  ],
  actions: [],
  tool_call_count: 0,
};

const detail = { ...waiting.approvals[0], investigation_id: "IQ-1", reasoning: "r", evidence: {}, arguments_schema: null };
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

afterEach(() => vi.unstubAllGlobals());

it("shows a failed approval load with a retry, and recovers without the investigation changing", async () => {
  let approvalReads = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) => {
      if (path === "/approvals/APR-1") {
        approvalReads += 1;
        return approvalReads === 1 ? json({ detail: "database unavailable" }, 503) : json(detail);
      }
      if (path.endsWith("/tool-calls")) return json([]);
      return json(waiting);
    }),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <SessionContext.Provider value={{ me: { user: "alice", role: "approver" }, signOut: () => {} }}>
        <MemoryRouter initialEntries={["/investigations/IQ-1"]}>
          <Routes>
            <Route path="/investigations/:id" element={<InvestigationPage />} />
          </Routes>
        </MemoryRouter>
      </SessionContext.Provider>
    </QueryClientProvider>,
  );

  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Couldn't load the approval: database unavailable");
  await userEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByText(/approval needed/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Approve" })).toBeInTheDocument();
});
