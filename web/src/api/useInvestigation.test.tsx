import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import type { Investigation } from "./types";
import { useInvestigation } from "./useInvestigation";

function investigation(over: Partial<Investigation>): Investigation {
  return {
    id: "IQ-1",
    user_query: "q",
    status: "running",
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
    approvals: [],
    actions: [],
    tool_call_count: 0,
    ...over,
  };
}

const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });

/** fetch stub: investigation reads return the next state in `states` (the last one repeats). */
function serve(states: Investigation[], stream: () => Response) {
  const calls: string[] = [];
  let read = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) => {
      calls.push(path);
      if (path.endsWith("/stream")) return stream();
      if (path.endsWith("/tool-calls")) return json([]);
      const state = states[Math.min(read, states.length - 1)];
      read += 1;
      return json(state);
    }),
  );
  return calls;
}

function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("useInvestigation", () => {
  it("follows the stream: each snapshot updates the investigation until `end`", async () => {
    const running = investigation({ status: "running", events: [{ node: "intake", duration_ms: 5, at: 1 }] });
    const settled = investigation({
      status: "awaiting_approval",
      events: [
        { node: "intake", duration_ms: 5, at: 1 },
        { node: "propose", duration_ms: 9, at: 2 },
      ],
    });
    let streams = 0;
    serve([investigation({ status: "queued" })], () => {
      streams += 1;
      const body = new ReadableStream({
        async start(controller) {
          const enc = new TextEncoder();
          controller.enqueue(enc.encode(`event: snapshot\ndata: ${JSON.stringify(running)}\n\n`));
          await new Promise((r) => setTimeout(r, 30));
          controller.enqueue(enc.encode(`event: snapshot\ndata: ${JSON.stringify(settled)}\n\nevent: end\ndata: awaiting_approval\n\n`));
          controller.close();
        },
      });
      return new Response(body);
    });
    const seen: string[] = [];   // every state rendered, in order
    const { result } = renderHook(
      () => {
        const q = useInvestigation("IQ-1", { pollMs: 10_000, waitingPollMs: 10_000 });
        if (q.data) seen.push(`${q.data.status}:${q.data.events.map((e) => e.node).join(",")}`);
        return q;
      },
      { wrapper: wrapper() },
    );
    await waitFor(() => expect(result.current.data?.status).toBe("awaiting_approval"));
    const distinct = seen.filter((v, i) => v !== seen[i - 1]);
    expect(distinct).toEqual(["queued:", "running:intake", "awaiting_approval:intake,propose"]);
    expect(streams).toBe(1);
  });

  it("polls when the stream closes before its end event", async () => {
    const calls = serve(
      [investigation({ status: "running" }), investigation({ status: "complete", updated_at: "2026-10-12T00:01:00Z" })],
      // A snapshot, then the connection drops: no `end` event.
      () => new Response(`event: snapshot\ndata: ${JSON.stringify(investigation({ status: "running" }))}\n\n`),
    );
    const { result } = renderHook(() => useInvestigation("IQ-1", { pollMs: 20 }), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.data?.status).toBe("complete"), { timeout: 2000 });
    expect(calls.filter((c) => c.endsWith("/stream"))).toHaveLength(1);
  });

  it("polls while an approval waits, so another approver's decision shows up", async () => {
    serve(
      [
        investigation({ status: "awaiting_approval" }),
        investigation({ status: "awaiting_approval" }),
        investigation({ status: "complete", updated_at: "2026-10-12T00:02:00Z" }),
      ],
      () => new Response("", { status: 500 }),
    );
    const { result } = renderHook(() => useInvestigation("IQ-1", { waitingPollMs: 20 }), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.data?.status).toBe("complete"), { timeout: 2000 });
  });

  it("refreshes separately fetched panels when the investigation changes, whatever the transport", async () => {
    // Tool calls differ before and after the run completes; only a refresh
    // triggered by the change can show the completed run's calls.
    let completedServed = false;
    let read = 0;
    const states = [
      investigation({ status: "running", tool_call_count: 0 }),
      investigation({ status: "complete", tool_call_count: 1, updated_at: "2026-10-12T00:03:00Z" }),
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (path: string) => {
        if (path.endsWith("/stream")) return new Response("", { status: 500 });
        if (path.endsWith("/tool-calls")) return json(completedServed ? [{ tool_name: "get_service_logs" }] : []);
        const state = states[Math.min(read, states.length - 1)]!;
        read += 1;
        if (state.status === "complete") completedServed = true;
        return json(state);
      }),
    );
    const { result } = renderHook(
      () => {
        const inv = useInvestigation("IQ-1", { pollMs: 20 });
        const tools = useQuery({
          queryKey: ["tool-calls", "IQ-1"],
          queryFn: async () => (await fetch("/investigations/IQ-1/tool-calls")).json() as Promise<unknown[]>,
        });
        return { inv, tools };
      },
      { wrapper: wrapper() },
    );
    await waitFor(() => expect(result.current.inv.data?.status).toBe("complete"), { timeout: 2000 });
    await waitFor(() => expect(result.current.tools.data).toEqual([{ tool_name: "get_service_logs" }]), {
      timeout: 2000,
    });
  });
});
