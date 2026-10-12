import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "./client";
import { readStream } from "./stream";
import type { Investigation } from "./types";
import { ACTIVE } from "../lib/format";

export interface LiveOptions {
  /** Polling interval when the stream is unavailable. */
  pollMs?: number;
  /** Polling interval while an approval waits: someone else may decide it. */
  waitingPollMs?: number;
}

/**
 * One investigation, kept live. While it is queued or running, the page follows
 * the server's event stream and writes each snapshot into the query cache. A
 * stream that fails, or closes before its `end` event, switches to polling.
 * While an approval waits, the page polls slowly, since a decision can come
 * from another browser. Whatever the transport, a changed investigation also
 * refreshes the panels that are fetched separately.
 */
export function useInvestigation(id: string, { pollMs = 3000, waitingPollMs = 5000 }: LiveOptions = {}) {
  const queryClient = useQueryClient();
  const [streamFailed, setStreamFailed] = useState(false);
  const query = useQuery({
    queryKey: ["investigation", id],
    queryFn: () => api.investigation(id),
    refetchOnWindowFocus: true,
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      if (!status) return false;
      if (ACTIVE.has(status)) return streamFailed ? pollMs : false;
      return status === "awaiting_approval" ? waitingPollMs : false;
    },
  });
  const data = query.data;
  const active = data ? ACTIVE.has(data.status) : false;

  useEffect(() => {
    if (!active || streamFailed) return;
    const controller = new AbortController();
    let ended = false;
    readStream(
      `/investigations/${encodeURIComponent(id)}/stream`,
      (message) => {
        if (message.event === "end") ended = true;
        if (message.event !== "snapshot") return;
        queryClient.setQueryData<Investigation>(["investigation", id], JSON.parse(message.data));
      },
      controller.signal,
    )
      .then(() => {
        // Closed without `end`: the run may still be going. Poll instead.
        if (!ended && !controller.signal.aborted) setStreamFailed(true);
      })
      .catch(() => {
        if (!controller.signal.aborted) setStreamFailed(true);
      });
    return () => controller.abort();
  }, [id, active, streamFailed, queryClient]);

  // After a fallback, try the stream again the next time the run becomes active.
  useEffect(() => {
    if (!active && streamFailed) setStreamFailed(false);
  }, [active, streamFailed]);

  // Panels fetched separately follow the investigation, from stream or poll alike.
  const version = data ? `${data.updated_at}|${data.status}|${data.tool_call_count}|${data.approvals.length}` : "";
  useEffect(() => {
    if (!version) return;
    void queryClient.invalidateQueries({ queryKey: ["tool-calls", id] });
    void queryClient.invalidateQueries({ queryKey: ["audit", id] });
    void queryClient.invalidateQueries({ queryKey: ["approval"] });
  }, [id, version, queryClient]);

  return query;
}
