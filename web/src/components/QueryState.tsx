import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { describeError } from "../api/client";

/**
 * Loading, error-with-retry, or the content. A failed request must not look
 * like an empty result: an approval that failed to load is not "no approval".
 */
export function QueryState<T>({
  query,
  what,
  children,
}: {
  query: UseQueryResult<T>;
  what: string;
  children: (data: T) => ReactNode;
}) {
  if (query.isError) {
    return (
      <div role="alert" className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
        Couldn't load {what}: {describeError(query.error)}
        <button
          type="button"
          onClick={() => void query.refetch()}
          className="ml-3 rounded border border-red-300 bg-white px-2 py-0.5 text-xs"
        >
          Retry
        </button>
      </div>
    );
  }
  if (query.data === undefined) return <p className="text-sm text-slate-500">Loading {what}…</p>;
  return <>{children(query.data)}</>;
}
