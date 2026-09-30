"use client";

/**
 * "The backend did not answer", rendered the same way everywhere it happens.
 *
 * This was written twice — once privately in `history-view.tsx` and again in
 * `compare-view.tsx` — which is the point at which the codebase's own rule kicks
 * in: `lib/format.ts` exists because four formatters had been copied into two
 * files, and `runs-table.tsx` explains why it did *not* abstract on the first
 * use. Two real callers is the threshold. Left copied, the two would drift in
 * exactly the copy that matters most — the "nothing is listening on the API port"
 * hint — so that a reader would get a different explanation depending on which
 * tab they were on when the backend was down.
 *
 * `ApiError.detail` is FastAPI's own sentence and is already written for a human,
 * so `message` is passed through rather than replaced. The `hint` is the one
 * addition, and it is per-caller because what the failed read needs from the
 * backend differs: the history needs the SQLite store, the comparison needs a
 * scenario that a solve can be run against.
 */

import { AlertTriangle, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";

export default function ApiFailure({
  title,
  message,
  onRetry,
  hint,
}: {
  title: string;
  message: string;
  onRetry: () => void;
  /** Why this particular read needs the API up. Omitted when nothing specific applies. */
  hint?: string;
}) {
  // A fetch that never reached a server reports itself as `TypeError: Failed to
  // fetch`; anything else got a response and is a different problem. Matching on
  // the text is crude, but the alternative is guessing from an error the platform
  // does not type, and the cost of a false negative is only a missing hint.
  const unreachable =
    message.includes("Failed to fetch") || message.includes("NetworkError");

  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border px-6 py-10 text-center">
      <span className="flex size-9 items-center justify-center rounded-full bg-danger/10 text-danger">
        <AlertTriangle className="size-4" aria-hidden />
      </span>
      <div>
        <p className="text-sm font-medium">{title}</p>
        <p className="mt-1 max-w-md text-xs break-words text-muted-foreground">{message}</p>
        {unreachable ? (
          <p className="mt-2 text-2xs text-muted-foreground">
            The API is expected on{" "}
            <code className="font-mono">
              {process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}
            </code>
            .{hint ? ` ${hint}` : null}
          </p>
        ) : null}
      </div>
      <Button variant="outline" size="sm" onClick={onRetry}>
        <RefreshCw className="size-3.5" aria-hidden />
        Try again
      </Button>
    </div>
  );
}
