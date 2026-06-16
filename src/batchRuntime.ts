import type { BatchJob } from "./types";
import { isAbortError, throwIfSignalAborted } from "./taskCleanup";

type BatchRuntimeHandlers = {
  processJob: (job: BatchJob, signal: AbortSignal) => Promise<void>;
  onJobAbort?: (job: BatchJob) => void;
  onJobFailure?: (job: BatchJob, error: unknown) => void;
};

export async function runCancellableBatchQueue(
  jobs: BatchJob[],
  signal: AbortSignal,
  handlers: BatchRuntimeHandlers,
): Promise<void> {
  for (const job of jobs) {
    if (job.status !== "queued") {
      continue;
    }

    try {
      throwIfSignalAborted(signal);
      await handlers.processJob(job, signal);
      throwIfSignalAborted(signal);
    } catch (error) {
      if (signal.aborted || isAbortError(error)) {
        handlers.onJobAbort?.(job);
        break;
      }

      handlers.onJobFailure?.(job, error);
    }
  }
}
