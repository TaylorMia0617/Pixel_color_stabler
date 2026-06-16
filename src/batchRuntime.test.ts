import { describe, expect, it } from "vitest";
import { runCancellableBatchQueue } from "./batchRuntime";
import { throwIfSignalAborted } from "./taskCleanup";
import type { BatchJob } from "./types";

const jobs: BatchJob[] = [
  { id: "a", name: "a.png", status: "queued", progress: 0 },
  { id: "b", name: "b.png", status: "queued", progress: 0 },
];

describe("runCancellableBatchQueue", () => {
  it("stops processing following jobs after an abort", async () => {
    const controller = new AbortController();
    const processed: string[] = [];
    const aborted: string[] = [];

    await runCancellableBatchQueue(jobs, controller.signal, {
      processJob: async (job, signal) => {
        processed.push(job.id);
        controller.abort();
        throwIfSignalAborted(signal);
      },
      onJobAbort: (job) => aborted.push(job.id),
    });

    expect(processed).toEqual(["a"]);
    expect(aborted).toEqual(["a"]);
  });

  it("continues after ordinary job failures", async () => {
    const controller = new AbortController();
    const processed: string[] = [];
    const failed: string[] = [];

    await runCancellableBatchQueue(jobs, controller.signal, {
      processJob: async (job) => {
        processed.push(job.id);
        if (job.id === "a") {
          throw new Error("bad image");
        }
      },
      onJobFailure: (job) => failed.push(job.id),
    });

    expect(processed).toEqual(["a", "b"]);
    expect(failed).toEqual(["a"]);
  });
});
