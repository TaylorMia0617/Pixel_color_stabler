import { describe, expect, it } from "vitest";
import {
  abortActiveTaskControllers,
  abortControllerRef,
  isAbortError,
  throwIfSignalAborted,
} from "./taskCleanup";

describe("task cleanup helpers", () => {
  it("aborts and clears a controller ref", () => {
    const controller = new AbortController();
    const ref = { current: controller };

    abortControllerRef(ref);

    expect(controller.signal.aborted).toBe(true);
    expect(ref.current).toBeNull();
  });

  it("aborts all active task controllers", () => {
    const first = new AbortController();
    const second = new AbortController();

    abortActiveTaskControllers([{ current: first }, { current: second }]);

    expect(first.signal.aborted).toBe(true);
    expect(second.signal.aborted).toBe(true);
  });

  it("identifies abort errors from aborted signals", () => {
    const controller = new AbortController();
    controller.abort();

    expect(() => throwIfSignalAborted(controller.signal)).toThrow(DOMException);

    try {
      throwIfSignalAborted(controller.signal);
    } catch (error) {
      expect(isAbortError(error)).toBe(true);
    }
  });
});
