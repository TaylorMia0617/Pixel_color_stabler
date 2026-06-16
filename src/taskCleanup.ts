export type ControllerRef = {
  current: AbortController | null;
};

export function abortControllerRef(ref: ControllerRef): void {
  ref.current?.abort();
  ref.current = null;
}

export function abortActiveTaskControllers(refs: ControllerRef[]): void {
  for (const ref of refs) {
    abortControllerRef(ref);
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export function throwIfSignalAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Task aborted.", "AbortError");
  }
}
