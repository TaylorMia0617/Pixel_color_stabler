import type { BatchJob, ImageFileEntry } from "./types";

export function entriesToBatchJobs(entries: ImageFileEntry[], existing: BatchJob[] = []): BatchJob[] {
  const seen = new Set(existing.map((job) => job.path || job.name));
  return entries
    .filter((entry) => !seen.has(entry.path))
    .map((entry) => ({
      id: `${entry.path}:${entry.size}`,
      path: entry.path,
      name: entry.name,
      sourceDir: entry.sourceDir,
      status: "queued",
      progress: 0,
    }));
}

export function filesToBatchJobs(files: File[], existing: BatchJob[] = []): BatchJob[] {
  const seen = new Set(existing.map((job) => job.name));
  return files
    .filter((file) => file.type.startsWith("image/") && !seen.has(file.name))
    .map((file) => ({
      id: `${file.name}:${file.size}:${file.lastModified}`,
      name: file.name,
      file,
      status: "queued",
      progress: 0,
    }));
}

export function removeBatchJob(jobs: BatchJob[], id: string): BatchJob[] {
  return jobs.filter((job) => job.id !== id);
}

export function collectBasketEntries(
  entries: ImageFileEntry[],
  seen: Set<string>,
  candidateSizes: Map<string, number>,
  requireStableSize: boolean,
): ImageFileEntry[] {
  return entries.filter((entry) => {
    if (seen.has(entry.path)) {
      return false;
    }

    if (!requireStableSize) {
      candidateSizes.set(entry.path, entry.size);
      return true;
    }

    const previousSize = candidateSizes.get(entry.path);
    candidateSizes.set(entry.path, entry.size);
    return previousSize === entry.size;
  });
}
