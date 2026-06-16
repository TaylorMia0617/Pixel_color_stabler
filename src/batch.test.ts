import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS, normalizeAppConfig } from "./appConfig";
import {
  collectBasketEntries,
  entriesToBatchJobs,
  filesToBatchJobs,
  removeBatchJob,
} from "./batch";

describe("batch helpers", () => {
  it("normalizes missing config fields to safe defaults", () => {
    const config = normalizeAppConfig({
      settings: { paletteSize: 4 } as typeof DEFAULT_SETTINGS,
      basketAutoScan: true,
    });

    expect(config.settings.paletteSize).toBe(4);
    expect(config.settings.strength).toBe(DEFAULT_SETTINGS.strength);
    expect(config.defaultExportFolder).toBeNull();
    expect(config.basketFolder).toBeNull();
    expect(config.basketAutoScan).toBe(true);
  });

  it("creates queued jobs from scanned file entries and skips existing paths", () => {
    const jobs = entriesToBatchJobs(
      [
        { path: "C:/in/a.png", name: "a.png", sourceDir: "C:/in", size: 10 },
        { path: "C:/in/b.jpg", name: "b.jpg", sourceDir: "C:/in", size: 20 },
      ],
      [{ id: "old", path: "C:/in/a.png", name: "a.png", status: "exported", progress: 100 }],
    );

    expect(jobs).toHaveLength(1);
    expect(jobs[0]).toMatchObject({ name: "b.jpg", status: "queued", progress: 0 });
  });

  it("creates queued jobs from image files only", () => {
    const image = new File(["a"], "a.png", { type: "image/png", lastModified: 1 });
    const text = new File(["b"], "b.txt", { type: "text/plain", lastModified: 1 });

    const jobs = filesToBatchJobs([image, text]);

    expect(jobs).toHaveLength(1);
    expect(jobs[0].name).toBe("a.png");
  });

  it("removes a batch job by id", () => {
    const jobs = [
      { id: "a", name: "a.png", status: "queued" as const, progress: 0 },
      { id: "b", name: "b.png", status: "failed" as const, progress: 0 },
    ];

    expect(removeBatchJob(jobs, "a")).toEqual([jobs[1]]);
  });

  it("queues Basket manual scans immediately and waits for stable size in auto scans", () => {
    const entry = { path: "C:/basket/a.png", name: "a.png", sourceDir: "C:/basket", size: 10 };
    const seen = new Set<string>();
    const candidates = new Map<string, number>();

    expect(collectBasketEntries([entry], seen, candidates, false)).toEqual([entry]);

    candidates.clear();
    expect(collectBasketEntries([entry], seen, candidates, true)).toEqual([]);
    expect(collectBasketEntries([entry], seen, candidates, true)).toEqual([entry]);
  });
});
