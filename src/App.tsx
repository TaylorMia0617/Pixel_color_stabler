import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Download,
  FolderInput,
  ImagePlus,
  RotateCcw,
  Save,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  ZoomIn,
} from "lucide-react";
import { CanvasPreview } from "./CanvasPreview";
import {
  exportPng,
  loadImageBytesToDocument,
  loadImageToDocument,
  stabilizeImageAsync,
} from "./imageProcessing";
import { t } from "./i18n";
import { downloadBlobAsPng, isTauriRuntime, saveBlobAsPng } from "./tauriExport";
import { DEFAULT_APP_CONFIG, DEFAULT_SETTINGS, normalizeAppConfig } from "./appConfig";
import { collectBasketEntries, entriesToBatchJobs, filesToBatchJobs, removeBatchJob } from "./batch";
import { runCancellableBatchQueue } from "./batchRuntime";
import {
  abortActiveTaskControllers,
  isAbortError,
  throwIfSignalAborted,
} from "./taskCleanup";
import {
  loadConfig,
  pickFolder,
  pickImageFiles,
  readImageFile,
  saveConfig,
  savePngToDirectory,
  scanImageFolder,
} from "./tauriBackend";
import type {
  AppConfig,
  BatchJob,
  BatchStatus,
  ImageDocument,
  ProcessedImage,
  ProcessingProgress,
  ProcessingSettings,
} from "./types";

const batchStatusLabels: Record<BatchStatus, Parameters<typeof t>[0]> = {
  queued: "queued",
  processing: "jobProcessing",
  exported: "exported",
  failed: "failed",
  skipped: "skipped",
};

const PROCESSING_PRESETS: Array<{
  key: Parameters<typeof t>[0];
  settings: ProcessingSettings;
}> = [
  {
    key: "presetSoft",
    settings: {
      ...DEFAULT_SETTINGS,
      strength: 72,
      paletteSize: 8,
      lumaStrength: 22,
      chromaStrength: 82,
      edgeProtect: 75,
      familyArea: 180,
      labelArea: 56,
      speckArea: 20,
      labelFilterSize: 3,
      dirtyBlock: {
        ...DEFAULT_SETTINGS.dirtyBlock,
        repairStrength: 72,
        maxDirtyArea: 620,
        surroundDominance: 0.62,
      },
    },
  },
  {
    key: "presetStandard",
    settings: DEFAULT_SETTINGS,
  },
  {
    key: "presetStrong",
    settings: {
      ...DEFAULT_SETTINGS,
      strength: 88,
      paletteSize: 10,
      lumaStrength: 20,
      chromaStrength: 93,
      edgeProtect: 65,
      familyArea: 320,
      labelArea: 96,
      speckArea: 28,
      labelFilterSize: 5,
      dirtyBlock: {
        ...DEFAULT_SETTINGS.dirtyBlock,
        maxDirtyArea: 1200,
        surroundRadius: 4,
        surroundDominance: 0.54,
        repairStrength: 92,
      },
    },
  },
  {
    key: "presetFlat",
    settings: {
      ...DEFAULT_SETTINGS,
      cleanupMode: "flat",
      strength: 100,
      paletteSize: 6,
      lumaStrength: 5,
      chromaStrength: 100,
      edgeProtect: 85,
      familyArea: 260,
      labelArea: 64,
      speckArea: 16,
      labelFilterSize: 5,
      dirtyBlock: {
        ...DEFAULT_SETTINGS.dirtyBlock,
        repairStrength: 92,
      },
    },
  },
  {
    key: "presetDirtyOnly",
    settings: {
      ...DEFAULT_SETTINGS,
      processingMode: "dirtyOnly",
      strength: 0,
      lumaStrength: 0,
      chromaStrength: 0,
      dirtyBlock: {
        ...DEFAULT_SETTINGS.dirtyBlock,
        repairStrength: 85,
      },
    },
  },
];

export function App() {
  const [document, setDocument] = useState<ImageDocument | null>(null);
  const [processed, setProcessed] = useState<ProcessedImage | null>(null);
  const [settings, setSettings] = useState<ProcessingSettings>(DEFAULT_SETTINGS);
  const [appConfig, setAppConfig] = useState<AppConfig>(DEFAULT_APP_CONFIG);
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [batchJobs, setBatchJobs] = useState<BatchJob[]>([]);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [basketSeen, setBasketSeen] = useState<Set<string>>(new Set());
  const basketCandidates = useRef<Map<string, number>>(new Map());
  const processingControllerRef = useRef<AbortController | null>(null);
  const batchControllerRef = useRef<AbortController | null>(null);
  const isClosingRef = useRef(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState<ProcessingProgress | null>(null);
  const [processRunId, setProcessRunId] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const suggestedExportName = useMemo(() => {
    if (!document) {
      return t("stabilizedFileName");
    }

    return `${document.name.replace(/\.[^.]+$/, "")}-${t("stabilizedFileSuffix")}.png`;
  }, [document]);

  useEffect(() => {
    if (!isTauriRuntime()) {
      return;
    }

    loadConfig()
      .then((config) => {
        const normalized = normalizeAppConfig(config);
        setAppConfig(normalized);
        setSettings(normalized.settings);
      })
      .catch(() => setAppConfig(DEFAULT_APP_CONFIG));
  }, []);

  useEffect(() => {
    isClosingRef.current = false;

    const cancelActiveTasks = () => {
      abortActiveTaskControllers([processingControllerRef, batchControllerRef]);
    };
    return () => {
      isClosingRef.current = true;
      cancelActiveTasks();
    };
  }, []);

  useEffect(() => {
    if (!document) {
      setProcessed(null);
      setProgress(null);
      return;
    }

    const controller = new AbortController();
    processingControllerRef.current?.abort();
    processingControllerRef.current = controller;
    setIsProcessing(true);
    setProcessed(null);
    setProgress({ phase: "clustering", percent: 0 });

    stabilizeImageAsync(
      document,
      settings,
      (nextProgress) => setProgress(nextProgress),
      controller.signal,
    )
      .then((nextProcessed) => {
        setProcessed(nextProcessed);
        setProgress({ phase: "rendering", percent: 100 });
      })
      .catch((nextError) => {
        if (isAbortError(nextError)) {
          return;
        }

        setError(t("processingFailed"));
      })
      .finally(() => {
        if (processingControllerRef.current === controller) {
          processingControllerRef.current = null;
        }
        if (!controller.signal.aborted) {
          setIsProcessing(false);
        }
      });

    return () => {
      controller.abort();
      if (processingControllerRef.current === controller) {
        processingControllerRef.current = null;
      }
    };
  }, [document, settings, processRunId]);

  const importFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError(t("chooseImageFile"));
      return;
    }

    setIsLoading(true);
    setError(null);
    setNotice(null);

    try {
      const nextDocument = await loadImageToDocument(file);
      setDocument(nextDocument);
    } catch (nextError) {
      setError(t("loadImageFailed"));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        void importFile(file);
      }
      event.target.value = "";
    },
    [importFile],
  );

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      const files = Array.from(event.dataTransfer.files);
      if (mode === "batch" && files.length > 1) {
        setBatchJobs((current) => [...current, ...filesToBatchJobs(files, current)]);
        return;
      }

      const file = files[0];
      if (file) {
        void importFile(file);
      }
    },
    [importFile, mode],
  );

  const exportProcessedImage = useCallback(async () => {
    if (!processed) {
      return;
    }

    setError(null);
    setNotice(null);

    try {
      const blob = await exportPng(processed.imageData);
      if (isTauriRuntime()) {
        await saveBlobAsPng(blob, suggestedExportName);
      } else {
        downloadBlobAsPng(blob, suggestedExportName);
      }
    } catch (nextError) {
      const detail = nextError instanceof Error ? nextError.message : String(nextError);
      setError(`${t("exportFailed")} ${detail}`);
    }
  }, [processed, suggestedExportName]);

  const saveDefaults = useCallback(async () => {
    const nextConfig = { ...appConfig, settings };
    setAppConfig(nextConfig);
    setError(null);
    setNotice(null);

    try {
      if (isTauriRuntime()) {
        await saveConfig(nextConfig);
      }
      setNotice(t("defaultsSaved"));
    } catch (nextError) {
      const detail = nextError instanceof Error ? nextError.message : String(nextError);
      setError(`${t("defaultsSaveFailed")} ${detail}`);
    }
  }, [appConfig, settings]);

  const restoreDefaults = useCallback(() => {
    setSettings(appConfig.settings);
    setNotice(t("defaultsRestored"));
  }, [appConfig.settings]);

  const chooseDefaultExportFolder = useCallback(async () => {
    if (!isTauriRuntime()) {
      return;
    }
    const folder = await pickFolder();
    if (!folder) {
      return;
    }
    const nextConfig = { ...appConfig, defaultExportFolder: folder };
    setAppConfig(nextConfig);
    await saveConfig(nextConfig);
  }, [appConfig]);

  const chooseBasketFolder = useCallback(async () => {
    if (!isTauriRuntime()) {
      return;
    }
    const folder = await pickFolder();
    if (!folder) {
      return;
    }
    const nextConfig = { ...appConfig, basketFolder: folder };
    setAppConfig(nextConfig);
    setBasketSeen(new Set());
    basketCandidates.current.clear();
    await saveConfig(nextConfig);
  }, [appConfig]);

  const toggleBasketAutoScan = useCallback(async () => {
    const nextConfig = { ...appConfig, basketAutoScan: !appConfig.basketAutoScan };
    setAppConfig(nextConfig);
    if (isTauriRuntime()) {
      await saveConfig(nextConfig);
    }
  }, [appConfig]);

  const addFilesToBatch = useCallback(async () => {
    if (!isTauriRuntime()) {
      return;
    }
    const entries = await pickImageFiles();
    setBatchJobs((current) => [...current, ...entriesToBatchJobs(entries, current)]);
  }, []);

  const addFolderToBatch = useCallback(async () => {
    if (!isTauriRuntime()) {
      return;
    }
    const folder = await pickFolder();
    if (!folder) {
      return;
    }
    const entries = await scanImageFolder(folder);
    setBatchJobs((current) => [...current, ...entriesToBatchJobs(entries, current)]);
  }, []);

  const scanBasket = useCallback(async (requireStableSize = false) => {
    if (isClosingRef.current || !isTauriRuntime() || !appConfig.basketFolder) {
      return;
    }
    const entries = await scanImageFolder(appConfig.basketFolder);
    const stableEntries = collectBasketEntries(
      entries,
      basketSeen,
      basketCandidates.current,
      requireStableSize,
    );
    if (!stableEntries.length) {
      return;
    }

    setBasketSeen((current) => {
      const next = new Set(current);
      for (const entry of stableEntries) {
        next.add(entry.path);
      }
      return next;
    });
    setBatchJobs((current) => [...current, ...entriesToBatchJobs(stableEntries, current)]);
  }, [appConfig.basketFolder, basketSeen]);

  useEffect(() => {
    if (!appConfig.basketAutoScan || !appConfig.basketFolder) {
      return;
    }

    const interval = window.setInterval(() => {
      void scanBasket(true);
    }, 5000);
    return () => window.clearInterval(interval);
  }, [appConfig.basketAutoScan, appConfig.basketFolder, scanBasket]);

  const deleteBatchJob = useCallback((id: string) => {
    setBatchJobs((current) => removeBatchJob(current, id));
  }, []);

  const runBatch = useCallback(async () => {
    if (isBatchRunning) {
      return;
    }

    batchControllerRef.current?.abort();
    const controller = new AbortController();
    batchControllerRef.current = controller;
    setIsBatchRunning(true);

    await runCancellableBatchQueue(batchJobs, controller.signal, {
      processJob: async (job, signal) => {
        setBatchJobs((current) =>
          current.map((item) =>
            item.id === job.id ? { ...item, status: "processing", progress: 5, error: undefined } : item,
          ),
        );

        const documentForJob = job.file
          ? await loadImageToDocument(job.file)
          : await loadImageBytesToDocument(await readImageFile(job.path!), job.name);
        throwIfSignalAborted(signal);

        const result = await stabilizeImageAsync(documentForJob, settings, (nextProgress) => {
          setBatchJobs((current) =>
            current.map((item) =>
              item.id === job.id ? { ...item, progress: nextProgress.percent } : item,
            ),
          );
        }, signal);
        throwIfSignalAborted(signal);

        const blob = await exportPng(result.imageData);
        throwIfSignalAborted(signal);

        const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
        let outputPath = "";
        if (isTauriRuntime() && job.path) {
          outputPath = await savePngToDirectory({
                outputDir: appConfig.defaultExportFolder,
                sourcePath: job.path,
                sourceName: job.name,
                suffix: t("stabilizedFileSuffix"),
                bytes,
              });
        } else {
          downloadBlobAsPng(blob, `${job.name.replace(/\.[^.]+$/, "")}-${t("stabilizedFileSuffix")}.png`);
          outputPath = t("browserDownload");
        }
        const changedPercent = Math.round(
          (result.stats.changedPixelCount / result.stats.totalPixelCount) * 100,
        );

        setBatchJobs((current) =>
          current.map((item) =>
            item.id === job.id
              ? {
                  ...item,
                  status: "exported",
                  progress: 100,
                  outputPath: outputPath || t("browserDownload"),
                  dimensions: `${documentForJob.width} x ${documentForJob.height}`,
                  changedPercent,
                }
              : item,
          ),
        );
      },
      onJobAbort: (job) => {
        setBatchJobs((current) =>
          current.map((item) =>
            item.id === job.id ? { ...item, status: "skipped", progress: 0 } : item,
          ),
        );
      },
      onJobFailure: (job, nextError) => {
        const detail = nextError instanceof Error ? nextError.message : String(nextError);
        setBatchJobs((current) =>
          current.map((item) =>
            item.id === job.id ? { ...item, status: "failed", error: detail, progress: 0 } : item,
          ),
        );
      },
    });

    if (batchControllerRef.current === controller) {
      batchControllerRef.current = null;
    }
    setIsBatchRunning(false);
  }, [appConfig.defaultExportFolder, batchJobs, isBatchRunning, settings]);

  const changedPercent = processed
    ? Math.round((processed.stats.changedPixelCount / processed.stats.totalPixelCount) * 100)
    : 0;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>{t("appTitle")}</h1>
          <p>{t("appSubtitle")}</p>
        </div>
        <label
          className="import-button"
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
        >
          <ImagePlus size={18} aria-hidden="true" />
          {t("importImage")}
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={handleInputChange}
          />
        </label>
      </header>

      <div className="mode-tabs" role="tablist" aria-label={t("modeTabs")}>
        <button
          type="button"
          className={mode === "single" ? "active" : ""}
          onClick={() => setMode("single")}
        >
          {t("singleMode")}
        </button>
        <button
          type="button"
          className={mode === "batch" ? "active" : ""}
          onClick={() => setMode("batch")}
        >
          {t("batchMode")}
        </button>
      </div>

      <section className="workspace">
        <aside className="controls" aria-label={t("controls")}>
          <div className="control-title">
            <SlidersHorizontal size={18} aria-hidden="true" />
            {t("controls")}
          </div>

          <div className="preset-grid" aria-label={t("presetStandard")}>
            {PROCESSING_PRESETS.map((preset) => (
              <button
                type="button"
                className="preset-button"
                key={preset.key}
                onClick={() => setSettings(preset.settings)}
                disabled={isBatchRunning}
              >
                {t(preset.key)}
              </button>
            ))}
          </div>

          <div className="mode-choice" aria-label={t("processingMode")}>
            <button
              type="button"
              className={settings.processingMode === "palette" ? "active" : ""}
              onClick={() =>
                setSettings((current) => ({
                  ...current,
                  processingMode: "palette",
                }))
              }
              disabled={isBatchRunning}
            >
              {t("processingModePalette")}
            </button>
            <button
              type="button"
              className={settings.processingMode === "dirtyOnly" ? "active" : ""}
              onClick={() =>
                setSettings((current) => ({
                  ...current,
                  processingMode: "dirtyOnly",
                }))
              }
              disabled={isBatchRunning}
            >
              {t("processingModeDirtyOnly")}
            </button>
            <button
              type="button"
              className={settings.processingMode === "dirtyThenPalette" ? "active" : ""}
              onClick={() =>
                setSettings((current) => ({
                  ...current,
                  processingMode: "dirtyThenPalette",
                }))
              }
              disabled={isBatchRunning}
            >
              {t("processingModeDirtyThenPalette")}
            </button>
          </div>

          <div className="segmented-control" aria-label={t("cleanupMode")}>
            <button
              type="button"
              className={settings.cleanupMode === "shaded" ? "active" : ""}
              onClick={() =>
                setSettings((current) => ({
                  ...current,
                  cleanupMode: "shaded",
                }))
              }
              disabled={isBatchRunning}
            >
              {t("cleanupModeShaded")}
            </button>
            <button
              type="button"
              className={settings.cleanupMode === "flat" ? "active" : ""}
              onClick={() =>
                setSettings((current) => ({
                  ...current,
                  cleanupMode: "flat",
                }))
              }
              disabled={isBatchRunning}
            >
              {t("cleanupModeFlat")}
            </button>
          </div>

          <label className="slider-row">
            <span>
              {t("paletteSize")}
              <strong>{settings.paletteSize}</strong>
            </span>
            <input
              type="range"
              min="1"
              max="16"
              value={settings.paletteSize}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  paletteSize: Number(event.target.value),
                }))
              }
              disabled={isBatchRunning}
            />
          </label>

          <label className="slider-row">
            <span>
              {t("strength")}
              <strong>{settings.strength}%</strong>
            </span>
            <input
              type="range"
              min="0"
              max="100"
              value={settings.strength}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  strength: Number(event.target.value),
                }))
              }
              disabled={isBatchRunning}
            />
          </label>

          <label className="slider-row">
            <span>
              {t("lumaStrength")}
              <strong>{settings.lumaStrength}%</strong>
            </span>
            <input
              type="range"
              min="0"
              max="100"
              value={settings.lumaStrength}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  lumaStrength: Number(event.target.value),
                }))
              }
              disabled={isBatchRunning}
            />
          </label>

          <label className="slider-row">
            <span>
              {t("chromaStrength")}
              <strong>{settings.chromaStrength}%</strong>
            </span>
            <input
              type="range"
              min="0"
              max="100"
              value={settings.chromaStrength}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  chromaStrength: Number(event.target.value),
                }))
              }
              disabled={isBatchRunning}
            />
          </label>

          <label className="slider-row">
            <span>
              {t("edgeProtect")}
              <strong>{settings.edgeProtect}%</strong>
            </span>
            <input
              type="range"
              min="0"
              max="100"
              value={settings.edgeProtect}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  edgeProtect: Number(event.target.value),
                }))
              }
              disabled={isBatchRunning}
            />
          </label>

          <label className="slider-row">
            <span>
              {t("zoom")}
              <strong>{Math.round(zoom * 100)}%</strong>
            </span>
            <input
              type="range"
              min="1"
              max="8"
              step="0.25"
              value={zoom}
              onChange={(event) => setZoom(Number(event.target.value))}
              disabled={!document}
            />
          </label>

          {isProcessing && progress && (
            <div className="progress-card" aria-live="polite">
              <div className="progress-copy">
                <span>{t("processing")}</span>
                <strong>{progress.percent}%</strong>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
              </div>
              <div className="progress-phase">
                {progress.phase === "clustering" ? t("clustering") : t("rendering")}
              </div>
            </div>
          )}

          <div className="button-row">
            <button
              type="button"
              className="secondary-button"
              onClick={() => {
                setSettings(DEFAULT_SETTINGS);
                setZoom(1);
              }}
              disabled={isBatchRunning}
              title={t("resetControls")}
            >
              <RotateCcw size={17} aria-hidden="true" />
            </button>
            <button
              type="button"
              className="primary-button"
              onClick={() => setProcessRunId((current) => current + 1)}
              disabled={!document || isProcessing}
            >
              <Sparkles size={17} aria-hidden="true" />
              {t("reprocess")}
            </button>
          </div>

          <div className="settings-actions">
            <button type="button" className="utility-button" onClick={() => void saveDefaults()}>
              <Save size={16} aria-hidden="true" />
              {t("saveDefaults")}
            </button>
            <button type="button" className="utility-button" onClick={restoreDefaults}>
              <RotateCcw size={16} aria-hidden="true" />
              {t("restoreDefaults")}
            </button>
            <button
              type="button"
              className="utility-button"
              onClick={() => void chooseDefaultExportFolder()}
            >
              <FolderInput size={16} aria-hidden="true" />
              {t("defaultExportFolder")}
            </button>
            <div className="path-hint">{appConfig.defaultExportFolder || t("sourceOutFallback")}</div>
          </div>

          {mode === "batch" && (
            <div className="basket-panel">
              <button
                type="button"
                className="utility-button"
                onClick={() => void chooseBasketFolder()}
              >
                <FolderInput size={16} aria-hidden="true" />
                {t("basketFolder")}
              </button>
              <div className="path-hint">{appConfig.basketFolder || t("notSet")}</div>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={appConfig.basketAutoScan}
                  onChange={() => void toggleBasketAutoScan()}
                />
                {t("basketAutoScan")}
              </label>
              <button type="button" className="utility-button" onClick={() => void scanBasket(false)}>
                {t("scanBasket")}
              </button>
            </div>
          )}

          <button
            type="button"
            className="zoom-button"
            onClick={() => setZoom(1)}
            disabled={!document || zoom === 1}
          >
            <ZoomIn size={17} aria-hidden="true" />
            {t("resetZoom")}
          </button>

          <button
            type="button"
            className="export-button"
            onClick={() => void exportProcessedImage()}
            disabled={!processed || isProcessing}
          >
            <Download size={17} aria-hidden="true" />
            {t("exportPng")}
          </button>

          {processed && (
            <dl className="stats">
              <div>
                <dt>{t("changed")}</dt>
                <dd>{changedPercent}%</dd>
              </div>
              <div>
                <dt>{t("paletteColors")}</dt>
                <dd>{processed.stats.clusterCount.toLocaleString()}</dd>
              </div>
              <div>
                <dt>{t("size")}</dt>
                <dd>
                  {document?.width} x {document?.height}
                </dd>
              </div>
            </dl>
          )}
        </aside>

        <section className={mode === "single" ? "preview-grid" : "batch-workspace"}>
          {mode === "single" && !document && (
            <label
              className="drop-zone"
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              <ImagePlus size={36} aria-hidden="true" />
              <span>{isLoading ? t("loadingImage") : t("dropImage")}</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={handleInputChange}
              />
            </label>
          )}

          {mode === "single" && document && (
            <>
              <CanvasPreview imageData={document.original} label={t("original")} zoom={zoom} />
              <CanvasPreview imageData={processed?.imageData} label={t("stabilized")} zoom={zoom} />
            </>
          )}

          {mode === "batch" && (
            <section className="batch-panel" aria-label={t("batchMode")}>
              <div className="batch-toolbar">
                <button type="button" className="primary-button" onClick={() => void addFilesToBatch()}>
                  {t("addFiles")}
                </button>
                <button type="button" className="utility-button" onClick={() => void addFolderToBatch()}>
                  {t("addFolder")}
                </button>
                <button
                  type="button"
                  className="export-button"
                  onClick={() => void runBatch()}
                  disabled={isBatchRunning || !batchJobs.some((job) => job.status === "queued")}
                >
                  {isBatchRunning ? t("batchRunning") : t("startBatch")}
                </button>
              </div>
              <div className="batch-summary">
                {t("batchSummary")
                  .replace("{total}", String(batchJobs.length))
                  .replace(
                    "{done}",
                    String(batchJobs.filter((job) => job.status === "exported").length),
                  )
                  .replace(
                    "{failed}",
                    String(batchJobs.filter((job) => job.status === "failed").length),
                  )}
              </div>
              <div className="job-list">
                {batchJobs.length === 0 && <div className="empty-jobs">{t("emptyBatch")}</div>}
                {batchJobs.map((job) => (
                  <article className="job-row" key={job.id}>
                    <div>
                      <strong>{job.name}</strong>
                      <span>{job.outputPath || job.error || job.path || t("queued")}</span>
                    </div>
                    <div className={`job-status ${job.status}`}>{t(batchStatusLabels[job.status])}</div>
                    <button
                      type="button"
                      className="job-delete"
                      onClick={() => deleteBatchJob(job.id)}
                      disabled={job.status === "processing"}
                      title={t("deleteJob")}
                      aria-label={`${t("deleteJob")} ${job.name}`}
                    >
                      <Trash2 size={16} aria-hidden="true" />
                    </button>
                    <div className="job-progress">
                      <div style={{ width: `${job.progress}%` }} />
                    </div>
                    <small>
                      {job.dimensions || ""}
                      {job.changedPercent !== undefined ? ` · ${t("changed")} ${job.changedPercent}%` : ""}
                    </small>
                  </article>
                ))}
              </div>
            </section>
          )}
        </section>
      </section>

      {error && <div className="toast" role="alert">{error}</div>}
      {notice && <div className="toast success" role="status">{notice}</div>}
    </main>
  );
}
