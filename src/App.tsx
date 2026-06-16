import { useCallback, useEffect, useMemo, useState } from "react";
import { Download, ImagePlus, RotateCcw, SlidersHorizontal, Sparkles, ZoomIn } from "lucide-react";
import { CanvasPreview } from "./CanvasPreview";
import { exportPng, loadImageToDocument, stabilizeImageAsync } from "./imageProcessing";
import { t } from "./i18n";
import { downloadBlobAsPng, isTauriRuntime, saveBlobAsPng } from "./tauriExport";
import type { ImageDocument, ProcessedImage, ProcessingProgress, ProcessingSettings } from "./types";

const DEFAULT_SETTINGS: ProcessingSettings = {
  tolerance: 28,
  strength: 100,
};

export function App() {
  const [document, setDocument] = useState<ImageDocument | null>(null);
  const [processed, setProcessed] = useState<ProcessedImage | null>(null);
  const [settings, setSettings] = useState<ProcessingSettings>(DEFAULT_SETTINGS);
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState<ProcessingProgress | null>(null);
  const [processRunId, setProcessRunId] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [error, setError] = useState<string | null>(null);

  const suggestedExportName = useMemo(() => {
    if (!document) {
      return t("stabilizedFileName");
    }

    return `${document.name.replace(/\.[^.]+$/, "")}-降噪.png`;
  }, [document]);

  useEffect(() => {
    if (!document) {
      setProcessed(null);
      setProgress(null);
      return;
    }

    const controller = new AbortController();
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
        if (nextError instanceof DOMException && nextError.name === "AbortError") {
          return;
        }

        setError(t("processingFailed"));
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsProcessing(false);
        }
      });

    return () => controller.abort();
  }, [document, settings, processRunId]);

  const importFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError(t("chooseImageFile"));
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const nextDocument = await loadImageToDocument(file);
      setDocument(nextDocument);
      setSettings(DEFAULT_SETTINGS);
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
      const file = event.dataTransfer.files[0];
      if (file) {
        void importFile(file);
      }
    },
    [importFile],
  );

  const exportProcessedImage = useCallback(async () => {
    if (!processed) {
      return;
    }

    setError(null);

    try {
      const blob = await exportPng(processed.imageData);
      if (isTauriRuntime()) {
        await saveBlobAsPng(blob, suggestedExportName);
      } else {
        downloadBlobAsPng(blob, suggestedExportName);
      }
    } catch (nextError) {
      setError(t("exportFailed"));
    }
  }, [processed, suggestedExportName]);

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

      <section className="workspace">
        <aside className="controls" aria-label={t("controls")}>
          <div className="control-title">
            <SlidersHorizontal size={18} aria-hidden="true" />
            {t("controls")}
          </div>

          <label className="slider-row">
            <span>
              {t("tolerance")}
              <strong>{settings.tolerance}</strong>
            </span>
            <input
              type="range"
              min="0"
              max="80"
              value={settings.tolerance}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  tolerance: Number(event.target.value),
                }))
              }
              disabled={!document || isProcessing}
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
              disabled={!document || isProcessing}
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
              disabled={!document || isProcessing}
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
                <dt>{t("clusters")}</dt>
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

        <section className="preview-grid">
          {!document && (
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

          {document && (
            <>
              <CanvasPreview imageData={document.original} label={t("original")} zoom={zoom} />
              <CanvasPreview imageData={processed?.imageData} label={t("stabilized")} zoom={zoom} />
            </>
          )}
        </section>
      </section>

      {error && <div className="toast" role="alert">{error}</div>}
    </main>
  );
}
