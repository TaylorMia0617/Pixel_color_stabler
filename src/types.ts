export type RgbColor = {
  r: number;
  g: number;
  b: number;
};

export type ImageDocument = {
  name: string;
  width: number;
  height: number;
  original: ImageData;
};

export type ProcessingSettings = {
  strength: number;
  paletteSize: number;
  lumaStrength: number;
  chromaStrength: number;
  edgeProtect: number;
};

export type AppConfig = {
  settings: ProcessingSettings;
  defaultExportFolder: string | null;
  basketFolder: string | null;
  basketAutoScan: boolean;
};

export type ImageFileEntry = {
  path: string;
  name: string;
  sourceDir: string;
  size: number;
};

export type BatchStatus = "queued" | "processing" | "exported" | "failed" | "skipped";

export type BatchJob = {
  id: string;
  path?: string;
  name: string;
  sourceDir?: string;
  file?: File;
  status: BatchStatus;
  progress: number;
  outputPath?: string;
  error?: string;
  dimensions?: string;
  changedPercent?: number;
};

export type ProcessedImage = {
  imageData: ImageData;
  stats: ProcessingStats;
};

export type ProcessingProgress = {
  phase: "clustering" | "rendering";
  percent: number;
};

export type ProcessingStats = {
  changedPixelCount: number;
  clusterCount: number;
  totalPixelCount: number;
};

export type ColorClusterResult = {
  imageData: ImageData;
  stats: ProcessingStats;
};
