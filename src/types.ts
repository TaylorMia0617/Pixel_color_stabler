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
  tolerance: number;
  strength: number;
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
