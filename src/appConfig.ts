import type { AppConfig, DirtyBlockSettings, ProcessingSettings } from "./types";

export const DEFAULT_SETTINGS: ProcessingSettings = {
  processingMode: "dirtyThenPalette",
  strength: 80,
  paletteSize: 8,
  lumaStrength: 25,
  chromaStrength: 88,
  edgeProtect: 70,
  cleanupMode: "shaded",
  dirtyClean: true,
  speckArea: 24,
  labelArea: 72,
  familyArea: 260,
  labelFilterSize: 3,
  dirtyBlock: {
    enabled: true,
    analysisPaletteSize: 12,
    maxDirtyArea: 900,
    minSpeckArea: 6,
    surroundRadius: 3,
    surroundDominance: 0.58,
    sameFamilyDeltaE: 10,
    edgeProtect: 75,
    detailProtect: 70,
    repairStrength: 85,
    connectivity: 4,
  },
};

export const DEFAULT_APP_CONFIG: AppConfig = {
  settings: DEFAULT_SETTINGS,
  defaultExportFolder: null,
  basketFolder: null,
  basketAutoScan: false,
};

export function normalizeAppConfig(config?: Partial<AppConfig> | null): AppConfig {
  const rawSettings: Partial<ProcessingSettings> = config?.settings ?? {};
  const labelFilterSize = rawSettings.labelFilterSize === 5 ? 5 : 3;
  const cleanupMode = rawSettings.cleanupMode === "flat" ? "flat" : "shaded";
  const processingMode =
    rawSettings.processingMode === "palette" ||
    rawSettings.processingMode === "dirtyOnly" ||
    rawSettings.processingMode === "dirtyThenPalette"
      ? rawSettings.processingMode
      : DEFAULT_SETTINGS.processingMode;
  const rawDirtyBlock: Partial<DirtyBlockSettings> = rawSettings.dirtyBlock ?? {};

  return {
    settings: {
      ...DEFAULT_SETTINGS,
      ...rawSettings,
      processingMode,
      cleanupMode,
      dirtyClean: rawSettings.dirtyClean ?? DEFAULT_SETTINGS.dirtyClean,
      speckArea: rawSettings.speckArea ?? DEFAULT_SETTINGS.speckArea,
      labelArea: rawSettings.labelArea ?? DEFAULT_SETTINGS.labelArea,
      familyArea: rawSettings.familyArea ?? DEFAULT_SETTINGS.familyArea,
      labelFilterSize,
      dirtyBlock: {
        ...DEFAULT_SETTINGS.dirtyBlock,
        ...rawDirtyBlock,
        connectivity: rawDirtyBlock.connectivity === 8 ? 8 : 4,
      },
    },
    defaultExportFolder: config?.defaultExportFolder ?? null,
    basketFolder: config?.basketFolder ?? null,
    basketAutoScan: Boolean(config?.basketAutoScan),
  };
}
