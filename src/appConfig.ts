import type { AppConfig, ProcessingSettings } from "./types";

export const DEFAULT_SETTINGS: ProcessingSettings = {
  strength: 80,
  paletteSize: 6,
  lumaStrength: 30,
  chromaStrength: 90,
  edgeProtect: 70,
};

export const DEFAULT_APP_CONFIG: AppConfig = {
  settings: DEFAULT_SETTINGS,
  defaultExportFolder: null,
  basketFolder: null,
  basketAutoScan: false,
};

export function normalizeAppConfig(config?: Partial<AppConfig> | null): AppConfig {
  return {
    settings: {
      ...DEFAULT_SETTINGS,
      ...(config?.settings ?? {}),
    },
    defaultExportFolder: config?.defaultExportFolder ?? null,
    basketFolder: config?.basketFolder ?? null,
    basketAutoScan: Boolean(config?.basketAutoScan),
  };
}
