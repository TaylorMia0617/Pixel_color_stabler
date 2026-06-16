import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import type { AppConfig, ImageFileEntry } from "./types";

export async function loadConfig(): Promise<AppConfig> {
  return invoke<AppConfig>("load_app_config");
}

export async function saveConfig(config: AppConfig): Promise<void> {
  await invoke("save_app_config", { config });
}

export async function pickImageFiles(): Promise<ImageFileEntry[]> {
  const selected = await open({
    multiple: true,
    directory: false,
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }],
  });
  const paths = Array.isArray(selected) ? selected : selected ? [selected] : [];
  return paths.map(pathToEntry);
}

export async function pickFolder(): Promise<string | null> {
  const selected = await open({ multiple: false, directory: true });
  return typeof selected === "string" ? selected : null;
}

export async function scanImageFolder(path: string): Promise<ImageFileEntry[]> {
  return invoke<ImageFileEntry[]>("scan_image_folder", { path });
}

export async function readImageFile(path: string): Promise<number[]> {
  return invoke<number[]>("read_image_file", { path });
}

export async function savePngToDirectory(args: {
  outputDir: string | null;
  sourcePath: string;
  sourceName: string;
  suffix: string;
  bytes: number[];
}): Promise<string> {
  return invoke<string>("save_png_to_directory", {
    outputDir: args.outputDir,
    sourcePath: args.sourcePath,
    sourceName: args.sourceName,
    suffix: args.suffix,
    bytes: args.bytes,
  });
}

function pathToEntry(path: string): ImageFileEntry {
  const normalized = path.replace(/\\/g, "/");
  const name = normalized.split("/").pop() || "image";
  const sourceDir = normalized.includes("/") ? normalized.slice(0, normalized.lastIndexOf("/")) : "";
  return { path, name, sourceDir, size: 0 };
}
