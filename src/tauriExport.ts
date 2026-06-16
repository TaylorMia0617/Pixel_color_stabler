import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";

export async function saveBlobAsPng(blob: Blob, suggestedName: string): Promise<void> {
  const path = await save({
    defaultPath: suggestedName,
    filters: [{ name: "PNG image", extensions: ["png"] }],
  });

  if (!path) {
    return;
  }

  const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
  await invoke("save_png_file", { path, bytes });
}

export function downloadBlobAsPng(blob: Blob, suggestedName: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = suggestedName;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function isTauriRuntime(): boolean {
  return "__TAURI_INTERNALS__" in window;
}
