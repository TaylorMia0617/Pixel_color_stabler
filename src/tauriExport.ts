import { save } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";

export async function saveBlobAsPng(blob: Blob, suggestedName: string): Promise<void> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const path = await save({
    defaultPath: suggestedName,
    filters: [{ name: "PNG image", extensions: ["png"] }],
  });

  if (!path) {
    return;
  }

  await writeFile(path, bytes);
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
