use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessingSettingsConfig {
    strength: u8,
    palette_size: u8,
    luma_strength: u8,
    chroma_strength: u8,
    edge_protect: u8,
}

impl Default for ProcessingSettingsConfig {
    fn default() -> Self {
        Self {
            strength: 80,
            palette_size: 6,
            luma_strength: 30,
            chroma_strength: 90,
            edge_protect: 70,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppConfig {
    settings: ProcessingSettingsConfig,
    default_export_folder: Option<String>,
    basket_folder: Option<String>,
    basket_auto_scan: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            settings: ProcessingSettingsConfig::default(),
            default_export_folder: None,
            basket_folder: None,
            basket_auto_scan: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageFileEntry {
    path: String,
    name: String,
    source_dir: String,
    size: u64,
}

#[tauri::command]
fn save_png_file(path: String, bytes: Vec<u8>) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|error| error.to_string())
}

#[tauri::command]
fn load_app_config(app: AppHandle) -> Result<AppConfig, String> {
    let path = config_path(&app)?;
    if !path.exists() {
        return Ok(AppConfig::default());
    }

    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&text).map_err(|error| error.to_string())
}

#[tauri::command]
fn save_app_config(app: AppHandle, config: AppConfig) -> Result<(), String> {
    let path = config_path(&app)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    let text = serde_json::to_string_pretty(&config).map_err(|error| error.to_string())?;
    std::fs::write(path, text).map_err(|error| error.to_string())
}

#[tauri::command]
fn scan_image_folder(path: String) -> Result<Vec<ImageFileEntry>, String> {
    let folder = PathBuf::from(path);
    let mut entries = Vec::new();
    let read_dir = std::fs::read_dir(&folder).map_err(|error| error.to_string())?;

    for entry in read_dir {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if !path.is_file() || !is_supported_image(&path) {
            continue;
        }

        let metadata = entry.metadata().map_err(|error| error.to_string())?;
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("image")
            .to_string();
        let source_dir = path
            .parent()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_string();

        entries.push(ImageFileEntry {
            path: path.to_string_lossy().to_string(),
            name,
            source_dir,
            size: metadata.len(),
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

#[tauri::command]
fn read_image_file(path: String) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|error| error.to_string())
}

#[tauri::command]
fn save_png_to_directory(
    output_dir: Option<String>,
    source_path: String,
    source_name: String,
    suffix: String,
    bytes: Vec<u8>,
) -> Result<String, String> {
    let dir = output_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| fallback_output_dir(&source_path));
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;

    let stem = Path::new(&source_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("image");
    let candidate = next_available_path(&dir, stem, &suffix);
    std::fs::write(&candidate, bytes).map_err(|error| error.to_string())?;
    Ok(candidate.to_string_lossy().to_string())
}

fn config_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_config_dir()
        .map_err(|error| error.to_string())?;
    Ok(dir.join("config.json"))
}

fn fallback_output_dir(source_path: &str) -> PathBuf {
    Path::new(source_path)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("out")
}

fn next_available_path(dir: &Path, stem: &str, suffix: &str) -> PathBuf {
    let base = format!("{stem}-{suffix}");
    let mut candidate = dir.join(format!("{base}.png"));
    let mut index = 2;
    while candidate.exists() {
        candidate = dir.join(format!("{base}-{index}.png"));
        index += 1;
    }
    candidate
}

fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp")
    )
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            save_png_file,
            load_app_config,
            save_app_config,
            scan_image_folder,
            read_image_file,
            save_png_to_directory
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
