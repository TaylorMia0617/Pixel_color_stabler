from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .color import lab_to_rgb_uint8, rgb_to_lab
from .config import StabilizeConfig
from .edges import edge_strength
from .mask import load_mask
from .palette import fit_palette_lab, nearest_palette_indices, stabilize_palette_ema
from .prefilter import smooth_chroma

_BASE_AREA_PIXELS = 512 * 512


def stabilize_image(
    image: str | Path | Image.Image,
    config: StabilizeConfig | None = None,
    mask: str | Path | Image.Image | None = None,
    palette_lab: np.ndarray | None = None,
) -> Image.Image:
    out, _ = stabilize_image_with_palette(image, config=config, mask=mask, palette_lab=palette_lab)
    return out


def stabilize_image_with_palette(
    image: str | Path | Image.Image,
    config: StabilizeConfig | None = None,
    mask: str | Path | Image.Image | None = None,
    palette_lab: np.ndarray | None = None,
) -> tuple[Image.Image, np.ndarray]:
    cfg = (config or StabilizeConfig()).normalized()
    pil = _load_image(image)
    rgba = np.array(pil.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    active_mask = load_mask(mask, pil.size)

    lab = rgb_to_lab(rgb)
    background_mask = _edge_connected_background(alpha, lab)
    process_mask = alpha > 0
    if active_mask is not None:
        process_mask &= active_mask
    process_mask &= ~background_mask
    palette_source = smooth_chroma(lab) if cfg.prefilter == "chroma" else lab
    palette = (
        palette_lab
        if palette_lab is not None
        else fit_palette_lab(
            palette_source,
            k=cfg.k,
            max_samples=cfg.max_samples,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            mask=process_mask,
        )
    )

    flat = lab.reshape(-1, 3)
    labels = nearest_palette_indices(flat, palette).reshape(lab.shape[:2])
    if cfg.dirty_clean:
        label_families = _classify_palette_families(palette)
        clean_labels = _clean_dirty_family_islands(labels, label_families, process_mask, cfg.island_area)
        clean_labels = _clean_dirty_label_islands(clean_labels, process_mask, cfg.island_area)
        clean_labels = _smooth_label_map(clean_labels, process_mask)
    else:
        clean_labels = labels
    target = palette[clean_labels.reshape(-1)].reshape(lab.shape)
    dirty_pixels = clean_labels != labels
    edges = edge_strength(lab[..., 0])
    protection = np.where(dirty_pixels, 1.0, 1.0 - cfg.edge_protect * edges)

    l_alpha = cfg.strength * cfg.luma_strength * protection
    ab_alpha = cfg.strength * cfg.chroma_strength * protection
    l_alpha = np.where(dirty_pixels, np.maximum(l_alpha, cfg.strength * 0.65), l_alpha)
    ab_alpha = np.where(dirty_pixels, np.maximum(ab_alpha, cfg.strength), ab_alpha)

    out_lab = lab.copy()
    out_lab[..., 0] = _lerp(lab[..., 0], target[..., 0], l_alpha)
    out_lab[..., 1] = _lerp(lab[..., 1], target[..., 1], ab_alpha)
    out_lab[..., 2] = _lerp(lab[..., 2], target[..., 2], ab_alpha)
    out_lab = np.where(process_mask[..., None], out_lab, lab)

    out_rgb = lab_to_rgb_uint8(out_lab)
    out_rgba = np.dstack([out_rgb, alpha]).astype(np.uint8)
    return Image.fromarray(out_rgba, mode="RGBA"), palette


def stabilize_batch(
    input_path: str | Path,
    output_path: str | Path,
    config: StabilizeConfig | None = None,
    mask: str | Path | Image.Image | None = None,
) -> list[Path]:
    cfg = (config or StabilizeConfig()).normalized()
    source = Path(input_path)
    target = Path(output_path)
    target.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    stable_palette: np.ndarray | None = None

    paths = _iter_images(source)
    for path in paths:
        out, current_palette = stabilize_image_with_palette(path, config=cfg, mask=mask)
        if cfg.stabilize == "ema":
            stable_palette = stabilize_palette_ema(current_palette, stable_palette, cfg.palette_ema)
            out, _ = stabilize_image_with_palette(path, config=cfg, mask=mask, palette_lab=stable_palette)
        elif cfg.reference == "first" and stable_palette is None:
            stable_palette = current_palette

        out_path = target / f"{path.stem}-stabilized.png"
        out.save(out_path)
        outputs.append(out_path)
    return outputs


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(image)


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.iterdir()):
        if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            yield child


def _lerp(source: np.ndarray, target: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return source * (1.0 - alpha) + target * alpha


def _edge_connected_background(alpha: np.ndarray, lab: np.ndarray) -> np.ndarray:
    height, width = alpha.shape
    background = np.zeros((height, width), dtype=bool)
    queue: list[tuple[int, int]] = []

    def enqueue(y: int, x: int) -> None:
        if background[y, x] or not _is_background_like(alpha[y, x], lab[y, x]):
            return
        background[y, x] = True
        queue.append((y, x))

    for x in range(width):
        enqueue(0, x)
        enqueue(height - 1, x)
    for y in range(1, height - 1):
        enqueue(y, 0)
        enqueue(y, width - 1)

    cursor = 0
    while cursor < len(queue):
        y, x = queue[cursor]
        cursor += 1
        if x > 0:
            enqueue(y, x - 1)
        if x + 1 < width:
            enqueue(y, x + 1)
        if y > 0:
            enqueue(y - 1, x)
        if y + 1 < height:
            enqueue(y + 1, x)

    return background


def _is_background_like(alpha: int, lab: np.ndarray) -> bool:
    return alpha == 0 or (lab[0] > 92 and abs(lab[1]) < 5.5 and abs(lab[2]) < 7)


def _clean_dirty_label_islands(labels: np.ndarray, process_mask: np.ndarray, base_area: int) -> np.ndarray:
    height, width = labels.shape
    clean = labels.copy()
    visited = np.zeros(labels.shape, dtype=bool)
    min_area = max(2, round(base_area * ((height * width) / _BASE_AREA_PIXELS)))

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not process_mask[y, x]:
                continue

            label = int(clean[y, x])
            component = _collect_component(clean, process_mask, visited, y, x, label)
            if len(component) >= min_area:
                continue

            replacement = _majority_neighbor_label(clean, process_mask, component, label)
            if replacement is None:
                continue

            for cy, cx in component:
                clean[cy, cx] = replacement

    return clean


def _classify_palette_families(palette: np.ndarray) -> np.ndarray:
    families = np.zeros(len(palette), dtype=np.uint8)
    for idx, color in enumerate(palette):
        chroma = float((color[1] ** 2 + color[2] ** 2) ** 0.5)
        if color[0] > 78 and chroma < 18:
            families[idx] = 3
        elif color[1] < -7 and color[2] > 0:
            families[idx] = 2
        elif color[1] > 10 and color[2] > -5:
            families[idx] = 1
        elif chroma < 14:
            families[idx] = 4
        else:
            families[idx] = 0
    return families


def _clean_dirty_family_islands(
    labels: np.ndarray,
    label_families: np.ndarray,
    process_mask: np.ndarray,
    base_area: int,
) -> np.ndarray:
    height, width = labels.shape
    clean = labels.copy()
    visited = np.zeros(labels.shape, dtype=bool)
    min_area = max(2, round(base_area * ((height * width) / _BASE_AREA_PIXELS)))

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not process_mask[y, x]:
                continue

            family = int(label_families[int(clean[y, x])])
            component = _collect_family_component(clean, label_families, process_mask, visited, y, x, family)
            if len(component) >= min_area:
                continue

            replacement = _majority_neighbor_label_by_family(
                clean,
                label_families,
                process_mask,
                component,
                family,
            )
            if replacement is None:
                continue

            for cy, cx in component:
                clean[cy, cx] = replacement

    return clean


def _collect_family_component(
    labels: np.ndarray,
    label_families: np.ndarray,
    process_mask: np.ndarray,
    visited: np.ndarray,
    start_y: int,
    start_x: int,
    family: int,
) -> list[tuple[int, int]]:
    height, width = labels.shape
    component: list[tuple[int, int]] = []
    queue = [(start_y, start_x)]
    visited[start_y, start_x] = True

    cursor = 0
    while cursor < len(queue):
        y, x = queue[cursor]
        cursor += 1
        component.append((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if (
                    visited[ny, nx]
                    or not process_mask[ny, nx]
                    or int(label_families[int(labels[ny, nx])]) != family
                ):
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))

    return component


def _majority_neighbor_label_by_family(
    labels: np.ndarray,
    label_families: np.ndarray,
    process_mask: np.ndarray,
    component: list[tuple[int, int]],
    own_family: int,
) -> int | None:
    height, width = labels.shape
    component_mask = np.zeros(labels.shape, dtype=bool)
    counts: dict[int, int] = {}

    for y, x in component:
        component_mask[y, x] = True

    for y, x in component:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                neighbor_label = int(labels[ny, nx])
                if (
                    not process_mask[ny, nx]
                    or component_mask[ny, nx]
                    or int(label_families[neighbor_label]) == own_family
                ):
                    continue
                counts[neighbor_label] = counts.get(neighbor_label, 0) + 1

    if not counts:
        return None
    return max(counts, key=counts.get)


def _smooth_label_map(labels: np.ndarray, process_mask: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    smooth = labels.copy()

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if not process_mask[y, x]:
                continue

            counts: dict[int, int] = {}
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny = y + dy
                    nx = x + dx
                    if not process_mask[ny, nx]:
                        continue
                    label = int(labels[ny, nx])
                    counts[label] = counts.get(label, 0) + 1

            best_label = max(counts, key=counts.get)
            if best_label != int(labels[y, x]) and counts[best_label] >= 5:
                smooth[y, x] = best_label

    return smooth


def _collect_component(
    labels: np.ndarray,
    process_mask: np.ndarray,
    visited: np.ndarray,
    start_y: int,
    start_x: int,
    label: int,
) -> list[tuple[int, int]]:
    height, width = labels.shape
    component: list[tuple[int, int]] = []
    queue = [(start_y, start_x)]
    visited[start_y, start_x] = True

    cursor = 0
    while cursor < len(queue):
        y, x = queue[cursor]
        cursor += 1
        component.append((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not process_mask[ny, nx] or int(labels[ny, nx]) != label:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))

    return component


def _majority_neighbor_label(
    labels: np.ndarray,
    process_mask: np.ndarray,
    component: list[tuple[int, int]],
    own_label: int,
) -> int | None:
    height, width = labels.shape
    component_mask = np.zeros(labels.shape, dtype=bool)
    counts: dict[int, int] = {}

    for y, x in component:
        component_mask[y, x] = True

    for y, x in component:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                neighbor_label = int(labels[ny, nx])
                if (
                    not process_mask[ny, nx]
                    or component_mask[ny, nx]
                    or neighbor_label == own_label
                ):
                    continue
                counts[neighbor_label] = counts.get(neighbor_label, 0) + 1

    if not counts:
        return None
    return max(counts, key=counts.get)
