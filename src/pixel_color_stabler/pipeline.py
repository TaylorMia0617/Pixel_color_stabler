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
    working_lab = lab
    if cfg.processing_mode != "palette" and cfg.dirty_clean:
        working_lab = _clean_dirty_blocks(lab, process_mask, cfg)

    if cfg.processing_mode == "dirtyOnly":
        out_rgb = lab_to_rgb_uint8(working_lab)
        out_rgba = np.dstack([out_rgb, alpha]).astype(np.uint8)
        return Image.fromarray(out_rgba, mode="RGBA"), np.zeros((0, 3), dtype=np.float32)

    palette_source = smooth_chroma(working_lab) if cfg.prefilter == "chroma" else working_lab
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

    flat = working_lab.reshape(-1, 3)
    labels = nearest_palette_indices(flat, palette).reshape(lab.shape[:2])
    if cfg.dirty_clean:
        label_families = _classify_palette_families(palette)
        speck_area = _scaled_area(cfg.speck_area, labels.shape, minimum=2)
        label_area = _scaled_area(cfg.label_area, labels.shape, minimum=8)
        family_area = _scaled_area(cfg.family_area, labels.shape, minimum=8)
        clean_labels = _clean_dirty_label_islands(labels, palette, process_mask, speck_area)
        clean_labels = _clean_dirty_family_islands(clean_labels, palette, label_families, process_mask, family_area)
        clean_labels = _clean_dirty_label_islands(clean_labels, palette, process_mask, label_area)
        clean_labels = _smooth_label_map(clean_labels, process_mask, cfg.label_filter_size)
    else:
        clean_labels = labels
    target = palette[clean_labels.reshape(-1)].reshape(lab.shape)
    dirty_pixels = clean_labels != labels
    edges = edge_strength(working_lab[..., 0])

    out_lab = working_lab.copy()
    if cfg.mode == "flat":
        out_lab = target
    else:
        protection = np.where(dirty_pixels, 1.0, 1.0 - cfg.edge_protect * edges)
        l_alpha = cfg.strength * cfg.luma_strength * protection
        ab_alpha = cfg.strength * cfg.chroma_strength * protection
        l_alpha = np.where(dirty_pixels, np.maximum(l_alpha, cfg.strength * 0.65), l_alpha)
        ab_alpha = np.where(dirty_pixels, np.maximum(ab_alpha, cfg.strength), ab_alpha)
        out_lab[..., 0] = _lerp(working_lab[..., 0], target[..., 0], l_alpha)
        out_lab[..., 1] = _lerp(working_lab[..., 1], target[..., 1], ab_alpha)
        out_lab[..., 2] = _lerp(working_lab[..., 2], target[..., 2], ab_alpha)
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


def _clean_dirty_blocks(lab: np.ndarray, process_mask: np.ndarray, cfg: StabilizeConfig) -> np.ndarray:
    if not np.any(process_mask):
        return lab

    palette = fit_palette_lab(
        lab,
        k=cfg.analysis_palette_size,
        max_samples=cfg.max_samples,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        mask=process_mask,
    )
    labels = nearest_palette_indices(lab.reshape(-1, 3), palette).reshape(lab.shape[:2])
    palette_families = _classify_dirty_families(palette)
    family_map = palette_families[labels]
    edges = edge_strength(lab[..., 0])

    clean = lab.copy()
    max_area = _scaled_area(cfg.max_dirty_area, labels.shape, minimum=8)
    min_speck_area = _scaled_area(cfg.min_speck_area, labels.shape, minimum=2)
    repair_alpha = cfg.repair_strength
    l_alpha = min(1.0, repair_alpha * 0.35)
    ab_alpha = min(1.0, repair_alpha * 0.9)

    visited_family = np.zeros(labels.shape, dtype=bool)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if visited_family[y, x] or not process_mask[y, x]:
                continue
            own_family = int(family_map[y, x])
            component = _collect_value_component(
                family_map,
                process_mask,
                visited_family,
                y,
                x,
                own_family,
                cfg.connectivity,
            )
            area = len(component)
            if area > max_area and area > min_speck_area:
                continue
            if _is_protected_component(component, edges, cfg):
                continue

            ring = _collect_ring(component, process_mask, labels.shape, cfg.surround_radius)
            winner = _majority_ring_value(family_map, ring, own_family)
            if winner is None:
                continue
            target_family, dominance = winner
            if dominance < cfg.surround_dominance or (target_family == own_family and area > min_speck_area):
                continue

            target = _median_lab_from_ring(clean, family_map, ring, target_family)
            if target is None:
                continue
            _repair_component(clean, component, target, l_alpha, ab_alpha)

    visited_label = np.zeros(labels.shape, dtype=bool)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if visited_label[y, x] or not process_mask[y, x]:
                continue
            own_label = int(labels[y, x])
            own_family = int(palette_families[own_label])
            component = _collect_value_component(
                labels,
                process_mask,
                visited_label,
                y,
                x,
                own_label,
                cfg.connectivity,
            )
            area = len(component)
            if area > max_area and area > min_speck_area:
                continue
            if _is_protected_component(component, edges, cfg):
                continue

            ring = _collect_ring(component, process_mask, labels.shape, cfg.surround_radius)
            label_winner = _majority_ring_value(labels, ring, own_label)
            if label_winner is None:
                continue
            target_label, dominance = label_winner
            target_family = int(palette_families[target_label])
            centroid_distance = float(np.sqrt(((palette[own_label] - palette[target_label]) ** 2).sum()))
            is_dirty = (
                area <= min_speck_area
                or target_family != own_family
                or centroid_distance >= cfg.same_family_delta_e
            )
            if not is_dirty or dominance < cfg.surround_dominance:
                continue

            target = _median_lab_from_ring(clean, labels, ring, target_label)
            if target is None:
                continue
            _repair_component(clean, component, target, l_alpha, ab_alpha)

    return np.where(process_mask[..., None], clean, lab)


def _classify_dirty_families(palette: np.ndarray) -> np.ndarray:
    families = np.zeros(len(palette), dtype=np.int16)
    for idx, color in enumerate(palette):
        families[idx] = _classify_dirty_family(color)
    return families


def _classify_dirty_family(color: np.ndarray) -> int:
    lightness = float(color[0])
    a = float(color[1])
    b = float(color[2])
    chroma = float((a * a + b * b) ** 0.5)
    hue = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    if lightness > 78 and chroma < 22:
        return 3
    if chroma < 10:
        return 4
    if 70 <= hue <= 175 and a < 0:
        return 2
    if hue <= 70 or hue >= 330 or (a > 12 and b > -10):
        return 1
    return 0


def _collect_value_component(
    values: np.ndarray,
    process_mask: np.ndarray,
    visited: np.ndarray,
    start_y: int,
    start_x: int,
    value: int,
    connectivity: int,
) -> list[tuple[int, int]]:
    height, width = values.shape
    offsets = _neighbor_offsets(connectivity)
    component: list[tuple[int, int]] = []
    queue = [(start_y, start_x)]
    visited[start_y, start_x] = True

    cursor = 0
    while cursor < len(queue):
        y, x = queue[cursor]
        cursor += 1
        component.append((y, x))
        for dy, dx in offsets:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if visited[ny, nx] or not process_mask[ny, nx] or int(values[ny, nx]) != value:
                continue
            visited[ny, nx] = True
            queue.append((ny, nx))

    return component


def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 8:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    return ((-1, 0), (0, -1), (0, 1), (1, 0))


def _collect_ring(
    component: list[tuple[int, int]],
    process_mask: np.ndarray,
    shape: tuple[int, int],
    radius: int,
) -> list[tuple[int, int]]:
    height, width = shape
    radius = max(1, int(radius))
    component_set = set(component)
    ring_set: set[tuple[int, int]] = set()

    for y, x in component:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if (ny, nx) in component_set or not process_mask[ny, nx]:
                    continue
                ring_set.add((ny, nx))

    return list(ring_set)


def _majority_ring_value(
    values: np.ndarray,
    ring: list[tuple[int, int]],
    own_value: int,
) -> tuple[int, float] | None:
    counts: dict[int, int] = {}
    total = 0
    for y, x in ring:
        value = int(values[y, x])
        if value == own_value:
            continue
        counts[value] = counts.get(value, 0) + 1
        total += 1
    if total == 0 or not counts:
        return None
    best_value = max(counts, key=counts.get)
    return best_value, counts[best_value] / total


def _median_lab_from_ring(
    lab: np.ndarray,
    values: np.ndarray,
    ring: list[tuple[int, int]],
    target_value: int,
) -> np.ndarray | None:
    samples = [lab[y, x] for y, x in ring if int(values[y, x]) == target_value]
    if not samples:
        return None
    return np.median(np.array(samples, dtype=np.float32), axis=0)


def _repair_component(
    lab: np.ndarray,
    component: list[tuple[int, int]],
    target: np.ndarray,
    l_alpha: float,
    ab_alpha: float,
) -> None:
    for y, x in component:
        lab[y, x, 0] = lab[y, x, 0] * (1.0 - l_alpha) + target[0] * l_alpha
        lab[y, x, 1] = lab[y, x, 1] * (1.0 - ab_alpha) + target[1] * ab_alpha
        lab[y, x, 2] = lab[y, x, 2] * (1.0 - ab_alpha) + target[2] * ab_alpha


def _is_protected_component(
    component: list[tuple[int, int]],
    edges: np.ndarray,
    cfg: StabilizeConfig,
) -> bool:
    if not component:
        return True
    values = np.array([edges[y, x] for y, x in component], dtype=np.float32)
    if float(values.mean()) > cfg.dirty_edge_protect:
        return True
    if float(values.max()) > cfg.detail_protect and len(component) > 2:
        return True
    return False


def _clean_dirty_label_islands(
    labels: np.ndarray,
    palette: np.ndarray,
    process_mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    height, width = labels.shape
    clean = labels.copy()
    visited = np.zeros(labels.shape, dtype=bool)
    min_area = max(2, int(min_area))

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not process_mask[y, x]:
                continue

            label = int(clean[y, x])
            component = _collect_component(clean, process_mask, visited, y, x, label)
            if len(component) >= min_area:
                continue

            replacement = _majority_neighbor_label(clean, palette, process_mask, component, label)
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
    palette: np.ndarray,
    label_families: np.ndarray,
    process_mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    height, width = labels.shape
    clean = labels.copy()
    visited = np.zeros(labels.shape, dtype=bool)
    min_area = max(2, int(min_area))

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
                palette,
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
    palette: np.ndarray,
    label_families: np.ndarray,
    process_mask: np.ndarray,
    component: list[tuple[int, int]],
    own_family: int,
) -> int | None:
    height, width = labels.shape
    component_mask = np.zeros(labels.shape, dtype=bool)
    counts: dict[int, int] = {}
    source_color = _average_component_color(labels, palette, component)

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

    return _best_neighbor_label(counts, source_color, palette)


def _smooth_label_map(labels: np.ndarray, process_mask: np.ndarray, filter_size: int) -> np.ndarray:
    height, width = labels.shape
    smooth = labels.copy()
    radius = 2 if filter_size == 5 else 1
    min_votes = 13 if filter_size == 5 else 5

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            if not process_mask[y, x]:
                continue

            counts: dict[int, int] = {}
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny = y + dy
                    nx = x + dx
                    if not process_mask[ny, nx]:
                        continue
                    label = int(labels[ny, nx])
                    counts[label] = counts.get(label, 0) + 1

            best_label = max(counts, key=counts.get)
            if best_label != int(labels[y, x]) and counts[best_label] >= min_votes:
                smooth[y, x] = best_label

    return smooth


def _scaled_area(base_area: int, shape: tuple[int, int], minimum: int) -> int:
    height, width = shape
    scale = (height * width) / _BASE_AREA_PIXELS
    return max(minimum, round(max(1, base_area) * scale))


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
    palette: np.ndarray,
    process_mask: np.ndarray,
    component: list[tuple[int, int]],
    own_label: int,
) -> int | None:
    height, width = labels.shape
    component_mask = np.zeros(labels.shape, dtype=bool)
    counts: dict[int, int] = {}
    source_color = palette[own_label]

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

    return _best_neighbor_label(counts, source_color, palette)


def _average_component_color(
    labels: np.ndarray,
    palette: np.ndarray,
    component: list[tuple[int, int]],
) -> np.ndarray:
    if not component:
        return np.zeros(3, dtype=np.float32)
    values = np.array([palette[int(labels[y, x])] for y, x in component], dtype=np.float32)
    return values.mean(axis=0)


def _best_neighbor_label(
    counts: dict[int, int],
    source_color: np.ndarray,
    palette: np.ndarray,
) -> int | None:
    if not counts:
        return None

    best_label: int | None = None
    best_score = -1.0
    for label, count in counts.items():
        distance = float(np.sqrt(((palette[label] - source_color) ** 2).sum()))
        score = count / (1.0 + distance / 20.0)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label
