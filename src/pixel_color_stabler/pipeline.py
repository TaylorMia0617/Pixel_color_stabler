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
            mask=active_mask,
        )
    )

    flat = lab.reshape(-1, 3)
    labels = nearest_palette_indices(flat, palette)
    target = palette[labels].reshape(lab.shape)
    edges = edge_strength(lab[..., 0])
    protection = 1.0 - cfg.edge_protect * edges

    l_alpha = cfg.strength * cfg.luma_strength * protection
    ab_alpha = cfg.strength * cfg.chroma_strength * protection

    out_lab = lab.copy()
    out_lab[..., 0] = _lerp(lab[..., 0], target[..., 0], l_alpha)
    out_lab[..., 1] = _lerp(lab[..., 1], target[..., 1], ab_alpha)
    out_lab[..., 2] = _lerp(lab[..., 2], target[..., 2], ab_alpha)
    if active_mask is not None:
        out_lab = np.where(active_mask[..., None], out_lab, lab)

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
