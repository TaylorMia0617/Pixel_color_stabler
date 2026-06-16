import numpy as np

from .color import rgb_to_lab


def mean_delta_e76(source_rgb: np.ndarray, output_rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    source = rgb_to_lab(source_rgb[..., :3]).reshape(-1, 3)
    output = rgb_to_lab(output_rgb[..., :3]).reshape(-1, 3)
    delta = np.sqrt(((source - output) ** 2).sum(axis=1))
    if mask is not None:
        delta = delta[mask.reshape(-1).astype(bool)]
    return float(delta.mean()) if len(delta) else 0.0


def low_gradient_chroma_variance(rgb: np.ndarray, edge_mask: np.ndarray) -> float:
    lab = rgb_to_lab(rgb[..., :3])
    flat = edge_mask < 0.2
    if not np.any(flat):
        return 0.0
    chroma = lab[..., 1:][flat]
    return float(chroma.var())
