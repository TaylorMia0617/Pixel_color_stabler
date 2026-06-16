import numpy as np


def edge_strength(l_channel: np.ndarray) -> np.ndarray:
    luma = l_channel.astype(np.float32)
    padded = np.pad(luma, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
        + padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    magnitude = np.sqrt(gx * gx + gy * gy)
    high = np.percentile(magnitude, 95)
    if high <= 0:
        return np.zeros_like(magnitude, dtype=np.float32)
    return np.clip(magnitude / high, 0.0, 1.0).astype(np.float32)
