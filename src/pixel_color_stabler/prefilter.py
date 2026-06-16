import numpy as np


def smooth_chroma(lab: np.ndarray) -> np.ndarray:
    out = lab.copy()
    for channel in (1, 2):
        out[..., channel] = _box_filter_3x3(lab[..., channel])
    return out


def _box_filter_3x3(channel: np.ndarray) -> np.ndarray:
    padded = np.pad(channel.astype(np.float32), 1, mode="edge")
    total = np.zeros_like(channel, dtype=np.float32)
    for y in range(3):
        for x in range(3):
            total += padded[y : y + channel.shape[0], x : x + channel.shape[1]]
    return total / 9.0
