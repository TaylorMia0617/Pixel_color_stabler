from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(mask: str | Path | Image.Image | None, size: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None

    pil = mask if isinstance(mask, Image.Image) else Image.open(mask)
    if pil.size != size:
        pil = pil.resize(size, Image.Resampling.NEAREST)
    values = np.array(pil.convert("L"), dtype=np.uint8)
    return values > 127
