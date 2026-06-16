import numpy as np


def fit_palette_lab(
    lab: np.ndarray,
    k: int,
    max_samples: int,
    max_iter: int,
    random_state: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    pixels = _masked_pixels(lab, mask)
    if pixels.size == 0:
        raise ValueError("Image or mask has no usable pixels.")

    rng = np.random.default_rng(random_state)
    sample_count = min(len(pixels), max_samples)
    sample_idx = rng.choice(len(pixels), size=sample_count, replace=False)
    sample = pixels[sample_idx]
    k = min(max(1, k), len(sample))

    centers = _initial_centers(sample, k, rng)
    for _ in range(max_iter):
        labels = nearest_palette_indices(sample, centers)
        next_centers = centers.copy()
        for idx in range(k):
            members = sample[labels == idx]
            if len(members):
                next_centers[idx] = members.mean(axis=0)
        if np.allclose(next_centers, centers, atol=0.01):
            centers = next_centers
            break
        centers = next_centers
    return centers.astype(np.float32)


def nearest_palette_indices(lab_pixels: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    flat = lab_pixels.reshape(-1, 3).astype(np.float32)
    distances = ((flat[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1)


def stabilize_palette_ema(
    current: np.ndarray,
    reference: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    if reference is None:
        return current

    matched = match_palette_order(current, reference)
    alpha = max(0.0, min(1.0, alpha))
    return (1.0 - alpha) * matched + alpha * reference


def match_palette_order(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pairs = _greedy_assignment(current, reference)
    matched = np.empty_like(reference)
    for current_idx, reference_idx in pairs:
        matched[reference_idx] = current[current_idx]
    return matched


def _masked_pixels(lab: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    flat = lab.reshape(-1, 3).astype(np.float32)
    if mask is None:
        return flat

    mask_flat = mask.reshape(-1).astype(bool)
    return flat[mask_flat]


def _greedy_assignment(current: np.ndarray, reference: np.ndarray) -> list[tuple[int, int]]:
    distances = ((current[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
    candidates: list[tuple[float, int, int]] = []
    for current_idx in range(distances.shape[0]):
        for reference_idx in range(distances.shape[1]):
            candidates.append((float(distances[current_idx, reference_idx]), current_idx, reference_idx))

    pairs: list[tuple[int, int]] = []
    used_current: set[int] = set()
    used_reference: set[int] = set()
    for _, current_idx, reference_idx in sorted(candidates):
        if current_idx in used_current or reference_idx in used_reference:
            continue
        pairs.append((current_idx, reference_idx))
        used_current.add(current_idx)
        used_reference.add(reference_idx)
        if len(pairs) == min(len(current), len(reference)):
            break
    return pairs


def _initial_centers(sample: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, 3), dtype=np.float32)
    centers[0] = sample[rng.integers(0, len(sample))]
    closest = ((sample - centers[0]) ** 2).sum(axis=1)
    for idx in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            centers[idx:] = centers[idx - 1]
            break
        choice = rng.choice(len(sample), p=closest / total)
        centers[idx] = sample[choice]
        closest = np.minimum(closest, ((sample - centers[idx]) ** 2).sum(axis=1))
    return centers
