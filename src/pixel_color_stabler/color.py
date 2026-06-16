import numpy as np


_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def rgb_uint8_to_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(rgb_uint8_to_float(rgb[..., :3]))
    xyz = linear @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    ).T
    scaled = xyz / _D65
    f = _lab_f(scaled)
    lab = np.empty_like(f, dtype=np.float32)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def lab_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    xyz = np.stack([_lab_f_inv(fx), _lab_f_inv(fy), _lab_f_inv(fz)], axis=-1) * _D65
    linear = xyz @ np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    ).T
    srgb = _linear_to_srgb(np.clip(linear, 0.0, 1.0))
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055)


def _lab_f(value: np.ndarray) -> np.ndarray:
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    return np.where(value > epsilon, np.cbrt(value), (kappa * value + 16.0) / 116.0)


def _lab_f_inv(value: np.ndarray) -> np.ndarray:
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    cubed = value**3
    return np.where(cubed > epsilon, cubed, (116.0 * value - 16.0) / kappa)
