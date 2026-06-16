import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from pixel_color_stabler import StabilizeConfig, stabilize_image
from pixel_color_stabler.color import lab_to_rgb_uint8, rgb_to_lab
from pixel_color_stabler.palette import fit_palette_lab, match_palette_order


def test_rgb_lab_round_trip_shape_and_dtype():
    rgb = np.array([[[10, 20, 30], [200, 180, 40]]], dtype=np.uint8)
    lab = rgb_to_lab(rgb)
    out = lab_to_rgb_uint8(lab)

    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


def test_palette_size_matches_request():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:2, :, :] = [20, 40, 200]
    rgb[2:, :, :] = [220, 80, 30]
    lab = rgb_to_lab(rgb)
    palette = fit_palette_lab(lab, k=2, max_samples=16, max_iter=10, random_state=0)

    assert palette.shape == (2, 3)


def test_stabilize_image_preserves_alpha_and_size():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = [40, 120, 200]
    rgba[..., 3] = 128
    rgba[0, 0, 3] = 0
    image = Image.fromarray(rgba, mode="RGBA")

    out = stabilize_image(image, StabilizeConfig(k=1, strength=1))
    out_arr = np.array(out)

    assert out.size == image.size
    assert out_arr[0, 0, 3] == 0
    assert out_arr[1, 1, 3] == 128


def test_masked_out_area_is_unchanged():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = [40, 120, 200]
    rgba[..., 3] = 255
    rgba[:2, :2, :3] = [220, 50, 20]
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 255

    image = Image.fromarray(rgba, mode="RGBA")
    mask_image = Image.fromarray(mask, mode="L")
    out = stabilize_image(image, StabilizeConfig(k=1, strength=1), mask=mask_image)
    out_arr = np.array(out)

    assert np.array_equal(out_arr[2:, 2:, :], rgba[2:, 2:, :])


def test_palette_matching_reorders_to_reference():
    current = np.array([[100, 10, 10], [20, 5, 5]], dtype=np.float32)
    reference = np.array([[20, 5, 5], [100, 10, 10]], dtype=np.float32)

    matched = match_palette_order(current, reference)

    assert np.array_equal(matched, reference)
