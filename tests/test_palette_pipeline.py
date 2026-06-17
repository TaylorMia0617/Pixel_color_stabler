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


def test_dirty_color_island_is_repainted_from_surrounding_color():
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    rgba[..., :3] = [220, 40, 35]
    rgba[..., 3] = 255
    rgba[2, 2, :3] = [30, 180, 30]

    image = Image.fromarray(rgba, mode="RGBA")
    out = stabilize_image(
        image,
        StabilizeConfig(
            k=2,
            strength=1,
            luma_strength=0.3,
            chroma_strength=0.9,
            edge_protect=1,
        ),
    )
    out_arr = np.array(out)

    assert out_arr[2, 2, 0] > 150
    assert out_arr[2, 2, 1] < 100


def test_red_island_inside_green_subject_is_repainted():
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    rgba[..., :3] = [35, 155, 45]
    rgba[..., 3] = 255
    rgba[2, 2, :3] = [220, 45, 35]

    image = Image.fromarray(rgba, mode="RGBA")
    out = stabilize_image(
        image,
        StabilizeConfig(
            k=2,
            strength=1,
            chroma_strength=1,
            edge_protect=1,
        ),
    )
    out_arr = np.array(out)

    assert out_arr[2, 2, 0] < 100
    assert out_arr[2, 2, 1] > 100


def test_dirty_only_repairs_green_island_without_global_palette_repaint():
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    for y in range(5):
        for x in range(5):
            rgba[y, x, :3] = [210 + (x % 3) * 4, 45, 38]
            rgba[y, x, 3] = 255
    rgba[2, 2, :3] = [35, 180, 35]

    image = Image.fromarray(rgba, mode="RGBA")
    out = stabilize_image(
        image,
        StabilizeConfig(
            processing_mode="dirtyOnly",
            strength=0,
            k=12,
            max_dirty_area=80,
            repair_strength=1,
        ),
    )
    out_arr = np.array(out)

    assert out_arr[2, 2, 0] > 150
    assert out_arr[2, 2, 1] < 100
    assert abs(int(out_arr[0, 0, 0]) - int(rgba[0, 0, 0])) <= 6


def test_dirty_only_repairs_red_island_inside_green_subject():
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    rgba[..., :3] = [38, 155, 45]
    rgba[..., 3] = 255
    rgba[2, 2, :3] = [225, 45, 35]

    image = Image.fromarray(rgba, mode="RGBA")
    out = stabilize_image(
        image,
        StabilizeConfig(
            processing_mode="dirtyOnly",
            strength=0,
            k=12,
            max_dirty_area=80,
            repair_strength=1,
        ),
    )
    out_arr = np.array(out)

    assert out_arr[2, 2, 0] < 100
    assert out_arr[2, 2, 1] > 110


def test_high_palette_dirty_patch_is_still_repainted():
    rgba = np.zeros((7, 7, 4), dtype=np.uint8)
    for y in range(7):
        for x in range(7):
            rgba[y, x, :3] = [215 + (x % 3) * 5, 35 + (y % 4) * 4, 35]
            rgba[y, x, 3] = 255
    rgba[3, 3, :3] = [40, 175, 35]
    rgba[3, 4, :3] = [40, 175, 35]

    image = Image.fromarray(rgba, mode="RGBA")
    out = stabilize_image(
        image,
        StabilizeConfig(
            k=12,
            strength=1,
            luma_strength=0.7,
            chroma_strength=1,
            edge_protect=0.8,
        ),
    )
    out_arr = np.array(out)

    assert out_arr[3, 3, 0] > 150
    assert out_arr[3, 3, 1] < 110


def test_flat_mode_repaints_directly_to_palette_color():
    rgba = np.array(
        [
            [[220, 50, 40, 255], [222, 52, 42, 255]],
            [[40, 150, 45, 255], [42, 152, 47, 255]],
        ],
        dtype=np.uint8,
    )

    image = Image.fromarray(rgba, mode="RGBA")
    out = stabilize_image(
        image,
        StabilizeConfig(
            k=2,
            mode="flat",
            strength=0.1,
            luma_strength=0,
            chroma_strength=0.1,
        ),
    )
    out_arr = np.array(out)

    assert not np.array_equal(out_arr[..., :3], rgba[..., :3])


def test_palette_matching_reorders_to_reference():
    current = np.array([[100, 10, 10], [20, 5, 5]], dtype=np.float32)
    reference = np.array([[20, 5, 5], [100, 10, 10]], dtype=np.float32)

    matched = match_palette_order(current, reference)

    assert np.array_equal(matched, reference)
