from pixel_color_stabler import StabilizeConfig


def test_config_normalizes_ranges():
    cfg = StabilizeConfig(
        k=0,
        strength=2,
        luma_strength=-1,
        chroma_strength=1.5,
        stabilize="unknown",  # type: ignore[arg-type]
        palette_ema=2,
        mode="unknown",  # type: ignore[arg-type]
        speck_area=0,
        label_area=0,
        family_area=0,
        label_filter_size=4,
        processing_mode="unknown",  # type: ignore[arg-type]
        analysis_palette_size=1,
        max_dirty_area=0,
        min_speck_area=0,
        surround_radius=0,
        surround_dominance=2,
        same_family_delta_e=-1,
        dirty_edge_protect=2,
        detail_protect=-1,
        repair_strength=2,
        connectivity=5,
    ).normalized()

    assert cfg.k == 1
    assert cfg.strength == 1
    assert cfg.luma_strength == 0
    assert cfg.chroma_strength == 1
    assert cfg.stabilize == "none"
    assert cfg.palette_ema == 1
    assert cfg.mode == "shaded"
    assert cfg.speck_area == 1
    assert cfg.label_area == 1
    assert cfg.family_area == 1
    assert cfg.label_filter_size == 3
    assert cfg.processing_mode == "dirtyThenPalette"
    assert cfg.analysis_palette_size == 2
    assert cfg.max_dirty_area == 1
    assert cfg.min_speck_area == 1
    assert cfg.surround_radius == 1
    assert cfg.surround_dominance == 1
    assert cfg.same_family_delta_e == 0
    assert cfg.dirty_edge_protect == 1
    assert cfg.detail_protect == 0
    assert cfg.repair_strength == 1
    assert cfg.connectivity == 4
