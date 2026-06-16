from pixel_color_stabler import StabilizeConfig


def test_config_normalizes_ranges():
    cfg = StabilizeConfig(
        k=0,
        strength=2,
        luma_strength=-1,
        chroma_strength=1.5,
        stabilize="unknown",  # type: ignore[arg-type]
        palette_ema=2,
    ).normalized()

    assert cfg.k == 1
    assert cfg.strength == 1
    assert cfg.luma_strength == 0
    assert cfg.chroma_strength == 1
    assert cfg.stabilize == "none"
    assert cfg.palette_ema == 1
