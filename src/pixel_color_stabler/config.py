from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StabilizeConfig:
    k: int = 8
    strength: float = 0.8
    luma_strength: float = 0.25
    chroma_strength: float = 0.88
    edge_protect: float = 0.7
    max_samples: int = 50_000
    random_state: int = 0
    prefilter: Literal["none", "chroma"] = "chroma"
    stabilize: Literal["none", "ema"] = "none"
    reference: Literal["none", "first"] = "none"
    palette_ema: float = 0.35
    max_iter: int = 40
    mode: Literal["shaded", "flat"] = "shaded"
    dirty_clean: bool = True
    speck_area: int = 24
    label_area: int = 72
    family_area: int = 260
    label_filter_size: int = 3
    processing_mode: Literal["palette", "dirtyOnly", "dirtyThenPalette"] = "dirtyThenPalette"
    analysis_palette_size: int = 12
    max_dirty_area: int = 900
    min_speck_area: int = 6
    surround_radius: int = 3
    surround_dominance: float = 0.58
    same_family_delta_e: float = 10
    dirty_edge_protect: float = 0.75
    detail_protect: float = 0.7
    repair_strength: float = 0.85
    connectivity: int = 4

    def normalized(self) -> "StabilizeConfig":
        stabilize = self.stabilize if self.stabilize in {"none", "ema"} else "none"
        label_filter_size = 5 if int(self.label_filter_size) >= 5 else 3
        processing_mode = (
            self.processing_mode
            if self.processing_mode in {"palette", "dirtyOnly", "dirtyThenPalette"}
            else "dirtyThenPalette"
        )
        connectivity = 8 if int(self.connectivity) == 8 else 4
        return StabilizeConfig(
            k=max(1, int(self.k)),
            strength=_clamp01(self.strength),
            luma_strength=_clamp01(self.luma_strength),
            chroma_strength=_clamp01(self.chroma_strength),
            edge_protect=_clamp01(self.edge_protect),
            max_samples=max(1, int(self.max_samples)),
            random_state=int(self.random_state),
            prefilter=self.prefilter,
            stabilize=stabilize,
            reference=self.reference if self.reference in {"none", "first"} else "none",
            palette_ema=_clamp01(self.palette_ema),
            max_iter=max(1, int(self.max_iter)),
            mode=self.mode if self.mode in {"shaded", "flat"} else "shaded",
            dirty_clean=bool(self.dirty_clean),
            speck_area=max(1, int(self.speck_area)),
            label_area=max(1, int(self.label_area)),
            family_area=max(1, int(self.family_area)),
            label_filter_size=label_filter_size,
            processing_mode=processing_mode,
            analysis_palette_size=max(2, int(self.analysis_palette_size)),
            max_dirty_area=max(1, int(self.max_dirty_area)),
            min_speck_area=max(1, int(self.min_speck_area)),
            surround_radius=max(1, int(self.surround_radius)),
            surround_dominance=_clamp01(self.surround_dominance),
            same_family_delta_e=max(0.0, float(self.same_family_delta_e)),
            dirty_edge_protect=_clamp01(self.dirty_edge_protect),
            detail_protect=_clamp01(self.detail_protect),
            repair_strength=_clamp01(self.repair_strength),
            connectivity=connectivity,
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
