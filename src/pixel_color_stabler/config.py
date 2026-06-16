from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StabilizeConfig:
    k: int = 6
    strength: float = 0.8
    luma_strength: float = 0.3
    chroma_strength: float = 0.9
    edge_protect: float = 0.7
    max_samples: int = 50_000
    random_state: int = 0
    prefilter: Literal["none", "chroma"] = "chroma"
    stabilize: Literal["none", "ema"] = "none"
    reference: Literal["none", "first"] = "none"
    palette_ema: float = 0.35
    max_iter: int = 40
    dirty_clean: bool = True
    island_area: int = 260

    def normalized(self) -> "StabilizeConfig":
        stabilize = self.stabilize if self.stabilize in {"none", "ema"} else "none"
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
            dirty_clean=bool(self.dirty_clean),
            island_area=max(1, int(self.island_area)),
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
