import argparse
from pathlib import Path

from .config import StabilizeConfig
from .pipeline import stabilize_batch, stabilize_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pixel-color-stabler",
        description="Reduce noisy AI-image color blocks with edge-aware Lab palette compression.",
    )
    parser.add_argument("input", help="Input image file or directory.")
    parser.add_argument("-o", "--output", required=True, help="Output image file or directory.")
    parser.add_argument("--k", type=int, default=6, help="Target palette size.")
    parser.add_argument("--strength", type=float, default=0.8, help="Overall remap strength, 0..1.")
    parser.add_argument("--luma-strength", type=float, default=0.3, help="Luminance remap strength, 0..1.")
    parser.add_argument("--chroma-strength", type=float, default=0.9, help="Chroma remap strength, 0..1.")
    parser.add_argument("--edge-protect", type=float, default=0.7, help="Reduce remap strength near edges, 0..1.")
    parser.add_argument("--max-samples", type=int, default=50_000, help="Maximum pixels sampled for palette fitting.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic palette seed.")
    parser.add_argument("--mask", help="Optional grayscale mask; white pixels are processed.")
    parser.add_argument(
        "--stabilize",
        choices=["none", "ema"],
        default="none",
        help="Batch palette stabilization mode.",
    )
    parser.add_argument(
        "--reference",
        choices=["none", "first"],
        default="none",
        help="Reserve first-image palette as the batch reference.",
    )
    parser.add_argument("--palette-ema", type=float, default=0.35, help="EMA blend for batch palette stabilization.")
    parser.add_argument(
        "--prefilter",
        choices=["none", "chroma"],
        default="chroma",
        help="Optional chroma smoothing before palette fitting.",
    )
    args = parser.parse_args(argv)

    cfg = StabilizeConfig(
        k=args.k,
        strength=args.strength,
        luma_strength=args.luma_strength,
        chroma_strength=args.chroma_strength,
        edge_protect=args.edge_protect,
        max_samples=args.max_samples,
        random_state=args.seed,
        prefilter=args.prefilter,
        stabilize=args.stabilize,
        reference=args.reference,
        palette_ema=args.palette_ema,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        parser.error(f"input path does not exist: {input_path}")

    if input_path.is_dir():
        outputs = stabilize_batch(input_path, output_path, config=cfg, mask=args.mask)
        print(f"Wrote {len(outputs)} image(s) to {output_path}")
        return 0

    out = stabilize_image(input_path, config=cfg, mask=args.mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
