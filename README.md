# Pixel Color Stabler

A lightweight tool to reduce noisy color blocks in AI-generated images by compressing colors toward a compact palette while preserving edges and luminance detail.

This repository contains two layers. The desktop app is the main product path; the Python package is an auxiliary research and batch-processing path.

- Main: a Tauri + Vite desktop UI for interactive image cleanup.
- Auxiliary: a Python CLI/API core that follows the research report in `deep-research-report (1).md`.

## Desktop App

The desktop app runs the main cleanup pipeline in the Vite/React frontend so it can be packaged with Tauri and used offline without a Python runtime. Its primary controls mirror the research report:

- Lab palette size
- Overall strength
- Luminance strength
- Chroma strength
- Edge protection
- Preview zoom and PNG export

## Python Auxiliary Features

- Lab color-space palette compression
- Edge-aware remapping
- Separate luminance and chroma strengths
- Deterministic NumPy palette fitting
- Masked region cleanup
- Batch palette stabilization with EMA
- Alpha preservation
- CLI and Python API

## Installation

Core Python dependencies:

```bash
pip install -U numpy Pillow
```

Better quality / future optional engines:

```bash
pip install -U opencv-python-headless scikit-image scikit-learn scipy
```

Development:

```bash
pip install -e ".[dev]"
```

Desktop UI:

```bash
npm install
npm run tauri dev
```

## Quick Start

```bash
python -m pixel_color_stabler input.png -o output.png --k 6
```

Replace `input.png` with a real file path. On Windows, quote paths that contain spaces or non-ASCII characters:

```powershell
python -m pixel_color_stabler "C:\Users\user\Downloads\image.png" -o "C:\Users\user\Downloads\output.png" --k 6
```

Strong cleanup:

```bash
python -m pixel_color_stabler input.png -o output.png --k 4 --strength 0.9
```

Nearly single-color harmonization:

```bash
python -m pixel_color_stabler input.png -o output.png --k 1 --luma-strength 0.2 --chroma-strength 0.95
```

Batch mode:

```bash
python -m pixel_color_stabler input_dir/ -o out_dir/ --k 4 --stabilize ema --reference first
```

Masked cleanup:

```bash
python -m pixel_color_stabler input.png -o output.png --k 1 --mask mask.png --luma-strength 0.2 --chroma-strength 0.95
```

## Python API

```python
from pixel_color_stabler import StabilizeConfig, stabilize_image

cfg = StabilizeConfig(k=6, strength=0.8, luma_strength=0.3, chroma_strength=0.9)
img = stabilize_image("input.png", config=cfg)
img.save("output.png")
```

## Recommended Defaults

- `k=6`
- `strength=0.8`
- `luma_strength=0.3`
- `chroma_strength=0.9`
- `edge_protect=0.7`
- `max_samples=50000`

## Limitations

- Aggressive settings can posterize gradients.
- Different Stable Diffusion styles need different palette sizes.
- Optional scikit/OpenCV engines from the report are planned but not required for the baseline core.
- SLIC-guided remapping from the report is planned as a quality-tier enhancement.
