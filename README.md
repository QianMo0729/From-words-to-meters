# From Words to Meters
## Overview
## Installation
## Dataset
The project uses RGB-D indoor-scene datasets to study the relationship between visual observations, language descriptions, and metric 3D spatial information.
For more details, please read [Data Documentation](DATA.md)
## Project Structure
```text
from-words-to-meters/
├── configs/                  # Project configuration files
├── data/
│   ├── raw/                  # Local SUN RGB-D data (not tracked)
│   ├── processed/            # Generated datasets (not tracked)
│   └── sunrgbd_loader.py     # SUN RGB-D metadata and sample loading
├── learning/                 # Development notes
├── results/
│   └── draw_test/            # Generated wireframe visualizations
├── scripts/                  # Project scripts
├── src/
│   ├── geometry/
│   │   └── projection.py     # 3D box construction and projection
│   └── viz/
│       └── draw.py           # RGB wireframe rendering
├── tests/
│   ├── draw_sample.py        # 20-sample visual validation
│   └── test_cuda.py          # CUDA availability check
├── .gitignore
├── DATA.md                   # Dataset documentation
├── PLAN.md                   # Project plan
├── README.md
└── requirements.txt
```
## Training
## Evaluation

Run the 20-sample SUN RGB-D wireframe visual check:

```powershell
.\.venv\Scripts\python.exe tests\draw_sample.py
```

The script renders sample IDs 0-19 to `results/draw_test/`.

Inspect every image for systematic errors in the projected 3D wireframes, including incorrect position, orientation, scale, or object association. Projected edges are preserved even when they extend beyond the image boundary. Only label anchors are constrained to the image bounds so that out-of-view labels do not enlarge the saved canvas.

## Results

Qualitative visualization outputs are generated locally under `results/draw_test/` and are not intended to be tracked by Git. Record the D4 validation outcome here only after all 20 images have been inspected.

## Roadmap
