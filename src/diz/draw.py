import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from geometry.projection import EDGES, box_corners, project
from data.sunrgbd_loader import load_groundtruth3DBB, load_sample


def draw_wireframes(rgb, uv_list, out_path, lw = 1.2):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)

    for uv in uv_list:
        for i, j in EDGES:
            ax.plot(uv[[i, j], 0], uv[[i, j], 1], lw=lw)

    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.axis("off")

    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
