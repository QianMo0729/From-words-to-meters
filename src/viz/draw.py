import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from geometry.projection import EDGES


def draw_wireframes(rgb, uv_list, out_path, lw=1.2, labels=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)

    H, W = rgb.shape[:2]

    cmap = plt.get_cmap("tab10")
    for k, uv in enumerate(uv_list):
        color = cmap(k % 10)
        for i, j in EDGES:
            ax.plot(uv[[i, j], 0], uv[[i, j], 1], lw=lw, color=color)

        if labels is not None:
            u0, v0 = np.nanmin(uv, axis=0)
            u0 = float(np.clip(u0, 2, W - 2))
            v0 = float(np.clip(v0, 12, H - 2))

            ax.text(u0, v0, labels[k], color="white", fontsize=7, clip_on=True, bbox=dict(fc=color, alpha=0.7, pad=1, ec="none"))

    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.axis("off")

    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
