import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.sunrgbd_loader import *
from src.geometry.projection import *
from src.viz.draw import *

OUTPUT_DIR = ROOT / "results" / "draw_test"

def draw_sample(id):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s = load_sample(id)

    uvs = [project(box_corners(b), s.K, s.Rtilt)[0] for b in s.boxes3d]
    labels = [b.classname for b in s.boxes3d]

    draw_wireframes(s.rgb, uvs, OUTPUT_DIR / f"draw_test{id}.png", labels=labels)

for i in range(20):
    draw_sample(i)
