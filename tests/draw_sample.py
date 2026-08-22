import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.sunrgbd_loader import *
from src.geometry.projection import *
from src.viz.draw import *

def draw_sample(id):
    s = load_sample(id)
    boxes = load_groundtruth3DBB(id)

    uvs = [project(box_corners(b), s.K, s.Rtilt)[0] for b in boxes]

    draw_wireframes(s.rgb, uvs, ROOT / f"results/draw_test/draw_test{id}.png", labels =[b.classname for b in boxes])
    return None

for i in range(0, 20):
    draw_sample(i)