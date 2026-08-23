import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache


# Get the current directory of this script
def get_current_directory() -> Path:
    return Path(__file__).resolve().parent

mat_path = get_current_directory() / "raw/SUNRGBD/SUNRGBDMeta3DBB_v2.mat"


@lru_cache(maxsize=2)
def get_metadata(metadata_path=mat_path):
    return scipy.io.loadmat(
        metadata_path,
        squeeze_me=True,
        struct_as_record=False,
    )["SUNRGBDMeta"]

# Load the .mat file and print its keys and structure for debugging purposes
def load_mat(mat_path = mat_path):
    print(f"\nLoading .mat file from: {mat_path}")

    meta = scipy.io.loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
    )
    
    print(meta.keys())

    data = meta["SUNRGBDMeta"]
    
    print(type(data))
    print(data.shape)
    print(data[0]._fieldnames) # 'sequenceName', 'Rtilt', 'K', 'depthpath', 'rgbpath', 'anno_extrinsics', 'depthname', 'rgbname', 'sensorType', 'valid', 'groundtruth3DBB']

    print(f".mat file loaded from: {mat_path}\n")

    return None

@dataclass
class Paths:
    depthpath: Path
    rgbpath: Path

# Get the path of the RGB image corresponding to a given id
def get_id_path(id, mat_path = mat_path) -> Path:
    data = get_metadata(mat_path)

    remote_prefix = "/n/fs/sun3d/data/"

    rgb_path = get_current_directory() / "raw" / Path(data[id].rgbpath).relative_to(remote_prefix)

    depth_path = get_current_directory() / "raw" / Path(data[id].depthpath).relative_to(remote_prefix)

    return Paths(rgbpath = rgb_path, depthpath = depth_path)


@dataclass
class Box3D:
    centroid: np.ndarray
    basis: np.ndarray
    coeffs: np.ndarray
    classname: str


@dataclass
class SUNRGBDMeta:
    rgb: np.ndarray | None
    K: np.ndarray
    depth: np.ndarray
    Rtilt: np.ndarray
    boxes3d: list[Box3D]


def load_sample(id, load_rgb=True):
    path = get_id_path(id = id)
    data = get_metadata()

    rgb = np.array(Image.open(path.rgbpath)) if load_rgb else None

    depth_raw = Image.open(path.depthpath)
    encode = np.array(depth_raw).astype(np.uint32)
    depth = ((encode >> 3) | ((encode & 0b111) << 13)).astype(np.float32) / 1000.0

    raw_boxes = np.atleast_1d(data[id].groundtruth3DBB)

    boxes3d = [
        Box3D(
            centroid = np.array(box.centroid),
            basis=np.array(box.basis),
            coeffs=np.array(box.coeffs),
            classname=str(box.classname),
        )
        for box in raw_boxes
    ]

    return SUNRGBDMeta(
        rgb = rgb,
        K = data[id].K,
        depth = np.array(depth),
        Rtilt = data[id].Rtilt,
        boxes3d = boxes3d
    )
