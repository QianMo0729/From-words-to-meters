import io
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path


# Get the current directory of this script
def get_current_directory() -> Path:
    return Path(__file__).resolve().parent

mat_path = get_current_directory() / "raw/SUNRGBD/SUNRGBDMeta3DBB_v2.mat"

# Load the .mat file and print its keys and structure for debugging purposes
def load_mat(mat_path):
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


# Get the path of the RGB image corresponding to a given id
def get_id_path(id, para, mat_path = mat_path) -> Path:
    meta = scipy.io.loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
    )
    data = meta["SUNRGBDMeta"]

    remote_prefix = "/n/fs/sun3d/data/"

    if para == "rgb":
        rgb_path = get_current_directory() / "raw" / Path(data[id].rgbpath).relative_to(remote_prefix)
        return rgb_path

    if para == "depth":
        depth_path = get_current_directory() / "raw" / Path(data[id].depthpath).relative_to(remote_prefix)
        return depth_path


# Load the RGB image corresponding to a given id and optionally display it
def load_rgb(id, show: bool = False) -> np.ndarray:
    print(f"\nLoading RGB image for id: {id}") 

    img = Image.open(get_id_path(id = id, para = "rgb"))

    if img is not None and show:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show(block = False)

    print(f"RGB image loaded for id: {id}\n")

    return np.array(img)


# Get the camera intrinsic matrix K for a given id
def load_K(id: int, mat_path = mat_path) -> np.ndarray:
    print(f"\nLoading camera intrinsic matrix K for id: {id}")

    meta = scipy.io.loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
    )
    data = meta["SUNRGBDMeta"]

    K = data[id].K

    print(f"Camera intrinsic matrix K loaded for id: {id}\n")

    return K


# Load the depth image corresponding to a given id and optionally display it
def load_depth(id, show: bool = False) -> np.ndarray:
    print(f"\nLoading depth image for id: {id}")

    depth_raw = Image.open(get_id_path(id = id, para = "depth"))
    encode = np.array(depth_raw).astype(np.uint32)
    depth = ((encode >> 3) | ((encode & 0b111) << 13)).astype(np.float32) / 1000.0

    print("shape:", depth.shape)
    print("dtype:", depth.dtype)
    print("min:", np.min(depth))
    print("max:", np.max(depth))
    print("median:", np.median(depth))

    if depth is not None and show:
        plt.figure()
        plt.imshow(depth)
        plt.axis('off')
        plt.show(block = False)

    print(f"Depth image loaded for id: {id}\n")

    return depth
