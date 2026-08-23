import numpy as np


_WORLD_TO_CAM_AXES = np.array([
    [1., 0.,  0.],
    [0., 0., -1.],
    [0., 1.,  0.],
])

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def world_to_camera(points_world: np.ndarray, Rtilt: np.ndarray):
    points_depth = (Rtilt.T @ points_world.T).T
    return (_WORLD_TO_CAM_AXES @ points_depth.T).T


def camera_to_world(points_camera: np.ndarray, Rtilt: np.ndarray):
    points_depth = (_WORLD_TO_CAM_AXES.T @ points_camera.T).T
    return (Rtilt @ points_depth.T).T


def camera_to_pixel(points_camera: np.ndarray, K: np.ndarray):
    pixel_homo = (K @ points_camera.T).T
    z_depth = points_camera[:, 2]

    uv = np.full((len(points_camera), 2), np.nan)

    valid = z_depth > 0

    uv[valid] = (pixel_homo[valid, :2]/ pixel_homo[valid, 2:3])

    return uv, z_depth


def project(points_world: np.ndarray, K: np.ndarray, Rtilt: np.ndarray):
    return camera_to_pixel(world_to_camera(points_world, Rtilt), K)


def box_corners(box):
    signs = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1, -1],
        [ 1, -1, -1],
    ])

    offsets = (signs * box.coeffs) @ box.basis

    return box.centroid + offsets
