import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sunrgbd_loader import load_sample
from src.geometry.projection import box_corners, project, world_to_camera


META_2D_PATH = ROOT / "data" / "raw" / "SUNRGBD" / "SUNRGBDMeta2DBB_v2.mat"
EPSILON_FLOOR_M = 0.06
EPSILON_QUADRATIC = 0.01
FALLBACK_MATCH_IOU = 0.10
MAX_OCCLUDED_RATIO = 0.30
MIN_HIT_PIXELS = 1000
MIN_DEPTH_COVERAGE = 0.50
MIN_IN_FRAME_RATIO = 0.50


@lru_cache(maxsize=1)
def get_2d_metadata():
    return scipy.io.loadmat(
        META_2D_PATH,
        squeeze_me=True,
        struct_as_record=False,
    )["SUNRGBDMeta2DBB"]


def get_2d_boxes(id):
    raw_boxes = get_2d_metadata()[id].groundtruth2DBB

    if isinstance(raw_boxes, np.ndarray) and raw_boxes.size == 0:
        return []

    return list(np.atleast_1d(raw_boxes))


def projected_box(box, sample):
    uv, z = project(box_corners(box), sample.K, sample.Rtilt)
    uv = uv - 1.0
    valid = np.isfinite(uv).all(axis=1) & (z > 0)

    if not np.any(valid):
        return None

    points = uv[valid]
    return np.array(
        [
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max(),
        ],
        dtype=np.float64,
    )


def annotation_box(box):
    x, y, width, height = np.asarray(box.gtBb2D, dtype=np.float64)
    return np.array([x - 1.0, y - 1.0, x - 1.0 + width, y - 1.0 + height])


def box_iou(first, second):
    if first is None or second is None:
        return 0.0

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection

    return intersection / union if union > 0 else 0.0


def match_2d_annotations(id, sample):
    boxes2d = get_2d_boxes(id)
    projected = [projected_box(box, sample) for box in sample.boxes3d]
    annotation_boxes = [annotation_box(box) for box in boxes2d]
    matches = [
        {"match_method": "none", "match_iou": None, "box2d_idx": None}
        for _ in sample.boxes3d
    ]
    matched_2d = set()

    def assign(available_2d, method, minimum_iou):
        class_names = sorted(
            {
                str(boxes2d[index].classname)
                for index in available_2d
            }
        )

        for class_name in class_names:
            object_indices = [
                index
                for index, box in enumerate(sample.boxes3d)
                if matches[index]["box2d_idx"] is None
                and str(box.classname) == class_name
            ]
            annotation_indices = [
                index
                for index in available_2d
                if index not in matched_2d
                and str(boxes2d[index].classname) == class_name
            ]

            if not object_indices or not annotation_indices:
                continue

            scores = np.array(
                [
                    [
                        box_iou(projected[object_index], annotation_boxes[annotation_index])
                        for annotation_index in annotation_indices
                    ]
                    for object_index in object_indices
                ]
            )
            rows, columns = linear_sum_assignment(-scores)

            for row, column in zip(rows, columns):
                score = float(scores[row, column])

                if score < minimum_iou:
                    continue

                object_index = object_indices[row]
                annotation_index = annotation_indices[column]
                matches[object_index] = {
                    "match_method": method,
                    "match_iou": score,
                    "box2d_idx": annotation_index,
                }
                matched_2d.add(annotation_index)

    def assign_by_iou(available_2d, method, minimum_iou):
        object_indices = [
            index
            for index in range(len(sample.boxes3d))
            if matches[index]["box2d_idx"] is None
        ]
        annotation_indices = [
            index
            for index in available_2d
            if index not in matched_2d
        ]

        if not object_indices or not annotation_indices:
            return

        scores = np.array(
            [
                [
                    box_iou(projected[object_index], annotation_boxes[annotation_index])
                    for annotation_index in annotation_indices
                ]
                for object_index in object_indices
            ]
        )
        rows, columns = linear_sum_assignment(-scores)

        for row, column in zip(rows, columns):
            score = float(scores[row, column])

            if score < minimum_iou:
                continue

            object_index = object_indices[row]
            annotation_index = annotation_indices[column]
            matches[object_index] = {
                "match_method": method,
                "match_iou": score,
                "box2d_idx": annotation_index,
            }
            matched_2d.add(annotation_index)

    official = [
        index
        for index, box in enumerate(boxes2d)
        if bool(np.asarray(box.has3dbox).item())
    ]
    fallback = [
        index
        for index, box in enumerate(boxes2d)
        if not bool(np.asarray(box.has3dbox).item())
    ]
    assign(official, "official_has3dbox", 0.0)
    assign_by_iou(official, "official_has3dbox+iou", FALLBACK_MATCH_IOU)
    assign(fallback, "class+iou", FALLBACK_MATCH_IOU)

    return matches


def convex_hull(points):
    points = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))

    if len(points) <= 1:
        return np.asarray(points, dtype=np.float64)

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_area(points):
    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0


def hull_pixel_count(points):
    if len(points) < 3:
        return 0

    y_start = int(np.ceil(points[:, 1].min()))
    y_stop = int(np.floor(points[:, 1].max()))

    if y_start > y_stop:
        return 0

    row_count = y_stop - y_start + 1

    if row_count > 100000:
        return max(1, int(round(polygon_area(points))))

    total = 0

    for chunk_start in range(y_start, y_stop + 1, 10000):
        chunk_stop = min(y_stop + 1, chunk_start + 10000)
        y = np.arange(chunk_start, chunk_stop, dtype=np.float64)
        left = np.full(len(y), np.inf)
        right = np.full(len(y), -np.inf)

        for start, end in zip(points, np.roll(points, -1, axis=0)):
            delta_y = end[1] - start[1]

            if abs(delta_y) < 1e-12:
                continue

            low = min(start[1], end[1])
            high = max(start[1], end[1])
            active = (y >= low) & (y <= high)

            if not np.any(active):
                continue

            x = start[0] + (y[active] - start[1]) * (end[0] - start[0]) / delta_y
            left[active] = np.minimum(left[active], x)
            right[active] = np.maximum(right[active], x)

        valid = np.isfinite(left) & np.isfinite(right)
        widths = np.maximum(
            0,
            np.floor(right[valid]).astype(np.int64)
            - np.ceil(left[valid]).astype(np.int64)
            + 1,
        )
        total += int(widths.sum())

    return total


def in_frame_ratio(box, sample, hit_pixels):
    uv, z = project(box_corners(box), sample.K, sample.Rtilt)

    if np.any(z <= 0) or not np.isfinite(uv).all():
        return 0.0

    full_hull_pixels = hull_pixel_count(convex_hull(uv - 1.0))

    if full_hull_pixels == 0:
        return 0.0

    return min(1.0, hit_pixels / full_hull_pixels)


def visibility_decision(record):
    if (
        record["t_near_median"] is None
        or record["t_far_median"] is None
        or record["t_near_median"] <= 0
        or record["t_far_median"] <= record["t_near_median"]
    ):
        return False, "invalid_geometry"
    if record["hit_pixels"] < MIN_HIT_PIXELS:
        return False, "too_few_pixels"
    if record["depth_coverage"] < MIN_DEPTH_COVERAGE:
        return False, "low_coverage"
    if record["in_frame_ratio"] < MIN_IN_FRAME_RATIO:
        return False, "truncated"
    if record["occluded_ratio"] is None or record["occluded_ratio"] > MAX_OCCLUDED_RATIO:
        return False, "occluded"
    return True, None


def set_visibility_decisions(record):
    record["visibility_pass"], record["reject_reason"] = visibility_decision(record)
    record["target_visibility_pass"] = bool(
        record["visibility_pass"] and record["has_2d_annotation"]
    )
    record["target_reject_reason"] = (
        record["reject_reason"]
        if not record["visibility_pass"]
        else None if record["has_2d_annotation"] else "no_2d_annotation"
    )


def build_visibility_records(
    id,
    sample,
    theory_depth,
    raw_depth,
    matches,
    epsilon,
):
    records = []

    for object_idx, (box, theory_points, raw_points, match) in enumerate(
        zip(sample.boxes3d, theory_depth, raw_depth, matches)
    ):
        hit_pixels = len(theory_points)
        finite_near = theory_points[:, 2][np.isfinite(theory_points[:, 2])]
        finite_far = theory_points[:, 3][np.isfinite(theory_points[:, 3])]
        t_near_median = float(np.median(finite_near)) if len(finite_near) else None
        t_far_median = float(np.median(finite_far)) if len(finite_far) else None
        scales_with_depth = epsilon is None
        object_epsilon = (
            max(EPSILON_FLOOR_M, EPSILON_QUADRATIC * t_near_median ** 2)
            if scales_with_depth and t_near_median is not None
            else EPSILON_FLOOR_M if scales_with_depth else float(epsilon)
        )
        record = {
            "image_id": int(id),
            "object_idx": int(object_idx),
            "class_name": str(box.classname),
            "t_near_median": (
                round(t_near_median, 6) if t_near_median is not None else None
            ),
            "t_far_median": (
                round(t_far_median, 6) if t_far_median is not None else None
            ),
            "in_frame_ratio": round(in_frame_ratio(box, sample, hit_pixels), 6),
            "has_2d_annotation": match["match_method"] != "none",
            "match_method": match["match_method"],
            "match_iou": (
                round(float(match["match_iou"]), 6)
                if match["match_iou"] is not None
                else None
            ),
            "epsilon_m": round(object_epsilon, 6),
            "epsilon_scales_with_depth": scales_with_depth,
            "occluded_ratio": None,
            "occluded_ratio_eps_0_15": None,
            "occluded_ratio_eps_0_25": None,
            "box_support_ratio": None,
            "behind_ratio": None,
            "valid_pixels": 0,
            "hit_pixels": int(hit_pixels),
            "depth_coverage": 0.0,
            "visibility_pass": False,
            "reject_reason": None,
            "target_visibility_pass": False,
            "target_reject_reason": None,
        }

        if hit_pixels == 0:
            set_visibility_decisions(record)
            records.append(record)
            continue

        z_near = theory_points[:, 2]
        z_far = theory_points[:, 3]
        observed = raw_points[:, 2]
        valid = (
            np.isfinite(z_near)
            & np.isfinite(z_far)
            & np.isfinite(observed)
            & (observed > 0)
        )

        valid_pixels = int(valid.sum())
        record["valid_pixels"] = valid_pixels
        record["depth_coverage"] = round(float(valid_pixels / hit_pixels), 6)

        if valid_pixels == 0:
            set_visibility_decisions(record)
            records.append(record)
            continue

        occluded = valid & (observed < z_near - object_epsilon)
        supported = (
            valid
            & (observed >= z_near - object_epsilon)
            & (observed <= z_far + object_epsilon)
        )
        behind = valid & (observed > z_far + object_epsilon)

        record["occluded_ratio"] = round(float(occluded.sum() / valid_pixels), 6)
        record["box_support_ratio"] = round(float(supported.sum() / valid_pixels), 6)
        record["behind_ratio"] = round(float(behind.sum() / valid_pixels), 6)
        set_visibility_decisions(record)
        records.append(record)

    return records


def compute_visibility(id: int, epsilon: float | None = None) -> list[dict]:
    sample = load_sample(id, load_rgb=False)
    theory_depth, raw_depth = get_depth(id, sample=sample)
    matches = match_2d_annotations(id, sample)
    records = build_visibility_records(
        id,
        sample,
        theory_depth,
        raw_depth,
        matches,
        epsilon,
    )
    fixed = {
        fixed_epsilon: build_visibility_records(
            id,
            sample,
            theory_depth,
            raw_depth,
            matches,
            fixed_epsilon,
        )
        for fixed_epsilon in (0.15, 0.25)
    }

    for object_idx, record in enumerate(records):
        record["occluded_ratio_eps_0_15"] = fixed[0.15][object_idx]["occluded_ratio"]
        record["occluded_ratio_eps_0_25"] = fixed[0.25][object_idx]["occluded_ratio"]

    return records


def compute_visibility_sweep(id: int, epsilons) -> dict:
    sample = load_sample(id, load_rgb=False)
    theory_depth, raw_depth = get_depth(id, sample=sample)
    matches = match_2d_annotations(id, sample)
    records = {
        "scaled": build_visibility_records(
            id,
            sample,
            theory_depth,
            raw_depth,
            matches,
            None,
        )
    }

    requested_epsilons = [float(epsilon) for epsilon in epsilons]
    fixed_epsilons = sorted(set(requested_epsilons) | {0.15, 0.25})

    fixed_records = {}

    for epsilon in fixed_epsilons:
        fixed_records[float(epsilon)] = build_visibility_records(
            id,
            sample,
            theory_depth,
            raw_depth,
            matches,
            float(epsilon),
        )

    for epsilon in requested_epsilons:
        records[epsilon] = fixed_records[epsilon]

    for object_idx, record in enumerate(records["scaled"]):
        record["occluded_ratio_eps_0_15"] = fixed_records[0.15][object_idx]["occluded_ratio"]
        record["occluded_ratio_eps_0_25"] = fixed_records[0.25][object_idx]["occluded_ratio"]

    return records


def get_depth(id: int, sample=None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if sample is None:
        sample = load_sample(id, load_rgb=False)

    h, w = sample.depth.shape

    theory_depth = []
    raw_depth = []

    for box in sample.boxes3d:
        corner_uv, corner_z = project(
            box_corners(box),
            sample.K,
            sample.Rtilt,
        )

        if np.all(corner_z <= 0):
            theory_depth.append(np.empty((0, 4), dtype=np.float32))
            raw_depth.append(np.empty((0, 3), dtype=np.float32))
            continue

        if np.any(corner_z <= 0):
            u_min, u_max = 0, w - 1
            v_min, v_max = 0, h - 1
        else:
            uv_index = corner_uv - 1.0

            if not np.isfinite(uv_index).all():
                theory_depth.append(np.empty((0, 4), dtype=np.float32))
                raw_depth.append(np.empty((0, 3), dtype=np.float32))
                continue

            u_min = max(0, int(np.floor(uv_index[:, 0].min())))
            u_max = min(w - 1, int(np.ceil(uv_index[:, 0].max())))
            v_min = max(0, int(np.floor(uv_index[:, 1].min())))
            v_max = min(h - 1, int(np.ceil(uv_index[:, 1].max())))

            if u_min > u_max or v_min > v_max:
                theory_depth.append(np.empty((0, 4), dtype=np.float32))
                raw_depth.append(np.empty((0, 3), dtype=np.float32))
                continue

        u_grid, v_grid = np.meshgrid(
            np.arange(u_min, u_max + 1),
            np.arange(v_min, v_max + 1),
        )
        u = u_grid.ravel()
        v = v_grid.ravel()

        pixels_for_k = np.stack(
            [
                u + 1.0,
                v + 1.0,
                np.ones(len(u)),
            ],
            axis=1,
        )
        rays = np.linalg.solve(sample.K, pixels_for_k.T).T

        center_camera = world_to_camera(
            box.centroid[None, :],
            sample.Rtilt,
        )[0]
        basis_camera = world_to_camera(
            box.basis,
            sample.Rtilt,
        )
        half_size = np.asarray(box.coeffs, dtype=np.float64)

        if (
            not np.isfinite(basis_camera).all()
            or not np.isfinite(half_size).all()
            or np.any(half_size <= 0)
        ):
            theory_depth.append(np.empty((0, 4), dtype=np.float32))
            raw_depth.append(np.empty((0, 3), dtype=np.float32))
            continue

        try:
            origin_local = np.linalg.solve(
                basis_camera.T,
                -center_camera,
            )
            rays_local = np.linalg.solve(
                basis_camera.T,
                rays.T,
            ).T
        except np.linalg.LinAlgError:
            theory_depth.append(np.empty((0, 4), dtype=np.float32))
            raw_depth.append(np.empty((0, 3), dtype=np.float32))
            continue

        parallel = np.abs(rays_local) < 1e-10
        outside_parallel = np.any(
            parallel
            & (
                np.abs(origin_local)[None, :]
                > half_size[None, :] + 1e-9
            ),
            axis=1,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half_size - origin_local) / rays_local
            t2 = (half_size - origin_local) / rays_local

        t_low = np.minimum(t1, t2)
        t_high = np.maximum(t1, t2)
        t_low[parallel] = -np.inf
        t_high[parallel] = np.inf

        t_enter = np.max(t_low, axis=1)
        z_far = np.min(t_high, axis=1)
        z_near = np.maximum(t_enter, 0.0)
        hit = (
            (~outside_parallel)
            & (z_far >= z_near)
            & (z_far > 0)
        )

        hit_u = u[hit]
        hit_v = v[hit]

        theory_depth.append(
            np.column_stack(
                [
                    hit_u,
                    hit_v,
                    z_near[hit],
                    z_far[hit],
                ]
            ).astype(np.float32, copy=False)
        )
        raw_depth.append(
            np.column_stack(
                [
                    hit_u,
                    hit_v,
                    sample.depth[hit_v, hit_u],
                ]
            ).astype(np.float32, copy=False)
        )

    return theory_depth, raw_depth
