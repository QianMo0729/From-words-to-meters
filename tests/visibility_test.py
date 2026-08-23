import sys
from pprint import pprint
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sunrgbd_loader import load_sample
from src.geometry.projection import EDGES, box_corners, project
from src.geometry.visibility import *


OUTPUT_ROOT = ROOT / "results" / "visibility_test"


def depth_limits(theory_depth, raw_depth):
    values = []

    for theory_points in theory_depth:
        if len(theory_points):
            theory_values = theory_points[:, 2]
            theory_values = theory_values[np.isfinite(theory_values)]
            if len(theory_values):
                values.append(theory_values)

    for raw_points in raw_depth:
        if len(raw_points):
            raw_values = raw_points[:, 2]
            raw_values = raw_values[np.isfinite(raw_values) & (raw_values > 0)]
            if len(raw_values):
                values.append(raw_values)

    if not values:
        return None, None

    values = np.concatenate(values)
    vmin = float(np.percentile(values, 2))
    vmax = float(np.percentile(values, 98))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None

    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    return vmin, vmax


def draw_boxes(ax, uv_list, labels, image_shape):
    h, w = image_shape
    cmap = plt.get_cmap("tab10")

    for box_index, (uv, label) in enumerate(zip(uv_list, labels)):
        uv_index = uv - 1.0
        color = cmap(box_index % 10)

        for start, end in EDGES:
            edge = uv_index[[start, end]]
            if np.isfinite(edge).all():
                ax.plot(edge[:, 0], edge[:, 1], color=color, lw=1.0)

        valid = np.isfinite(uv_index).all(axis=1)
        if np.any(valid):
            u0, v0 = np.min(uv_index[valid], axis=0)
            u0 = float(np.clip(u0, 2, w - 2))
            v0 = float(np.clip(v0, 12, h - 2))
            ax.text(
                u0,
                v0,
                f"{box_index}: {label}",
                color="white",
                fontsize=7,
                clip_on=True,
                bbox=dict(fc=color, alpha=0.75, pad=1, ec="none"),
            )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")


def save_rgb(rgb, uv_list, labels, output_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)
    draw_boxes(ax, uv_list, labels, rgb.shape[:2])
    ax.set_title(title)
    fig.savefig(output_path, bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def save_depth_map(
    depth_map,
    uv_list,
    labels,
    output_path,
    title,
    vmin,
    vmax,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    image = ax.imshow(
        np.ma.masked_invalid(depth_map),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    draw_boxes(ax, uv_list, labels, depth_map.shape)
    ax.set_title(title)

    if np.isfinite(depth_map).any():
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Depth (m)")

    fig.savefig(output_path, bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def save_classification_map(
    classification,
    uv_list,
    labels,
    output_path,
    title,
):
    colors = ["#000000", "#E69F00", "#0072B2", "#CC79A7"]
    category_labels = ["Invalid depth", "Occluded", "Box support", "Behind box"]
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        np.ma.masked_where(classification < 0, classification),
        cmap=cmap,
        norm=norm,
    )
    draw_boxes(ax, uv_list, labels, classification.shape)
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for color, label in zip(colors, category_labels)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def make_maps(image_shape, theory_points, raw_points, epsilon):
    near_map = np.full(image_shape, np.nan, dtype=np.float32)
    raw_map = np.full(image_shape, np.nan, dtype=np.float32)
    classification = np.full(image_shape, -1, dtype=np.int8)

    if not len(theory_points):
        return near_map, raw_map, classification

    u = theory_points[:, 0].astype(np.intp)
    v = theory_points[:, 1].astype(np.intp)
    z_near = theory_points[:, 2]
    z_far = theory_points[:, 3]
    observed = raw_points[:, 2]
    valid = np.isfinite(observed) & (observed > 0)
    occluded = valid & (observed < z_near - epsilon)
    supported = (
        valid
        & (observed >= z_near - epsilon)
        & (observed <= z_far + epsilon)
    )
    behind = valid & (observed > z_far + epsilon)
    codes = np.zeros(len(theory_points), dtype=np.int8)
    codes[occluded] = 1
    codes[supported] = 2
    codes[behind] = 3

    near_map[v, u] = z_near
    raw_map[v[valid], u[valid]] = observed[valid]
    classification[v, u] = codes

    return near_map, raw_map, classification


def resolve_epsilon(theory_points, epsilon):
    if epsilon is not None:
        return float(epsilon)

    if not len(theory_points):
        return EPSILON_FLOOR_M

    z_near = theory_points[:, 2]
    z_near = z_near[np.isfinite(z_near)]

    if not len(z_near):
        return EPSILON_FLOOR_M

    t_near_median = float(np.median(z_near))
    return max(EPSILON_FLOOR_M, EPSILON_QUADRATIC * t_near_median ** 2)


def save_visibility_set(
    sample,
    id,
    uv_list,
    labels,
    near_map,
    raw_map,
    classification,
    sample_dir,
    prefix,
    scope,
    vmin,
    vmax,
):
    save_rgb(
        sample.rgb,
        uv_list,
        labels,
        sample_dir / f"{prefix}_rgb.png",
        f"Sample {id} | {scope} | RGB",
    )
    save_depth_map(
        near_map,
        uv_list,
        labels,
        sample_dir / f"{prefix}_theory_near.png",
        f"Sample {id} | {scope} | Theory near depth",
        vmin,
        vmax,
    )
    save_depth_map(
        raw_map,
        uv_list,
        labels,
        sample_dir / f"{prefix}_raw_depth.png",
        f"Sample {id} | {scope} | Observed depth",
        vmin,
        vmax,
    )
    save_classification_map(
        classification,
        uv_list,
        labels,
        sample_dir / f"{prefix}_classification.png",
        f"Sample {id} | {scope} | Visibility classification",
    )


def draw_visibility(id: int, merge: bool = False, epsilon: float | None = None):
    sample = load_sample(id)
    theory_depth, raw_depth = get_depth(id, sample=sample)

    if len(theory_depth) != len(sample.boxes3d):
        raise ValueError("Theory depth count does not match boxes3d")

    if len(raw_depth) != len(sample.boxes3d):
        raise ValueError("Raw depth count does not match boxes3d")

    for box_index, (theory_points, raw_points) in enumerate(
        zip(theory_depth, raw_depth)
    ):
        if not np.array_equal(theory_points[:, :2], raw_points[:, :2]):
            raise ValueError(f"UV mismatch for sample {id}, box {box_index}")

    sample_root = OUTPUT_ROOT / f"visibility_test{id}"
    sample_dir = sample_root / ("merge" if merge else "not_merge")
    sample_dir.mkdir(parents=True, exist_ok=True)

    for output_file in sample_root.glob("*.png"):
        output_file.unlink()

    for output_file in sample_dir.glob("*.png"):
        output_file.unlink()

    h, w = sample.depth.shape
    image_shape = (h, w)
    uv_list = [
        project(box_corners(box), sample.K, sample.Rtilt)[0]
        for box in sample.boxes3d
    ]
    labels = [box.classname for box in sample.boxes3d]

    if not merge:
        for box_index, (box, theory_points, raw_points, corner_uv) in enumerate(
            zip(sample.boxes3d, theory_depth, raw_depth, uv_list)
        ):
            near_map, raw_map, classification = make_maps(
                image_shape,
                theory_points,
                raw_points,
                resolve_epsilon(theory_points, epsilon),
            )
            vmin, vmax = depth_limits([theory_points], [raw_points])
            prefix = f"visibility_test{id}_box{box_index}"
            save_visibility_set(
                sample,
                id,
                [corner_uv],
                [box.classname],
                near_map,
                raw_map,
                classification,
                sample_dir,
                prefix,
                f"Box {box_index}: {box.classname}",
                vmin,
                vmax,
            )

        return

    near_map = np.full((h, w), np.nan, dtype=np.float32)
    raw_map = np.full((h, w), np.nan, dtype=np.float32)
    classification = np.full((h, w), -1, dtype=np.int8)
    owner_near = np.full((h, w), np.inf, dtype=np.float32)

    for theory_points, raw_points in zip(
        theory_depth,
        raw_depth,
    ):
        if not len(theory_points):
            continue

        u = theory_points[:, 0].astype(np.intp)
        v = theory_points[:, 1].astype(np.intp)
        z_near = theory_points[:, 2]
        z_far = theory_points[:, 3]
        observed = raw_points[:, 2]
        object_epsilon = resolve_epsilon(theory_points, epsilon)
        valid = np.isfinite(observed) & (observed > 0)
        occluded = valid & (observed < z_near - object_epsilon)
        supported = (
            valid
            & (observed >= z_near - object_epsilon)
            & (observed <= z_far + object_epsilon)
        )
        behind = valid & (observed > z_far + object_epsilon)
        codes = np.zeros(len(theory_points), dtype=np.int8)
        codes[occluded] = 1
        codes[supported] = 2
        codes[behind] = 3

        raw_map[v[valid], u[valid]] = observed[valid]
        closer = z_near < owner_near[v, u]
        selected_u = u[closer]
        selected_v = v[closer]
        near_map[selected_v, selected_u] = z_near[closer]
        owner_near[selected_v, selected_u] = z_near[closer]
        classification[selected_v, selected_u] = codes[closer]

    vmin, vmax = depth_limits(theory_depth, raw_depth)
    prefix = f"visibility_test{id}"
    save_visibility_set(
        sample,
        id,
        uv_list,
        labels,
        near_map,
        raw_map,
        classification,
        sample_dir,
        prefix,
        "All 3D boxes",
        vmin,
        vmax,
    )

pprint(compute_visibility(0), sort_dicts=False)

for i in range(0, 20):
    draw_visibility(i, merge = False)