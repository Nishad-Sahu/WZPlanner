#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render trajectory-only overlays from an evaluation JSON file.

Trajectory selection rules:
1) Keep only trajectory predictions (boundary_type == driving_center_line).
2) If multiple trajectories exist for one frame, sort by |constant coefficient|.
   - closest to 0 gets id 0, next id 1, etc.
3) Filter out trajectories that are almost parallel to any previously kept trajectory.

The output directory contains rendered images, ``_summary.json``, and
``_missing_images.txt``.
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import cv2
import numpy as np

# Must match how RGB was rendered.
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
CAM_FOV = 90  # degrees

# Color palette in BGR; cycles by trajectory id.
PALETTE_BRG = [
    (0, 0, 255),      # red
    (0, 165, 255),    # orange
    (0, 255, 255),    # yellow
    (0, 255, 0),      # green
    (255, 255, 0),    # cyan
    (255, 0, 0),      # blue
    (255, 0, 255),    # magenta
    (128, 0, 255),
]
SCENARIO_RE = re.compile(r"^wz_(\d+)_")


def build_camera_intrinsics(width, height, fov_deg):
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def project_vehicle_pts_to_image(pts_vehicle, K, img_w, img_h, pitch_deg=15.0):
    if pts_vehicle.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    # Vehicle -> camera base with same offsets as existing pipeline.
    cam_base = np.column_stack(
        [pts_vehicle[:, 1], -pts_vehicle[:, 2] + 1.7, pts_vehicle[:, 0] - 1.6]
    ).astype(np.float32)

    pitch = math.radians(pitch_deg)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cam = np.column_stack(
        [
            cam_base[:, 0],
            cam_base[:, 1] * cos_p - cam_base[:, 2] * sin_p,
            cam_base[:, 1] * sin_p + cam_base[:, 2] * cos_p,
        ]
    ).astype(np.float32)

    z = cam[:, 2]
    m = z > 0.1
    if not np.any(m):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    cam = cam[m]
    z = cam[:, 2]

    img_homo = (K @ cam.T).T
    u = img_homo[:, 0] / img_homo[:, 2]
    v = img_homo[:, 1] / img_homo[:, 2]

    inb = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[inb].astype(np.int32)
    v = v[inb].astype(np.int32)
    depths = z[inb].astype(np.float32)
    return np.stack([u, v], axis=1), depths


def sample_boundary_vehicle(coeffs, x_range, num_pts=300):
    x0, x1 = float(x_range[0]), float(x_range[1])
    xs = np.linspace(x0, x1, num_pts, dtype=np.float32)
    ys = np.polyval(np.array(coeffs, dtype=np.float32), xs).astype(np.float32)
    zs = np.zeros_like(xs, dtype=np.float32)
    return np.column_stack([xs, ys, zs])


def poly_values(coeffs, xs):
    return np.polyval(np.array(coeffs, dtype=np.float64), xs)


def poly_slopes(coeffs, xs):
    p = np.poly1d(np.array(coeffs, dtype=np.float64))
    dp = np.polyder(p)
    return np.array(dp(xs), dtype=np.float64)


def pick_best_image_path(paths):
    """Prefer canonical non-viz/non-small image folders."""

    def score(p):
        norm = p.replace("\\", "/")
        parent = os.path.basename(os.path.dirname(norm))
        s = 0
        if "/images/" in norm:
            s += 100
        if parent == "images":
            s += 100
        if parent.startswith("images_") and "viz" not in parent and "small" not in parent:
            s += 60
        if "images_viz" in norm:
            s -= 120
        if "images_small" in norm:
            s -= 30
        return s, -len(norm), norm

    return max(paths, key=score)


def build_image_index(workspace_root):
    pattern = os.path.join(workspace_root, "wz_scenario_*", "**", "*_camera_raw.png")
    by_name = defaultdict(list)
    for path in glob.iglob(pattern, recursive=True):
        if os.path.isfile(path):
            by_name[os.path.basename(path)].append(path)

    index = {}
    for name, paths in by_name.items():
        index[name] = pick_best_image_path(paths)
    return index


def extract_scenario_id(image_name):
    if not image_name:
        return None
    m = SCENARIO_RE.match(os.path.basename(image_name))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def normalize_type(boundary_type):
    if boundary_type is None:
        return ""
    return str(boundary_type).strip().lower()


def is_trajectory_prediction(pred):
    return normalize_type(pred.get("boundary_type")) == "driving_center_line"


def extract_trajectories(predictions, negate_coeffs):
    trajectories = []
    for pred in predictions:
        if not is_trajectory_prediction(pred):
            continue
        coeffs = pred.get("coeffs")
        x_range = pred.get("range")
        if not isinstance(coeffs, list) or len(coeffs) < 1:
            continue
        if not isinstance(x_range, list) or len(x_range) != 2:
            continue
        coeffs = [float(c) for c in coeffs]
        if negate_coeffs:
            coeffs = [-c for c in coeffs]
        trajectories.append(
            {
                "coeffs": coeffs,
                "range": [float(x_range[0]), float(x_range[1])],
            }
        )
    return trajectories


def sort_trajectories_by_constant_coeff(trajectories):
    ordered = sorted(trajectories, key=lambda t: abs(float(t["coeffs"][-1])))
    for i, t in enumerate(ordered):
        t["traj_id"] = i
    return ordered


def are_almost_parallel(t1, t2, angle_thresh_deg, offset_std_thresh, min_overlap):
    # Evaluate on the overlapping x-range only.
    lo = max(float(t1["range"][0]), float(t2["range"][0]))
    hi = min(float(t1["range"][1]), float(t2["range"][1]))
    if not math.isfinite(lo) or not math.isfinite(hi) or (hi - lo) < min_overlap:
        return False

    xs = np.linspace(lo, hi, 80, dtype=np.float64)
    m1 = poly_slopes(t1["coeffs"], xs)
    m2 = poly_slopes(t2["coeffs"], xs)
    a1 = np.degrees(np.arctan(m1))
    a2 = np.degrees(np.arctan(m2))
    angle_diff = np.abs(a1 - a2)
    mean_angle_diff = float(np.mean(angle_diff))

    y1 = poly_values(t1["coeffs"], xs)
    y2 = poly_values(t2["coeffs"], xs)
    delta = y1 - y2
    offset_std = float(np.std(delta))

    return mean_angle_diff <= angle_thresh_deg and offset_std <= offset_std_thresh


def filter_parallel_trajectories(
    sorted_trajectories,
    angle_thresh_deg,
    offset_std_thresh,
    min_overlap,
):
    kept = []
    filtered = []
    for traj in sorted_trajectories:
        reject = False
        reject_against = None
        for prev in kept:
            if are_almost_parallel(
                traj,
                prev,
                angle_thresh_deg=angle_thresh_deg,
                offset_std_thresh=offset_std_thresh,
                min_overlap=min_overlap,
            ):
                reject = True
                reject_against = prev["traj_id"]
                break
        if reject:
            filtered.append({"traj_id": traj["traj_id"], "parallel_to": reject_against})
        else:
            kept.append(traj)
    return kept, filtered


def render_trajectory_overlay(rgb_path, kept_trajectories, out_path, thickness=3, num_pts=350):
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {rgb_path}")
    H, W = img.shape[:2]

    K = build_camera_intrinsics(CAM_WIDTH, CAM_HEIGHT, CAM_FOV)
    if (W, H) != (CAM_WIDTH, CAM_HEIGHT):
        sx = W / float(CAM_WIDTH)
        sy = H / float(CAM_HEIGHT)
        K = K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    for traj in kept_trajectories:
        coeffs = traj["coeffs"]
        x_range = traj["range"]
        tid = traj["traj_id"]
        pts_vehicle = sample_boundary_vehicle(coeffs, x_range, num_pts=num_pts)
        pix, _ = project_vehicle_pts_to_image(pts_vehicle, K, W, H, pitch_deg=15.0)
        if pix.shape[0] < 2:
            continue

        color = PALETTE_BRG[tid % len(PALETTE_BRG)]
        poly = pix.reshape(-1, 1, 2)
        cv2.polylines(
            img,
            [poly],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        p0 = tuple(int(v) for v in pix[0])
        cv2.putText(
            img,
            f"{tid}",
            p0,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(out_path, img)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-json",
        required=True,
        help="Path to all_eval_predictions.json",
    )
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root containing wz_scenario_* folders",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output folder for trajectory overlay images",
    )
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--num-pts", type=int, default=350)
    parser.add_argument(
        "--scenario-ids",
        nargs="*",
        type=int,
        default=None,
        help="Only process these scenario IDs parsed from image_name (e.g. 201 202).",
    )
    parser.add_argument(
        "--parallel-angle-deg",
        type=float,
        default=6.0,
        help="Mean angle-difference threshold in degrees for parallel filtering.",
    )
    parser.add_argument(
        "--parallel-offset-std",
        type=float,
        default=0.4,
        help="Std-dev threshold of lateral distance for parallel filtering.",
    )
    parser.add_argument(
        "--parallel-min-overlap",
        type=float,
        default=8.0,
        help="Minimum overlapping x-range length needed to compare trajectories.",
    )
    parser.add_argument(
        "--negate-coeffs",
        dest="negate_coeffs",
        action="store_true",
        help="Negate all coefficients before rendering.",
    )
    parser.add_argument(
        "--no-negate-coeffs",
        dest="negate_coeffs",
        action="store_false",
        help="Do not negate coefficients before rendering.",
    )
    parser.set_defaults(negate_coeffs=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick tests (0 = all)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.pred_json, "r") as f:
        data = json.load(f)

    image_index = build_image_index(args.workspace_root)
    print(f"Loaded predictions: {len(data)}")
    print(f"Indexed local images: {len(image_index)}")
    scenario_id_set = set(args.scenario_ids or [])

    rendered = 0
    missing = 0
    failed = 0
    filtered_out_by_scenario = 0
    missing_images = []
    entries_with_no_trajectory = 0
    entries_with_multi_trajectory = 0
    total_trajectories_before = 0
    total_trajectories_after = 0
    total_filtered_parallel = 0

    total = len(data) if args.limit <= 0 else min(len(data), args.limit)
    for i, item in enumerate(data[:total], start=1):
        image_name = item.get("image_name")
        preds = item.get("predictions", [])

        if not image_name:
            failed += 1
            continue
        scenario_id = extract_scenario_id(image_name)
        if scenario_id_set and scenario_id not in scenario_id_set:
            filtered_out_by_scenario += 1
            continue

        rgb_path = image_index.get(image_name)
        if rgb_path is None:
            missing += 1
            missing_images.append(image_name)
            if i % 200 == 0 or i == total:
                print(
                    f"[{i}/{total}] rendered={rendered} missing={missing} failed={failed} "
                    f"traj_before={total_trajectories_before} traj_after={total_trajectories_after}"
                )
            continue

        trajectories = extract_trajectories(preds, negate_coeffs=args.negate_coeffs)
        if len(trajectories) == 0:
            entries_with_no_trajectory += 1
        if len(trajectories) > 1:
            entries_with_multi_trajectory += 1

        total_trajectories_before += len(trajectories)
        sorted_trajs = sort_trajectories_by_constant_coeff(trajectories)
        kept, filtered = filter_parallel_trajectories(
            sorted_trajs,
            angle_thresh_deg=args.parallel_angle_deg,
            offset_std_thresh=args.parallel_offset_std,
            min_overlap=args.parallel_min_overlap,
        )
        total_trajectories_after += len(kept)
        total_filtered_parallel += len(filtered)

        base = (
            image_name[: -len("_camera_raw.png")]
            if image_name.endswith("_camera_raw.png")
            else os.path.splitext(image_name)[0]
        )
        out_name = f"{base}_camera_traj_viz.png"
        out_path = os.path.join(args.out_dir, out_name)

        try:
            render_trajectory_overlay(
                rgb_path,
                kept,
                out_path,
                thickness=args.thickness,
                num_pts=args.num_pts,
            )
            rendered += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] {image_name}: {exc}")

        if i % 200 == 0 or i == total:
            print(
                f"[{i}/{total}] rendered={rendered} missing={missing} failed={failed} "
                f"traj_before={total_trajectories_before} traj_after={total_trajectories_after}"
            )

    summary = {
        "pred_json": os.path.abspath(args.pred_json),
        "workspace_root": os.path.abspath(args.workspace_root),
        "out_dir": os.path.abspath(args.out_dir),
        "total_entries_seen": total,
        "rendered": rendered,
        "missing_images": missing,
        "filtered_out_by_scenario": filtered_out_by_scenario,
        "failed": failed,
        "entries_with_no_trajectory": entries_with_no_trajectory,
        "entries_with_multi_trajectory": entries_with_multi_trajectory,
        "total_trajectories_before_filtering": total_trajectories_before,
        "total_trajectories_after_filtering": total_trajectories_after,
        "total_filtered_as_parallel": total_filtered_parallel,
        "parallel_angle_deg": args.parallel_angle_deg,
        "parallel_offset_std": args.parallel_offset_std,
        "parallel_min_overlap": args.parallel_min_overlap,
        "negate_coeffs": args.negate_coeffs,
        "scenario_ids": sorted(scenario_id_set),
    }

    summary_path = os.path.join(args.out_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    missing_path = os.path.join(args.out_dir, "_missing_images.txt")
    with open(missing_path, "w") as f:
        for name in missing_images:
            f.write(name + "\n")

    print("Done.")
    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")
    print(f"missing: {missing_path}")


if __name__ == "__main__":
    main()
