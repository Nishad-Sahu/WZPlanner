#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render boundary-prediction overlays from an evaluation JSON file.

Reads:
  - all_eval_predictions.json entries with:
      image_name, predictions[{boundary_type, coeffs, range}]
  - local raw images found under wz_scenario_*/**/*_camera_raw.png

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

# Colors in BGR for OpenCV.
BGR = {
    "lane_boundary": (0, 180, 255),
    "workzone_boundary": (255, 215, 0),
    "driving_center_line": (0, 0, 255),
}
SCENARIO_RE = re.compile(r"^wz_(\d+)_")


def build_camera_intrinsics(width, height, fov_deg):
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def project_vehicle_pts_to_image(pts_vehicle, K, img_w, img_h, pitch_deg=15.0):
    """Project vehicle-frame points to image pixels."""
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


def render_overlay(
    rgb_path,
    predictions,
    out_path,
    thickness=2,
    num_pts=350,
    negate_coeffs=False,
):
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

    for pred in predictions:
        coeffs = pred.get("coeffs")
        x_range = pred.get("range")
        btype = pred.get("boundary_type", "unknown")
        if not isinstance(coeffs, list) or len(coeffs) < 1:
            continue
        if not isinstance(x_range, list) or len(x_range) != 2:
            continue
        if negate_coeffs:
            coeffs = [-float(c) for c in coeffs]

        pts_vehicle = sample_boundary_vehicle(coeffs, x_range, num_pts=num_pts)
        pix, _ = project_vehicle_pts_to_image(pts_vehicle, K, W, H, pitch_deg=15.0)
        if pix.shape[0] < 2:
            continue

        color = BGR.get(btype, (0, 255, 0))
        poly = pix.reshape(-1, 1, 2)
        cv2.polylines(
            img,
            [poly],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
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
        help="Output folder for overlay images",
    )
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--num-pts", type=int, default=350)
    parser.add_argument(
        "--scenario-ids",
        nargs="*",
        type=int,
        default=None,
        help="Only process these scenario IDs parsed from image_name (e.g. 201 202).",
    )
    parser.add_argument(
        "--negate-coeffs",
        action="store_true",
        help="Negate all prediction polynomial coefficients before rendering.",
    )
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
    empty_predictions = 0
    filtered_out_by_scenario = 0
    missing_images = []

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
        if not preds:
            empty_predictions += 1

        rgb_path = image_index.get(image_name)
        if rgb_path is None:
            missing += 1
            missing_images.append(image_name)
            if i % 200 == 0 or i == total:
                print(
                    f"[{i}/{total}] rendered={rendered} missing={missing} "
                    f"empty={empty_predictions} failed={failed}"
                )
            continue

        base = (
            image_name[:-len("_camera_raw.png")]
            if image_name.endswith("_camera_raw.png")
            else os.path.splitext(image_name)[0]
        )
        out_name = f"{base}_camera_pred_viz.png"
        out_path = os.path.join(args.out_dir, out_name)

        try:
            render_overlay(
                rgb_path,
                preds,
                out_path,
                thickness=args.thickness,
                num_pts=args.num_pts,
                negate_coeffs=args.negate_coeffs,
            )
            rendered += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] {image_name}: {exc}")

        if i % 200 == 0 or i == total:
            print(
                f"[{i}/{total}] rendered={rendered} missing={missing} "
                f"empty={empty_predictions} failed={failed}"
            )

    summary = {
        "pred_json": os.path.abspath(args.pred_json),
        "workspace_root": os.path.abspath(args.workspace_root),
        "out_dir": os.path.abspath(args.out_dir),
        "total_entries_seen": total,
        "rendered": rendered,
        "missing_images": missing,
        "entries_with_empty_predictions": empty_predictions,
        "filtered_out_by_scenario": filtered_out_by_scenario,
        "failed": failed,
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
