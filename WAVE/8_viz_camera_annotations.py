#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render boundary annotations over camera images.

An image named ``<base>_camera_raw.png`` is paired with ``<base>.json`` in the
annotation directory. Rendered images are written as
``<base>_camera_viz.png``.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import cv2

# Camera intrinsics used by the WAVE capture scripts.
CAM_WIDTH  = 1920
CAM_HEIGHT = 1080
CAM_FOV    = 90  # degrees

# Colors in BGR for OpenCV
BGR = {
    "lane_boundary": (0, 180, 255),        # orange-ish
    "workzone_boundary": (255, 215, 0),    # cyan-ish
    "driving_center_line": (0, 0, 255),    # red
}

def build_camera_intrinsics(width, height, fov_deg):
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)

def project_vehicle_pts_to_image(pts_vehicle, K, img_w, img_h, pitch_deg=15.0):
    """
    Vehicle frame: X=forward, Y=right, Z=up
    Camera base:   X=right,   Y=down,  Z=forward

    The translation offsets match the camera mount used during WAVE capture.
    """
    if pts_vehicle.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    # Vehicle to camera-base axes (Y, -Z, X), including sensor offsets.
    cam_base = np.column_stack([
        pts_vehicle[:, 1],
        -pts_vehicle[:, 2] + 1.7,
        pts_vehicle[:, 0] - 1.6
    ]).astype(np.float32)

    pitch = math.radians(pitch_deg)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    cam = np.column_stack([
        cam_base[:, 0],
        cam_base[:, 1] * cos_p - cam_base[:, 2] * sin_p,
        cam_base[:, 1] * sin_p + cam_base[:, 2] * cos_p
    ]).astype(np.float32)

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

def render_overlay(rgb_path, ann_path, out_path, thickness=2, num_pts=350):
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {rgb_path}")
    H, W = img.shape[:2]

    with open(ann_path, "r") as f:
        ann = json.load(f)

    K = build_camera_intrinsics(CAM_WIDTH, CAM_HEIGHT, CAM_FOV)

    # If saved RGB resolution differs, scale K to actual image size
    if (W, H) != (CAM_WIDTH, CAM_HEIGHT):
        sx = W / float(CAM_WIDTH)
        sy = H / float(CAM_HEIGHT)
        K = K.copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy

    for b in ann.get("boundaries", []):
        btype = b.get("boundary_type", "unknown")
        coeffs = b["coefficients"]
        x_range = b["y_range"]

        pts_vehicle = sample_boundary_vehicle(coeffs, x_range, num_pts=num_pts)
        pix, _ = project_vehicle_pts_to_image(pts_vehicle, K, W, H, pitch_deg=15.0)
        if pix.shape[0] < 2:
            continue

        color = BGR.get(btype, (0, 255, 0))
        poly = pix.reshape(-1, 1, 2)
        cv2.polylines(img, [poly], isClosed=False, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)

    cv2.imwrite(out_path, img)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=350)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([
        path for path in args.images_dir.iterdir()
        if path.is_file() and path.name.lower().endswith("_camera_raw.png")
    ])

    total = len(img_files)
    ok = 0
    skipped = 0

    for i, rgb_path in enumerate(img_files):
        base = rgb_path.name[:-len("_camera_raw.png")]
        ann_path = args.annotations_dir / f"{base}.json"
        out_path = args.output_dir / f"{base}_camera_viz.png"

        if not ann_path.exists():
            skipped += 1
            continue

        try:
            render_overlay(
                str(rgb_path),
                str(ann_path),
                str(out_path),
                thickness=args.thickness,
                num_pts=args.num_points,
            )
            ok += 1
        except Exception as e:
            skipped += 1
            print(f"[WARN] {base}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[{i+1}/{total}] rendered={ok}, skipped={skipped}")

    print(f"Done. rendered={ok}, skipped={skipped}, output_dir={args.output_dir}")

if __name__ == "__main__":
    main()
