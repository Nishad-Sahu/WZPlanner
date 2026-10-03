#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate WAVE annotations without depth-based visibility filtering.

The script combines manual lane polylines, work-zone object locations, and the
recorded ego trajectory. Camera images and annotations are copied into flat
output directories with a common scenario and frame identifier.
"""

import argparse
import glob
import json
import math
import shutil
from pathlib import Path

import numpy as np
import carla
from scipy.signal import savgol_filter

# ---------------------------
# Config
# ---------------------------
DATASET_ROOT = Path(".")
FRAME_DIR    = DATASET_ROOT / "frames"
EGO_LOG      = DATASET_ROOT / "ego_log.json"

OUT_IMAGES      = DATASET_ROOT / "images"
OUT_ANNOTATIONS = DATASET_ROOT / "annotations"

LANE_JSON_PATTERNS = [
    "lane_boundaries_*.json",
    "lane_boundary_*.json",
]

LOOKAHEAD_DIST = 30.0
X_MIN, X_MAX = 0.0, 72.0
Y_BAND = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def configure_dataset(dataset_root: Path) -> None:
    global DATASET_ROOT, FRAME_DIR, EGO_LOG, OUT_IMAGES, OUT_ANNOTATIONS

    DATASET_ROOT = dataset_root
    FRAME_DIR = DATASET_ROOT / "frames"
    EGO_LOG = DATASET_ROOT / "ego_log.json"
    OUT_IMAGES = DATASET_ROOT / "images"
    OUT_ANNOTATIONS = DATASET_ROOT / "annotations"

# ---------------------------
# Geometry helpers
# ---------------------------
def get_matrix(tf: carla.Transform) -> np.ndarray:
    r = tf.rotation
    pitch, yaw, roll = map(math.radians, [r.pitch, r.yaw, r.roll])
    cy, sy, cp, sp, cr, sr = (
        math.cos(yaw), math.sin(yaw),
        math.cos(pitch), math.sin(pitch),
        math.cos(roll), math.sin(roll)
    )
    R = np.array([
        [cp*cy, cy*sp*sr - sy*cr, -cy*sp*cr - sy*sr],
        [cp*sy, sy*sp*sr + cy*cr, -sy*sp*cr + cy*sr],
        [sp,    -cp*sr,            cp*cr]
    ], dtype=np.float64)
    T = np.array([tf.location.x, tf.location.y, tf.location.z], dtype=np.float64).reshape((3, 1))
    M = np.eye(4, dtype=np.float64)
    M[:3, :3], M[:3, 3] = R, T[:, 0]
    return M

def transform_points(M_4x4: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    if pts_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    pts_h = np.hstack([pts_xyz, np.ones((len(pts_xyz), 1), dtype=np.float64)])
    out = (M_4x4 @ pts_h.T).T
    return out[:, :3]

# ---------------------------
# Fit cubic y(x)
# ---------------------------
def fit_poly_vehicle_smooth(points_xy: np.ndarray):
    pts = np.asarray(points_xy, dtype=np.float64)
    if len(pts) < 1:
        return None, None

    pts = pts[np.argsort(pts[:, 0])]

    s = np.zeros(len(pts), dtype=np.float64)
    s[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    if s[-1] < 1e-6:
        return None, None

    s_uniform = np.linspace(0, s[-1], len(pts))
    x_interp = np.interp(s_uniform, s, pts[:, 0])
    y_interp = np.interp(s_uniform, s, pts[:, 1])

    if len(y_interp) > 9:
        y_interp = savgol_filter(y_interp, 9, 3)

    coeffs = np.polyfit(x_interp, y_interp, 3)
    x_range = [float(np.min(x_interp)), float(np.max(x_interp))]
    return coeffs.tolist(), x_range

def dedup_lane_boundaries(boundaries, coeff_round=6, xr_round=3):
    out = []
    seen = set()
    for b in boundaries:
        if b.get("boundary_type") != "lane_boundary":
            out.append(b)
            continue
        coeffs = tuple(np.round(np.array(b["coefficients"], dtype=np.float64), coeff_round))
        xr = tuple(np.round(np.array(b["y_range"], dtype=np.float64), xr_round))
        key = (coeffs, xr)
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out

# ---------------------------
# Manual lane boundaries
# ---------------------------
def load_manual_lane_boundaries(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    polylines = None
    if isinstance(data, dict):
        if "polylines" in data:
            polylines = data["polylines"]
        elif "boundaries" in data:
            polylines = data["boundaries"]
        elif "lane_boundaries" in data:
            polylines = data["lane_boundaries"]
        elif "points_world" in data or "points" in data:
            polylines = [data]
        else:
            polylines = [data]
    elif isinstance(data, list):
        polylines = data
    else:
        return []

    out = []
    for p in polylines:
        pts = None
        if isinstance(p, dict):
            pts = p.get("points_world", None)
            if pts is None:
                pts = p.get("points", None)
        else:
            pts = p
        if pts is None:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            continue
        out.append(arr)
    return out

def load_all_manual_lane_boundaries(dataset_root: Path):
    files = []
    for pat in LANE_JSON_PATTERNS:
        files.extend(sorted(dataset_root.glob(pat)))

    uniq, seen = [], set()
    for p in files:
        if p.name in seen:
            continue
        seen.add(p.name)
        uniq.append(p)

    polylines_world = []
    for fp in uniq:
        try:
            polys = load_manual_lane_boundaries(fp)
            if polys:
                polylines_world.extend(polys)
                print(f"Loaded {len(polys)} lane polylines from {fp.name}")
        except Exception as e:
            print(f"WARNING: Skipping {fp.name}: {e}")
    return polylines_world

# ---------------------------
# Work-zone loader
# ---------------------------
def load_workzones():
    wz_files = sorted(glob.glob(str(DATASET_ROOT / "workzone_objects_*.json")))
    wz_all = []
    for f in wz_files:
        data = json.load(open(f))
        pts = np.array([[o["x"], o["y"], o["z"]] for o in data.get("objects", [])], dtype=np.float64)
        if len(pts) > 0:
            wz_all.append(pts)
            print(f"Loaded {len(pts)} work zone polylines from {f}")
    return wz_all

# ---------------------------
# Trajectory selection
# ---------------------------
def get_trajectory_points(poses, i, ego_tf):
    fwd = np.array([math.cos(math.radians(ego_tf.rotation.yaw)),
                    math.sin(math.radians(ego_tf.rotation.yaw))], dtype=np.float64)
    ego_xy = np.array([ego_tf.location.x, ego_tf.location.y], dtype=np.float64)
    pts = []
    for j in range(i, len(poses)):
        p = poses[j]["ego_transform"]
        rel = np.array([p["x"], p["y"]], dtype=np.float64) - ego_xy
        if np.dot(rel, fwd) < 0:
            continue
        if np.linalg.norm(rel) > LOOKAHEAD_DIST:
            break
        pts.append([p["x"], p["y"], p["z"]])
    return pts

# ---------------------------
# Image source selection
# ---------------------------
def find_frame_camera_image(frame_dir: Path) -> Path | None:
    """
    Tries common names first, then falls back to any PNG/JPG in the frame folder.
    """
    candidates = [
        frame_dir / "rgb.png",
        frame_dir / "camera_raw.png",
        frame_dir / "image.png",
    ]
    for c in candidates:
        if c.exists():
            return c

    # try patterns
    pats = ["*_camera_raw.png", "*.png", "*.jpg", "*.jpeg"]
    for pat in pats:
        hits = sorted(frame_dir.glob(pat))
        if hits:
            return hits[0]
    return None

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    configure_dataset(args.dataset_root)

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_ANNOTATIONS.mkdir(parents=True, exist_ok=True)

    ego_data = json.load(open(EGO_LOG))

    # Global fields (Time, Rain, Work_zone_object are now per-frame in meta.json)
    workzone_scenario_number = ego_data.get("workzone_scenario_number", ego_data.get("workzone_scenario", ""))

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"Connected to map: {world.get_map().name}")

    lane_boundaries_world = load_all_manual_lane_boundaries(DATASET_ROOT)
    if not lane_boundaries_world:
        raise RuntimeError(f"No lane boundary polylines found under {DATASET_ROOT} with patterns {LANE_JSON_PATTERNS}")
    print(f"Total lane polylines loaded: {len(lane_boundaries_world)}")

    wz_all = load_workzones()

    poses = ego_data.get("samples", [])
    print(f"Processing {len(poses)} samples from ego_log...")

    # If frame count mismatches, iterate by pose and look up its frame_id folder
    for idx, pose in enumerate(poses):
        frame_id = pose.get("frame_id", idx)
        frame_dir = FRAME_DIR / f"{int(frame_id):06d}"
        if not frame_dir.exists():
            # fallback: if folders are not zero-padded
            alt = FRAME_DIR / str(frame_id)
            if alt.exists():
                frame_dir = alt
            else:
                continue

        # Read per-frame conditions from meta.json
        meta_path = frame_dir / "meta.json"
        if not meta_path.exists():
            print(f"WARNING: No meta.json in {frame_dir}; skipping.")
            continue
        with open(meta_path, "r") as mf:
            frame_meta = json.load(mf)

        time_str = frame_meta.get("Time", "unknown")
        rain_str = frame_meta.get("Rain", "unknown")
        wz_obj   = frame_meta.get("Work_zone_object", "unknown")

        # base filename
        base = f"wz_{workzone_scenario_number}_{frame_id}_{time_str}_{rain_str}_{wz_obj}"
        ann_path = OUT_ANNOTATIONS / f"{base}.json"
        img_out_name = f"{base}_camera_raw.png"
        img_out_path = OUT_IMAGES / img_out_name

        # copy camera image
        src_img = find_frame_camera_image(frame_dir)
        if src_img is None:
            print(f"WARNING: No camera image found in {frame_dir}; skipping image copy.")
        else:
            shutil.copy2(src_img, img_out_path)

        # pose -> ego transform
        ego_tf_dict = pose["ego_transform"]
        ego_tf = carla.Transform(
            carla.Location(x=ego_tf_dict["x"], y=ego_tf_dict["y"], z=-ego_tf_dict["z"]),
            carla.Rotation(pitch=ego_tf_dict["pitch"], yaw=ego_tf_dict["yaw"], roll=ego_tf_dict["roll"])
        )
        world_2_vehicle = np.linalg.inv(get_matrix(ego_tf))

        # boundaries
        boundaries_json = []

        # 1) lane boundaries (manual)
        for pts_world in lane_boundaries_world:
            pts_vehicle = transform_points(world_2_vehicle, pts_world)
            mask = (
                (pts_vehicle[:, 0] > X_MIN) &
                (pts_vehicle[:, 0] < X_MAX) &
                (np.abs(pts_vehicle[:, 1]) < Y_BAND)
            )
            pts_front = pts_vehicle[mask]
            if len(pts_front) < 5:
                continue
            coeffs, xr = fit_poly_vehicle_smooth(pts_front[:, :2])
            if coeffs is None:
                continue
            boundaries_json.append({
                "coefficients": coeffs,
                "y_range": xr,
                "boundary_type": "lane_boundary"
            })

        # 2) work-zone boundaries
        for wz_pts in wz_all:
            pts_vehicle = transform_points(world_2_vehicle, wz_pts)
            mask = (
                (pts_vehicle[:, 0] > X_MIN) &
                (pts_vehicle[:, 0] < (X_MAX+20)) &
                (np.abs(pts_vehicle[:, 1]) < (Y_BAND+20))
            )
            pts_front = pts_vehicle[mask]
            if len(pts_front) < 1:
                continue
            coeffs, xr = fit_poly_vehicle_smooth(pts_front[:, :2])
            if coeffs is None:
                continue
            boundaries_json.append({
                "coefficients": coeffs,
                "y_range": xr,
                "boundary_type": "workzone_boundary"
            })

        # 3) trajectory (unchanged)
        ahead_points = get_trajectory_points(poses, idx, ego_tf)
        if len(ahead_points) >= 4:
            ahead_points = np.asarray(ahead_points, dtype=np.float64)
            pts_vehicle = transform_points(world_2_vehicle, ahead_points)
            coeffs, xr = fit_poly_vehicle_smooth(pts_vehicle[:, :2])
            if coeffs is not None:
                boundaries_json.append({
                    "coefficients": coeffs,
                    "y_range": xr,
                    "boundary_type": "driving_center_line"
                })

        boundaries_json = dedup_lane_boundaries(boundaries_json)

        ann = {
            "image_name": img_out_name,
            "metadata": {
                "frame_id": frame_id,
                "workzone_scenario_number": workzone_scenario_number,
                "Time": time_str,
                "Rain": rain_str,
                "Work_zone_object": wz_obj
            },
            "boundaries": boundaries_json,
            "total_boundaries": len(boundaries_json)
        }

        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(poses):
            print(f"{base}: saved annotation ({len(boundaries_json)} boundaries)")

    print("\nAnnotation generation complete.")
    print(f"  Images:      {OUT_IMAGES}")
    print(f"  Annotations: {OUT_ANNOTATIONS}")

if __name__ == "__main__":
    main()
