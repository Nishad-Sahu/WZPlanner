#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate WAVE annotations with depth-based visibility filtering.

For lane, work-zone, and trajectory points, the script projects each 3D point
to the image, decodes the raw depth value at that pixel, and compares the
reconstructed camera coordinate with the projected coordinate. Points within
the configured tolerance are retained for polynomial fitting.

Requires per-frame depth image at:
  frames/<frame_id>/depth.png   (raw depth encoding, NOT logarithmic)
"""

import argparse
import glob
import json
import math
import shutil
from pathlib import Path

import numpy as np
import cv2
import carla
from scipy.signal import savgol_filter

# ---------------------------
# Dataset paths are configured from --dataset-root.
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

LOOKAHEAD_DIST = 20.0
X_MIN, X_MAX = 0.0, 62.0
Y_BAND = 15.0

# These values match the camera configuration used by the replay scripts.
CAM_WIDTH  = 1920
CAM_HEIGHT = 1080
CAM_FOV    = 90.0
PITCH_DEG_FOR_PROJ = 15.0  # keep same as viz script

DEPTH_TOL_METERS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--depth-tolerance", type=float, default=DEPTH_TOL_METERS)
    return parser.parse_args()


def configure_dataset(dataset_root: Path, depth_tolerance: float) -> None:
    global DATASET_ROOT, FRAME_DIR, EGO_LOG, OUT_IMAGES, OUT_ANNOTATIONS
    global DEPTH_TOL_METERS

    DATASET_ROOT = dataset_root
    FRAME_DIR = DATASET_ROOT / "frames"
    EGO_LOG = DATASET_ROOT / "ego_log.json"
    OUT_IMAGES = DATASET_ROOT / "images"
    OUT_ANNOTATIONS = DATASET_ROOT / "annotations"
    DEPTH_TOL_METERS = depth_tolerance

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
# Camera intrinsics + projection (copied/adapted from 8_viz_camera_annotations.py)
# ---------------------------
def build_camera_intrinsics(width, height, fov_deg):
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)

def project_vehicle_pts_to_image_with_camcoords(pts_vehicle, K, img_w, img_h, pitch_deg=15.0):
    """
    Returns:
      pix: (N,2) int32
      cam: (N,3) float32  (camera coords used for projection, units meters)
    Uses the SAME convention as 8_viz_camera_annotations.py:
      cam_base = [Y, -Z+1.7, X-1.6], then pitch rotation.
    """
    if pts_vehicle.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

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
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

    cam = cam[m]
    img_homo = (K @ cam.T).T
    u = img_homo[:, 0] / img_homo[:, 2]
    v = img_homo[:, 1] / img_homo[:, 2]

    inb = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    if not np.any(inb):
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

    u = u[inb].astype(np.int32)
    v = v[inb].astype(np.int32)
    cam = cam[inb]
    return np.stack([u, v], axis=1), cam

# ---------------------------
# Depth decoding (RAW depth png from CARLA depth camera)
# ---------------------------
def decode_carla_depth_png_to_meters(depth_bgra_u8: np.ndarray) -> np.ndarray:
    """
    depth_bgra_u8: HxWx4 uint8 from cv2.imread(..., IMREAD_UNCHANGED)
    CARLA raw depth encoding: normalized = (R + G*256 + B*256^2) / (256^3 - 1)
    depth_m = 1000 * normalized
    """
    if depth_bgra_u8 is None or depth_bgra_u8.ndim != 3 or depth_bgra_u8.shape[2] < 3:
        raise ValueError("Depth image must be BGRA/RGBA with at least 3 channels.")

    B = depth_bgra_u8[:, :, 0].astype(np.uint32)
    G = depth_bgra_u8[:, :, 1].astype(np.uint32)
    R = depth_bgra_u8[:, :, 2].astype(np.uint32)

    depth_norm = (R + G * 256 + B * (256**2)).astype(np.float64) / float((256**3) - 1)
    depth_m = 1000.0 * depth_norm
    return depth_m

def depth_filter_points_vehicle(pts_vehicle_xyz: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                                pitch_deg: float, tol_m: float):
    """
    Projects 3D vehicle pts -> pixel, samples depth at pixel, backprojects to 3D cam coords,
    compares to the 'true' cam coords used by the projection. Keeps points with 3D error <= tol_m.
    Returns filtered pts_vehicle_xyz.
    """
    H, W = depth_m.shape[:2]
    pix, cam_true = project_vehicle_pts_to_image_with_camcoords(
        pts_vehicle_xyz.astype(np.float32), K, W, H, pitch_deg=pitch_deg
    )
    if pix.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    u = pix[:, 0]
    v = pix[:, 1]
    d = depth_m[v, u].astype(np.float32)  # meters

    # Backproject pixel + depth into camera coords:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u.astype(np.float32) - cx) / fx * d
    y = (v.astype(np.float32) - cy) / fy * d
    z = d
    cam_est = np.stack([x, y, z], axis=1)

    err = np.linalg.norm(cam_est - cam_true, axis=1)
    keep = err <= tol_m

    # Need to map kept points back to original pts list:
    # project_vehicle_pts_to_image_with_camcoords internally filtered points by z>0.1 and in-bounds,
    # Recompute the indices after filtering to preserve row alignment.
    # easiest: redo projection but keep indices.
    return _filter_with_indices(pts_vehicle_xyz, K, W, H, pitch_deg, depth_m, tol_m)

def _filter_with_indices(pts_vehicle_xyz: np.ndarray, K: np.ndarray, W: int, H: int,
                         pitch_deg: float, depth_m: np.ndarray, tol_m: float):
    pts = pts_vehicle_xyz.astype(np.float32)
    # Build cam coords for all points (same as projection)
    cam_base = np.column_stack([pts[:, 1], -pts[:, 2] + 1.7, pts[:, 0] - 1.6]).astype(np.float32)
    pitch = math.radians(pitch_deg)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cam = np.column_stack([
        cam_base[:, 0],
        cam_base[:, 1] * cos_p - cam_base[:, 2] * sin_p,
        cam_base[:, 1] * sin_p + cam_base[:, 2] * cos_p
    ]).astype(np.float32)

    z = cam[:, 2]
    m_z = z > 0.1
    if not np.any(m_z):
        return np.empty((0, 3), dtype=np.float64)

    idx_z = np.where(m_z)[0]
    cam_z = cam[m_z]

    img_homo = (K @ cam_z.T).T
    u = img_homo[:, 0] / img_homo[:, 2]
    v = img_homo[:, 1] / img_homo[:, 2]

    m_inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(m_inb):
        return np.empty((0, 3), dtype=np.float64)

    idx_inb = idx_z[np.where(m_inb)[0]]
    u_i = u[m_inb].astype(np.int32)
    v_i = v[m_inb].astype(np.int32)
    cam_true = cam[idx_inb]

    d = depth_m[v_i, u_i].astype(np.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u_i.astype(np.float32) - cx) / fx * d
    y = (v_i.astype(np.float32) - cy) / fy * d
    z = d
    cam_est = np.stack([x, y, z], axis=1)

    err = np.linalg.norm(cam_est - cam_true, axis=1)
    keep = err <= tol_m

    kept_indices = idx_inb[keep]
    return pts_vehicle_xyz[kept_indices].astype(np.float64)

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
            print(f"Loaded {len(pts)} work zone points from {f}")
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
    for c in [frame_dir / "rgb.png", frame_dir / "camera_raw.png", frame_dir / "image.png"]:
        if c.exists():
            return c
    hits = sorted(frame_dir.glob("*.png"))
    return hits[0] if hits else None

def load_frame_depth_meters(frame_dir: Path):
    depth_path = frame_dir / "depth.png"
    if not depth_path.exists():
        return None
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return None
    return decode_carla_depth_png_to_meters(depth_img)

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    configure_dataset(args.dataset_root, args.depth_tolerance)

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_ANNOTATIONS.mkdir(parents=True, exist_ok=True)

    ego_data = json.load(open(EGO_LOG))
    workzone_scenario_number = ego_data.get("workzone_scenario_number", ego_data.get("workzone_scenario", ""))

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"Connected to map: {world.get_map().name}")

    lane_boundaries_world = load_all_manual_lane_boundaries(DATASET_ROOT)
    if not lane_boundaries_world:
        raise RuntimeError(f"No lane boundary polylines found under {DATASET_ROOT} with patterns {LANE_JSON_PATTERNS}")

    wz_all = load_workzones()

    poses = ego_data.get("samples", [])
    print(f"Processing {len(poses)} samples from ego_log...")

    K = build_camera_intrinsics(CAM_WIDTH, CAM_HEIGHT, CAM_FOV)

    for idx, pose in enumerate(poses):
        frame_id = pose.get("frame_id", idx)
        frame_dir = FRAME_DIR / f"{int(frame_id):06d}"
        if not frame_dir.exists():
            alt = FRAME_DIR / str(frame_id)
            if alt.exists():
                frame_dir = alt
            else:
                continue

        meta_path = frame_dir / "meta.json"
        if not meta_path.exists():
            continue
        frame_meta = json.load(open(meta_path))

        time_str = frame_meta.get("Time", "unknown")
        rain_str = frame_meta.get("Rain", "unknown")
        wz_obj   = frame_meta.get("Work_zone_object", "unknown")

        base = f"wz_{workzone_scenario_number}_{frame_id}_{time_str}_{rain_str}_{wz_obj}"
        ann_path = OUT_ANNOTATIONS / f"{base}.json"
        img_out_name = f"{base}_camera_raw.png"
        img_out_path = OUT_IMAGES / img_out_name

        src_img = find_frame_camera_image(frame_dir)
        if src_img is not None:
            shutil.copy2(src_img, img_out_path)

        depth_m = load_frame_depth_meters(frame_dir)
        if depth_m is None:
        # Retain unfiltered points when a depth image is unavailable.
            depth_m = None

        ego_tf_dict = pose["ego_transform"]
        ego_tf = carla.Transform(
            carla.Location(x=ego_tf_dict["x"], y=ego_tf_dict["y"], z=-ego_tf_dict["z"]),
            carla.Rotation(pitch=ego_tf_dict["pitch"], yaw=ego_tf_dict["yaw"], roll=ego_tf_dict["roll"])
        )
        world_2_vehicle = np.linalg.inv(get_matrix(ego_tf))

        boundaries_json = []

        # 1) lane boundaries
        for pts_world in lane_boundaries_world:
            pts_vehicle = transform_points(world_2_vehicle, pts_world)
            mask = (
                (pts_vehicle[:, 0] > X_MIN) &
                (pts_vehicle[:, 0] < X_MAX) &
                (np.abs(pts_vehicle[:, 1]) < Y_BAND)
            )
            pts_front = pts_vehicle[mask]
            if len(pts_front) < 2:
                continue

            if depth_m is not None:
                pts_front = depth_filter_points_vehicle(pts_front, depth_m, K, PITCH_DEG_FOR_PROJ, DEPTH_TOL_METERS)
                if len(pts_front) < 2:
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
                (pts_vehicle[:, 0] < (X_MAX + 20)) &
                (np.abs(pts_vehicle[:, 1]) < (Y_BAND + 20))
            )
            pts_front = pts_vehicle[mask]
            if len(pts_front) < 1:
                continue

            if depth_m is not None:
                pts_front = depth_filter_points_vehicle(pts_front, depth_m, K, PITCH_DEG_FOR_PROJ, DEPTH_TOL_METERS+3)
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

        # 3) trajectory
        ahead_points = get_trajectory_points(poses, idx, ego_tf)
        if len(ahead_points) >= 4:
            ahead_points = np.asarray(ahead_points, dtype=np.float64)
            pts_vehicle = transform_points(world_2_vehicle, ahead_points)

            if depth_m is not None:
                pts_vehicle = depth_filter_points_vehicle(
                    pts_vehicle, depth_m, K, PITCH_DEG_FOR_PROJ, DEPTH_TOL_METERS
                )

            if len(pts_vehicle) >= 2:
                coeffs, xr = fit_poly_vehicle_smooth(pts_vehicle[:, :2])
                xr[0] =0
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
    print(f"Images: {OUT_IMAGES}")
    print(f"Annotations: {OUT_ANNOTATIONS}")

if __name__ == "__main__":
    main()
