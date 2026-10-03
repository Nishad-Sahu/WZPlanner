#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay a recorded route and capture synchronized CARLA sensor data.

The vehicle is warmed up with physics enabled and then advanced through the
recorded poses in synchronous mode. Each sample stores the measured vehicle
pose together with the corresponding sensor frame.
"""

import argparse
import glob
import json
import math
import queue
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import carla
from carla import ColorConverter as cc


# ========================== CONFIGURATION ==========================
POSES_FILE      = Path("ego_positions_filtered.json")
DATASET_DIR     = Path(".")
FRAMES_DIR      = DATASET_DIR / "frames"

CAM_WIDTH       = 1920
CAM_HEIGHT      = 1080
CAM_FOV         = 90
CAM_TRANSFORM   = carla.Transform(
                      carla.Location(x=1.6, z=1.7),
                      carla.Rotation(pitch=-15))

LIDAR_CHANNELS  = 64
LIDAR_RANGE     = 100
LIDAR_PPS       = 300_000
LIDAR_ROT_FREQ  = 10
LIDAR_TRANSFORM = carla.Transform(carla.Location(x=1.6, z=1.7))

TICK_DT         = 0.1
WARMUP_TICKS    = 15
SETTLE_TICKS    = 3       # ticks after teleport before capture tick
SENSOR_TIMEOUT  = 5.0     # seconds to wait for sensor data
MAX_RETRIES     = 3       # retry capture if sensors don't respond
POSE_TOLERANCE  = 0.10    # metres - warn if actual pose deviates
# ------------------ Dataset meta (written into ego_log.json) ------------------
WORKZONE_SCENARIO_NUMBER = 0
MAX_EGO_POSITIONS        = None  # None processes all filtered ego poses

TIME_OPTIONS = ["day", "dusk", "night"]                 # written to ego_log as bracket names
RAIN_OPTIONS = ["no_rain", "medium_rain", "heavy_rain"] # written to ego_log

# Work-zone object variants:
#   (blueprint_id, log_name, yaw_deg_override or None)
WORKZONE_OBJECT_OPTIONS = [
    ("static.prop.constructioncone", "traffic_cone", None),
    ("static.prop.streetbarrier",    "barrier",      90.0),
    ("static.prop.trafficcone01",    "traffic_barrel", None),
    ("static.prop.trafficcone02",    "grey_cone",    None),
]



# ---------- Master annotation generation (lanes) ----------
GENERATE_MASTER_ANNOTATIONS = True
LANE_LATERAL_MIN  = -15.0   # meters in ego(vehicle) frame
LANE_LATERAL_MAX  =  15.0
LANE_X_MIN        =  0.0    # forward range in vehicle frame used for fitting
LANE_X_MAX        =  70.0
LANE_DISCOVERY_X_MAX = 70.0 # forward range used to discover lanes
LANE_DISCOVERY_X_SAMPLES = 36
LANE_DISCOVERY_Y_STEP    = 1.0
LANE_WALK_STEP    = 2.0    # meters along lane for sampling
LANE_WALK_DIST    = 80.0   # meters forward/back from seed
LANE_FIT_DEG      = 3
LANE_DEDUP_COEFF_ROUND = 6
LANE_DEDUP_XR_ROUND    = 3



# ========================== HELPERS ================================
def mkdir(p):
    p.mkdir(parents=True, exist_ok=True)


def to_ply(filename, pts):
    header = (f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n")
    np.savetxt(filename, pts, header=header, comments='', fmt='%.3f')


def transform_to_dict(tr):
    return dict(
        x=tr.location.x, y=tr.location.y, z=tr.location.z,
        pitch=tr.rotation.pitch, yaw=tr.rotation.yaw, roll=tr.rotation.roll
    )


def pose_to_transform(p, z_offset=0.0):
    return carla.Transform(
        carla.Location(x=p["x"], y=p["y"], z=p["z"] + z_offset),
        carla.Rotation(pitch=p["pitch"], yaw=p["yaw"], roll=p["roll"])
    )


def drain_queue(q):
    """Discard all pending items, return count drained."""
    count = 0
    while not q.empty():
        try:
            q.get_nowait()
            count += 1
        except queue.Empty:
            break
    return count


def get_latest(q, timeout=SENSOR_TIMEOUT):
    """Block until at least one item arrives, then return the newest."""
    try:
        item = q.get(timeout=timeout)
    except queue.Empty:
        return None
    # Drain to the very latest
    while not q.empty():
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
    return item

# ========================== WEATHER / TIME HELPERS ==========================
def apply_time_and_rain(world: carla.World, time_name: str, rain_name: str):
    """
    Apply a simple mapping for (time of day, rain intensity).
    - time_name: day/dusk/night
    - rain_name: no_rain/medium_rain/heavy_rain
    """
    t = (time_name or "").lower()
    r = (rain_name or "").lower()

    w = carla.WeatherParameters()

    # Time of day via sun altitude
    if t == "day":
        w.sun_altitude_angle = 45.0
        w.sun_azimuth_angle = 0.0
    elif t == "dusk":
        w.sun_altitude_angle = 10.0
        w.sun_azimuth_angle = 0.0
    elif t == "night":
        w.sun_altitude_angle = -25.0
        w.sun_azimuth_angle = 0.0
    else:
        w.sun_altitude_angle = 45.0

    # Rain intensity
    if r == "no_rain":
        w.precipitation = 0.0
        w.precipitation_deposits = 0.0
        w.wetness = 0.0
        w.puddles = 0.0
        w.cloudiness = 10.0
    elif r == "medium_rain":
        w.precipitation = 50.0
        w.precipitation_deposits = 40.0
        w.wetness = 50.0
        w.puddles = 30.0
        w.cloudiness = 70.0
    elif r == "heavy_rain":
        w.precipitation = 90.0
        w.precipitation_deposits = 80.0
        w.wetness = 90.0
        w.puddles = 60.0
        w.cloudiness = 90.0
    else:
        w.precipitation = 0.0

    world.set_weather(w)
# ============================================================================



def verify_pose(ego, target_tf, idx, tol=POSE_TOLERANCE):
    actual = ego.get_transform()
    dx = actual.location.x - target_tf.location.x
    dy = actual.location.y - target_tf.location.y
    dz = actual.location.z - target_tf.location.z
    dist = (dx**2 + dy**2 + dz**2) ** 0.5
    if dist > tol:
        print(f"  WARNING: Frame {idx:06d}: pose drift {dist:.3f} m")
    return actual


def destroy_old_actors(world):
    keywords = ["vehicle", "sensor.camera", "sensor.lidar"]
    for actor in world.get_actors():
        if any(k in actor.type_id for k in keywords):
            try:
                actor.destroy()
            except RuntimeError:
                pass
                

# =================== WORK-ZONE OBJECT SPAWNING ===================
# Uses the same JSON format as 3_workzone_object_instantiator.py:
#   {"objects":[{"type":"cone"/"barrel", "x":..,"y":..,"z":..}, ...]}
# Spawns actors BEFORE path replay starts, and destroys them during cleanup.
# Paths/patterns remain the same as in the original scripts.
WZ_JSON_GLOB = str(DATASET_DIR / "workzone_objects_*.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Scenario output directory")
    parser.add_argument(
        "--poses-file",
        type=Path,
        default=None,
        help="Filtered ego poses; default: <dataset-dir>/ego_positions_filtered.json",
    )
    parser.add_argument("--workzone-scenario-number", type=int, required=True)
    parser.add_argument("--max-ego-positions", type=int, default=MAX_EGO_POSITIONS)
    parser.add_argument(
        "--workzone-json-glob",
        default=None,
        help="Optional glob for workzone_objects_*.json",
    )
    return parser.parse_args()


def configure_run(args: argparse.Namespace) -> None:
    global DATASET_DIR, FRAMES_DIR, POSES_FILE
    global WORKZONE_SCENARIO_NUMBER, MAX_EGO_POSITIONS, WZ_JSON_GLOB

    DATASET_DIR = args.dataset_dir
    FRAMES_DIR = DATASET_DIR / "frames"
    POSES_FILE = args.poses_file or DATASET_DIR / "ego_positions_filtered.json"
    WORKZONE_SCENARIO_NUMBER = args.workzone_scenario_number
    MAX_EGO_POSITIONS = args.max_ego_positions
    WZ_JSON_GLOB = args.workzone_json_glob or str(DATASET_DIR / "workzone_objects_*.json")

def _select_wz_blueprint(bp_lib, obj_type: str):
    t = (obj_type or "").lower()
    # Keep the exact mapping used in 3_workzone_object_instantiator.py
    if t == "cone":
        return bp_lib.find("static.prop.trafficcone01")
    if t == "barrel":
        return bp_lib.find("static.prop.barrel")
    # Extra aliases (optional)
    if t in ["traffic_barrel", "trafficbarrel", "drum"]:
        return bp_lib.find("static.prop.barrel")
    if t in ["traffic_cone", "trafficcone"]:
        return bp_lib.find("static.prop.trafficcone01")
    return None

def spawn_workzone_objects(world, bp_lib, wz_glob: str | None = None):
    """
    Spawn work-zone props from all JSON files matching wz_glob.
    Returns list of spawned actors.
    """
    wz_glob = wz_glob or WZ_JSON_GLOB
    wz_files = sorted(glob.glob(wz_glob))
    if not wz_files:
        print(f"WARNING: No workzone_objects_*.json files found for pattern: {wz_glob}")
        return []

    spawned_actors = []
    for wz_path in wz_files:
        try:
            data = json.load(open(wz_path))
            objs = data.get("objects", [])
        except Exception as e:
            print(f"WARNING: Failed to read {wz_path}: {e}")
            continue

        print(f"\nLoading {len(objs)} work-zone objects from {wz_path} ...")
        spawned = 0
        for obj in objs:
            bp = _select_wz_blueprint(bp_lib, obj.get("type", ""))
            if bp is None:
                print(f"WARNING: Unknown type {obj.get('type')} - skipping.")
                continue

            tr = carla.Transform(
                carla.Location(x=float(obj["x"]), y=float(obj["y"]), z=float(obj.get("z", 0.0)) + 0.0)
            )
            actor = world.try_spawn_actor(bp, tr)
            if actor:
                spawned_actors.append(actor)
                spawned += 1

        print(f"File {Path(wz_path).name}: spawned {spawned}/{len(objs)} objects.")

    print(f"\nTotal spawned work-zone props: {len(spawned_actors)}")
    return spawned_actors
# ================================================================================

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_matrix(tf):
    r = tf.rotation
    pitch, yaw, roll = map(math.radians,
                           [r.pitch, r.yaw, r.roll])

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    R = np.array([
        [cp*cy, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [cp*sy, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

    T = np.array([tf.location.x,
                  tf.location.y,
                  tf.location.z])

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3]  = T
    return M


def transform_to_dict(tf):
    return {
        "x": tf.location.x,
        "y": tf.location.y,
        "z": tf.location.z,
        "pitch": tf.rotation.pitch,
        "yaw": tf.rotation.yaw,
        "roll": tf.rotation.roll
    }

def matrix_to_list(M):
    return M.tolist()


def build_camera_intrinsics(width, height, fov):
    fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return K

# ---------------------------------------------------------------------
# Lane annotation generation (within lateral band; includes opposite direction lanes)
# ---------------------------------------------------------------------
def _transform_points(M_4x4, pts_xyz):
    """Apply 4x4 matrix to Nx3 points."""
    pts_h = np.hstack([pts_xyz, np.ones((len(pts_xyz), 1), dtype=pts_xyz.dtype)])
    out = (M_4x4 @ pts_h.T).T
    return out[:, :3]


def _waypoint_key(wp):
    # Unique-ish lane identifier in CARLA
    return (wp.road_id, wp.section_id, wp.lane_id)


def _dedup_boundaries(boundaries,
                      coeff_round=LANE_DEDUP_COEFF_ROUND,
                      xr_round=LANE_DEDUP_XR_ROUND):
    """Remove duplicate lane boundaries by rounding coefficients/x_range."""
    out = []
    seen = set()
    for b in boundaries:
        coeffs = tuple(np.round(np.array(b["coefficients"], dtype=np.float64), coeff_round))
        xr = tuple(np.round(np.array(b["x_range"], dtype=np.float64), xr_round))
        key = (b.get("boundary_type", "unknown"), coeffs, xr)
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def _discover_lanes_in_band(carla_map, vehicle_to_world):
    """
    Discover all driving lanes in ego(vehicle) frame within y in [LANE_LATERAL_MIN, LANE_LATERAL_MAX]
    and x in [LANE_X_MIN, LANE_DISCOVERY_X_MAX].
    This includes opposite-direction lanes.
    """
    xs = np.linspace(LANE_X_MIN, LANE_DISCOVERY_X_MAX, LANE_DISCOVERY_X_SAMPLES, dtype=np.float64)
    ys = np.arange(LANE_LATERAL_MIN, LANE_LATERAL_MAX + 1e-6, LANE_DISCOVERY_Y_STEP, dtype=np.float64)

    lane_wps = {}
    # sample grid in vehicle frame -> world -> nearest driving lane waypoint
    for x in xs:
        for y in ys:
            p_veh = np.array([[x, y, 0.0]], dtype=np.float64)
            p_w = _transform_points(vehicle_to_world, p_veh)[0]
            loc = carla.Location(x=float(p_w[0]), y=float(p_w[1]), z=float(p_w[2] + 0.5))
            wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            lane_wps[_waypoint_key(wp)] = wp

    return list(lane_wps.values())


def _walk_lane_points_vehicle(carla_map, wp0, world_to_vehicle):
    """Walk forward/back from a seed waypoint and return sampled points in vehicle frame."""
    pts_world = []

    # forward
    wp = wp0
    dist = 0.0
    while dist < LANE_WALK_DIST:
        loc = wp.transform.location
        pts_world.append([loc.x, loc.y, loc.z])
        nxt = wp.next(LANE_WALK_STEP)
        if not nxt:
            break
        wp = nxt[0]
        dist += LANE_WALK_STEP

    # backward
    wp = wp0
    dist = 0.0
    while dist < LANE_WALK_DIST:
        prv = wp.previous(LANE_WALK_STEP)
        if not prv:
            break
        wp = prv[0]
        loc = wp.transform.location
        pts_world.append([loc.x, loc.y, loc.z])
        dist += LANE_WALK_STEP

    if not pts_world:
        return np.empty((0, 3), dtype=np.float64)

    pts_world = np.array(pts_world, dtype=np.float64)
    pts_vehicle = _transform_points(world_to_vehicle, pts_world)

    # keep only points within lateral band (and forward-ish for stability)
    m = (
        (pts_vehicle[:, 1] >= LANE_LATERAL_MIN) &
        (pts_vehicle[:, 1] <= LANE_LATERAL_MAX) &
        (pts_vehicle[:, 0] >= LANE_X_MIN) &
        (pts_vehicle[:, 0] <= LANE_X_MAX)
    )
    return pts_vehicle[m]


def _fit_lane_cubic_y_of_x(pts_vehicle):
    """Fit y = f(x) cubic for points in vehicle frame. Returns (coeffs, x_range) or None."""
    if pts_vehicle.shape[0] < 20:
        return None

    x = pts_vehicle[:, 0]
    y = pts_vehicle[:, 1]

    # sort by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove lateral outliers introduced by lane-walk sampling.
    y_med = np.median(y)
    m = np.abs(y - y_med) < 30.0
    x = x[m]
    y = y[m]
    if len(x) < 20:
        return None

    coeffs = np.polyfit(x, y, deg=LANE_FIT_DEG).tolist()
    xr = [float(x.min()), float(x.max())]
    return coeffs, xr


def generate_master_annotations_lane_band(carla_map, vehicle_to_world, world_to_vehicle):
    """
    Generate lane boundaries within lateral band around ego and deduplicate.
    Returns a list of boundary dicts (only lane_boundary).
    """
    boundaries = []
    lane_wps = _discover_lanes_in_band(carla_map, vehicle_to_world)

    for wp in lane_wps:
        pts_vehicle = _walk_lane_points_vehicle(carla_map, wp, world_to_vehicle)
        fit = _fit_lane_cubic_y_of_x(pts_vehicle)
        if fit is None:
            continue
        coeffs, xr = fit
        boundaries.append({
            "coefficients": coeffs,
            "x_range": xr,
            "boundary_type": "lane_boundary"
        })

    boundaries = _dedup_boundaries(boundaries)
    return boundaries

def spawn_workzone_objects_variant(world, bp_lib, blueprint_id: str, yaw_override, wz_glob: str | None = None):
    """
    Spawn work-zone props from all JSON files matching wz_glob, but ignore each object's type
    and use the provided blueprint_id for ALL objects.
    yaw_override: if not None, forces this yaw (deg) for every object (e.g., barrier = 90).
    Returns list of spawned actors.
    """
    wz_glob = wz_glob or WZ_JSON_GLOB
    wz_files = sorted(glob.glob(wz_glob))
    if not wz_files:
        print(f"WARNING: No workzone_objects_*.json files found for pattern: {wz_glob}")
        return []

    bp = bp_lib.find(blueprint_id)
    if bp is None:
        raise RuntimeError(f"Blueprint not found: {blueprint_id}")

    spawned_actors = []
    for wz_path in wz_files:
        try:
            data = json.load(open(wz_path))
            objs = data.get("objects", [])
        except Exception as e:
            print(f"WARNING: Failed to read {wz_path}: {e}")
            continue

        spawned = 0
        for obj in objs:
            yaw = float(yaw_override) if yaw_override is not None else 0.0
            tr = carla.Transform(
                carla.Location(x=float(obj["x"]), y=float(obj["y"]), z=float(obj.get("z", 0.0))),
                carla.Rotation(yaw=yaw)
            )
            actor = world.try_spawn_actor(bp, tr)
            if actor:
                spawned_actors.append(actor)
                spawned += 1
        print(f"{Path(wz_path).name}: spawned {spawned}/{len(objs)} with {blueprint_id}")

    print(f"Total spawned props for {blueprint_id}: {len(spawned_actors)}")
    return spawned_actors

# ========================== MAIN ===================================
def main():
    args = parse_args()
    configure_run(args)

    # --- load poses ---
    if not POSES_FILE.exists():
        raise FileNotFoundError(f"Missing {POSES_FILE}")
    poses = json.load(open(POSES_FILE))["positions"]
    poses = poses[:MAX_EGO_POSITIONS]   # limit to configured number of ego positions
    total_expected = len(poses) * len(TIME_OPTIONS) * len(RAIN_OPTIONS) * len(WORKZONE_OBJECT_OPTIONS)
    print(
        f"Replaying {len(poses)} poses x {len(TIME_OPTIONS)} times x "
        f"{len(RAIN_OPTIONS)} rain x {len(WORKZONE_OBJECT_OPTIONS)} "
        f"work-zone objects = {total_expected} total frames"
    )

    # --- connect ---
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    carla_map = world.get_map()
    print(f"Map: {world.get_map().name}")

    original_settings = world.get_settings()

    # --- synchronous mode ---
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TICK_DT
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    destroy_old_actors(world)
    world.tick()

    # ----------------------------------------------------------
    # Spawn ego with physics ON (needed for sensor initialisation)
    # ----------------------------------------------------------
    ego_bp = bp_lib.filter('vehicle.mini.cooper_s')[0]
    start_tf = pose_to_transform(poses[0], z_offset=0.3)
    ego = world.spawn_actor(ego_bp, start_tf)
    ego.set_simulate_physics(True)

    # ----------------------------------------------------------
    # Attach sensors
    # ----------------------------------------------------------
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_WIDTH))
    cam_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
    cam_bp.set_attribute("fov", str(CAM_FOV))
    cam_bp.set_attribute("sensor_tick", "0.0")
    cam = world.spawn_actor(cam_bp, CAM_TRANSFORM, attach_to=ego)

    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels",          str(LIDAR_CHANNELS))
    lidar_bp.set_attribute("range",             str(LIDAR_RANGE))
    lidar_bp.set_attribute("points_per_second",  str(LIDAR_PPS))
    lidar_bp.set_attribute("rotation_frequency", str(LIDAR_ROT_FREQ))
    lidar_bp.set_attribute("sensor_tick",       "0.0")
    lidar = world.spawn_actor(lidar_bp, LIDAR_TRANSFORM, attach_to=ego)

    # --- thread-safe queues ---
    cam_q   = queue.Queue()
    lidar_q = queue.Queue()
    cam.listen(lambda img: cam_q.put(img))
    lidar.listen(lambda data: lidar_q.put(data))

    # ----------------------------------------------------------
    # Warm-up with physics ON - sensors fully initialise
    # ----------------------------------------------------------
    print(f"Warming up ({WARMUP_TICKS} ticks with physics ON)...")
    for _ in range(WARMUP_TICKS):
        world.tick()
        time.sleep(0.02)

    # Verify sensors are alive before proceeding
    drain_queue(cam_q)
    drain_queue(lidar_q)
    world.tick()
    test_cam = get_latest(cam_q, timeout=5.0)
    test_lid = get_latest(lidar_q, timeout=5.0)
    if test_cam is None or test_lid is None:
        raise RuntimeError(
            f"Sensors not responding after warm-up! "
            f"Camera: {'OK' if test_cam else 'FAIL'}, "
            f"LiDAR: {'OK' if test_lid else 'FAIL'}")
    print("Both sensors verified alive")

    # ----------------------------------------------------------
    # Now disable physics so teleports stick perfectly
    # ----------------------------------------------------------
    physics_enabled = True
    ego.set_simulate_physics(False)
    world.tick()

    drain_queue(cam_q); drain_queue(lidar_q)
    world.tick()
    test_cam = get_latest(cam_q, timeout=3.0)
    test_lid = get_latest(lidar_q, timeout=3.0)

    if test_cam is None or test_lid is None:
        print("WARNING: Sensors stop with physics OFF - keeping physics ON")
        ego.set_simulate_physics(True)
        physics_enabled = True
        for _ in range(5):
            world.tick()
    else:
        physics_enabled = False
        print("Sensors remain active with physics disabled; using direct pose replay")

    drain_queue(cam_q); drain_queue(lidar_q)

    # ----------------------------------------------------------
    # Output setup
    # ----------------------------------------------------------
    mkdir(FRAMES_DIR)
    samples = []
    skipped = 0

    # ----------------------------------------------------------
    # Generate data: Work-zone object -> pose -> (time, rain)
    # Total frames = poses x time settings x rain settings x object variants.
    # ----------------------------------------------------------
    frame_id = 0
    wz_spawned_actors = []

    def destroy_wz():
        nonlocal wz_spawned_actors
        try:
            for a in wz_spawned_actors:
                try:
                    a.destroy()
                except Exception:
                    pass
            world.tick()
        except Exception:
            pass
        wz_spawned_actors = []

    try:
        for (bp_id, wz_name, yaw_override) in WORKZONE_OBJECT_OPTIONS:

            destroy_wz()
            print(f"\n{'='*60}\nWork-zone object variant: {wz_name} ({bp_id})\n{'='*60}")
            wz_spawned_actors = spawn_workzone_objects_variant(world, bp_lib, bp_id, yaw_override, WZ_JSON_GLOB)
            world.tick()

            for p_idx, p in enumerate(poses):
                print(f"  Pose {p_idx+1}/{len(poses)} | work_zone={wz_name} | frame_id={frame_id}")
                target_tf = pose_to_transform(p)

                # place ego once per pose
                ego.set_transform(target_tf)
                if physics_enabled:
                    ego.set_target_velocity(carla.Vector3D(0,0,0))
                    ego.set_target_angular_velocity(carla.Vector3D(0,0,0))
                drain_queue(cam_q); drain_queue(lidar_q)
                for _ in range(SETTLE_TICKS):
                    world.tick()

                for time_name in TIME_OPTIONS:
                    for rain_name in RAIN_OPTIONS:

                        apply_time_and_rain(world, time_name, rain_name)
                        for _ in range(2):
                            world.tick()

                        captured = False
                        for attempt in range(MAX_RETRIES):
                            drain_queue(cam_q); drain_queue(lidar_q)
                            for _ in range(SETTLE_TICKS):
                                world.tick()
                            world.tick()

                            img = get_latest(cam_q, timeout=SENSOR_TIMEOUT)
                            pcl = get_latest(lidar_q, timeout=SENSOR_TIMEOUT)

                            if img is None or pcl is None:
                                print(f"WARNING: frame {frame_id}: sensor timeout (attempt {attempt+1}/{MAX_RETRIES})")
                                if not physics_enabled:
                                    ego.set_simulate_physics(True)
                                    physics_enabled = True
                                    for _ in range(3):
                                        world.tick()
                                continue

                            img.convert(cc.Raw)
                            rgb = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
                                (CAM_HEIGHT, CAM_WIDTH, 4)
                            )[:, :, :3][:, :, ::-1]  # BGR to RGB
                            pts = np.frombuffer(pcl.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]

                            frame_dir = FRAMES_DIR / f"{frame_id:06d}"
                            mkdir(frame_dir)
                            Image.fromarray(rgb).save(frame_dir / "rgb.png")
                            to_ply(frame_dir / "lidar.ply", pts)

                            actual_tf = ego.get_transform()
                            cam_tf    = cam.get_transform()
                            lidar_tf  = lidar.get_transform()

                            vehicle_to_world = get_matrix(actual_tf)
                            world_to_vehicle = np.linalg.inv(vehicle_to_world)

                            world_to_camera  = np.linalg.inv(get_matrix(cam_tf))
                            world_to_lidar   = np.linalg.inv(get_matrix(lidar_tf))

                            vehicle_to_camera = world_to_camera @ vehicle_to_world
                            vehicle_to_lidar  = world_to_lidar @ vehicle_to_world

                            meta = {
                                "frame_id": frame_id,
                                "timestamp_wall": datetime.utcnow().isoformat(),
                                "workzone_scenario_number": WORKZONE_SCENARIO_NUMBER,
                                "Time": time_name,
                                "Rain": rain_name,
                                "Work_zone_object": wz_name,
                                "camera_intrinsic": build_camera_intrinsics(CAM_WIDTH, CAM_HEIGHT, CAM_FOV).tolist(),
                                "vehicle_to_world": vehicle_to_world.tolist(),
                                "world_to_vehicle": world_to_vehicle.tolist(),
                                "vehicle_to_camera": vehicle_to_camera.tolist(),
                                "world_to_camera": world_to_camera.tolist(),
                                "vehicle_to_lidar": vehicle_to_lidar.tolist(),
                                "world_to_lidar": world_to_lidar.tolist(),
                            }
                            with open(frame_dir / "meta.json", "w") as f:
                                json.dump(meta, f, indent=2)

                            samples.append({
                                "frame_id": frame_id,
                                "workzone_scenario_number": WORKZONE_SCENARIO_NUMBER,
                                "Time": time_name,
                                "Rain": rain_name,
                                "Work_zone_object": wz_name,
                                "ego_transform": transform_to_dict(actual_tf),
                                "camera_transform": transform_to_dict(cam_tf),
                                "lidar_transform": transform_to_dict(lidar_tf),
                            })

                            captured = True
                            break

                        if not captured:
                            skipped += 1

                        frame_id += 1

    finally:
        cam.stop(); lidar.stop()
        cam.destroy(); lidar.destroy()
        ego.destroy()
        destroy_wz()
        world.tick()

        ego_log = {
            "frames": len(samples),
            "workzone_scenario_number": WORKZONE_SCENARIO_NUMBER,
            "max_ego_positions": MAX_EGO_POSITIONS,
            "Time": TIME_OPTIONS,
            "Rain": RAIN_OPTIONS,
            "Work_zone_object": [x[1] for x in WORKZONE_OBJECT_OPTIONS],
            "total_expected_frames": total_expected,
            "samples": samples
        }
        with open(DATASET_DIR / "ego_log.json", "w") as f:
            json.dump(ego_log, f, indent=2)

        world.apply_settings(original_settings)

    print(f"\n{'='*60}")
    print(f"Replay complete - frames saved: {len(samples)} (skipped: {skipped})")
    print(f"Output: {DATASET_DIR}")


if __name__ == "__main__":
    main()

