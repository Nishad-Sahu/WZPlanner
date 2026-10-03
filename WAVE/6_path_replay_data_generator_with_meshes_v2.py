#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay a route with mesh-enabled work-zone variants and depth capture."""

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

WORKZONE_SCENARIO_NUMBER = 0
MAX_EGO_POSITIONS        = None  # None processes all filtered ego poses

TIME_OPTIONS = ["day"]
RAIN_OPTIONS = ["no_rain"]

# Work-zone prop variants represented as (blueprint, label, yaw override).
#   (blueprint_id, log_name, yaw_deg_override or None)
WORKZONE_OBJECT_OPTIONS = []

# Work-zone anchor positions.
WZ_JSON_GLOB = str(DATASET_DIR / "workzone_objects_*.json")

# ------------------ Mesh assets (static.prop.mesh) ------------------
MESH_ASSET_PATHS = [
    "/Game/Carla/Static/GuardRail/SM_Secfence_03.SM_Secfence_03",
]

# Stable labels written to the frame metadata.
MESH_LOG_NAME_MAP = {
    "/Game/Carla/Static/GuardRail/SM_Secfence_03.SM_Secfence_03": "zipper_barrier",
}

MESH_SCALE_DEFAULT = 1.0
MESH_Z_OFFSET      = -0.1
MESH_YAW_DEG       = 90.0  # default yaw for meshes
MESH_YAW_MAP = {}
# -------------------------------------------------------------------


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
    parser.add_argument("--workzone-json-glob", default=None)
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


# ========================== HELPERS ================================
def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_ply(filename: Path, pts: np.ndarray):
    header = (f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n")
    np.savetxt(filename, pts, header=header, comments='', fmt='%.3f')

def transform_to_dict(tr: carla.Transform):
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
    count = 0
    while not q.empty():
        try:
            q.get_nowait()
            count += 1
        except queue.Empty:
            break
    return count

def get_latest(q, timeout=SENSOR_TIMEOUT):
    try:
        item = q.get(timeout=timeout)
    except queue.Empty:
        return None
    while not q.empty():
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
    return item

def apply_time_and_rain(world: carla.World, time_name: str, rain_name: str):
    t = (time_name or "").lower()
    r = (rain_name or "").lower()
    w = carla.WeatherParameters()

    if t == "day":
        w.sun_altitude_angle = 45.0
    elif t == "dusk":
        w.sun_altitude_angle = 10.0
    elif t == "night":
        w.sun_altitude_angle = -25.0
    else:
        w.sun_altitude_angle = 45.0

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

def get_matrix(tf: carla.Transform):
    r = tf.rotation
    pitch, yaw, roll = map(math.radians, [r.pitch, r.yaw, r.roll])

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    R = np.array([
        [cp*cy, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [cp*sy, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ], dtype=np.float64)

    T = np.array([tf.location.x, tf.location.y, tf.location.z], dtype=np.float64)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3]  = T
    return M

def build_camera_intrinsics(width, height, fov):
    fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float64)

def destroy_old_actors(world):
    keywords = ["vehicle", "sensor.camera", "sensor.lidar", "static.prop"]
    for actor in world.get_actors():
        if any(k in actor.type_id for k in keywords):
            try:
                actor.destroy()
            except Exception:
                pass


# =================== MESH-ONLY WORK-ZONE SPAWNING ===================
def load_wz_anchors(wz_glob: str):
    wz_files = sorted(glob.glob(wz_glob))
    anchors = []
    for wz_path in wz_files:
        try:
            data = json.load(open(wz_path))
            anchors.extend(data.get("objects", []))
        except Exception as e:
            print(f"WARNING: Failed to read {wz_path}: {e}")
    return anchors, wz_files

def spawn_mesh_variant(world, bp_lib, anchors, mesh_path: str,
                       scale: float = MESH_SCALE_DEFAULT,
                       z_offset: float = MESH_Z_OFFSET,
                       yaw_deg: float = None):
    """
    Spawn the SAME mesh at all anchor positions.
    Returns (actors, meta)
    """
    mesh_bp = bp_lib.find("static.prop.mesh")
    if mesh_bp is None:
        raise RuntimeError("Blueprint not found: static.prop.mesh")

    actors = []
    meta = []
    log_name = MESH_LOG_NAME_MAP.get(mesh_path, Path(mesh_path).stem)
    if yaw_deg is None:
        yaw_deg = float(MESH_YAW_MAP.get(mesh_path, MESH_YAW_DEG))

    for obj in anchors:
        bp = mesh_bp  # blueprint handle
        if bp.has_attribute("mesh_path"):
            bp.set_attribute("mesh_path", str(mesh_path))
        if bp.has_attribute("scale"):
            bp.set_attribute("scale", str(scale))
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", str(log_name))

        tr = carla.Transform(
            carla.Location(
                x=float(obj["x"]),
                y=float(obj["y"]),
                z=float(obj.get("z", 0.0)) + float(z_offset),
            ),
            carla.Rotation(yaw=float(yaw_deg))
        )

        a = world.try_spawn_actor(bp, tr)
        if a:
            actors.append(a)
            meta.append({
                "mesh_path": str(mesh_path),
                "log_name": str(log_name),
                "transform": transform_to_dict(tr),
            })

    return actors, meta


# =================== PROP WORK-ZONE SPAWNING ===================
def spawn_prop_variant(world, bp_lib, anchors, bp_id: str, log_name: str, yaw_deg_override=None):
    """
    Spawn a prop blueprint (static.prop.*) at all anchor positions.
    Returns (actors, meta)
    """
    bp = bp_lib.find(bp_id)
    if bp is None:
        raise RuntimeError(f"Blueprint not found: {bp_id}")

    actors = []
    meta = []
    for obj in anchors:
        yaw = float(yaw_deg_override) if yaw_deg_override is not None else float(obj.get("yaw", 0.0))
        tr = carla.Transform(
            carla.Location(
                x=float(obj["x"]),
                y=float(obj["y"]),
                z=float(obj.get("z", 0.0)),
            ),
            carla.Rotation(yaw=yaw)
        )
        a = world.try_spawn_actor(bp, tr)
        if a:
            actors.append(a)
            meta.append({
                "bp_id": bp_id,
                "log_name": log_name,
                "transform": transform_to_dict(tr),
            })
    return actors, meta

# ========================== MAIN ================================
def main():
    args = parse_args()
    configure_run(args)

    if not POSES_FILE.exists():
        raise FileNotFoundError(f"Missing {POSES_FILE}")
    poses = json.load(open(POSES_FILE))["positions"][:MAX_EGO_POSITIONS]

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    print(f"Map: {world.get_map().name}")

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TICK_DT
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    destroy_old_actors(world)
    world.tick()

    anchors, wz_files = load_wz_anchors(WZ_JSON_GLOB)
    if not anchors:
        raise RuntimeError(
            f"No anchor points found. Expected work-zone JSON files at: {WZ_JSON_GLOB}\n"
            f"Files found: {len(wz_files)}"
        )

    total_variants = len(WORKZONE_OBJECT_OPTIONS) + len(MESH_ASSET_PATHS)
    total_expected = len(poses) * len(TIME_OPTIONS) * len(RAIN_OPTIONS) * total_variants
    print(
        f"Replaying {len(poses)} poses x {len(TIME_OPTIONS)} times x "
        f"{len(RAIN_OPTIONS)} rain x {total_variants} configured work-zone "
        f"variants = {total_expected} total frames"
    )

    # Spawn ego with physics ON (needed for sensor init)
    ego_bp = bp_lib.filter('vehicle.mini.cooper_s')[0]
    ego = world.spawn_actor(ego_bp, pose_to_transform(poses[0], z_offset=0.3))
    ego.set_simulate_physics(True)

    # Sensors
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_WIDTH))
    cam_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
    cam_bp.set_attribute("fov", str(CAM_FOV))
    cam_bp.set_attribute("sensor_tick", "0.0")
    cam = world.spawn_actor(cam_bp, CAM_TRANSFORM, attach_to=ego)

    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(CAM_WIDTH))
    depth_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
    depth_bp.set_attribute("fov", str(CAM_FOV))
    depth_bp.set_attribute("sensor_tick", "0.0")
    depth_cam = world.spawn_actor(depth_bp, CAM_TRANSFORM, attach_to=ego)

    sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    sem_bp.set_attribute("image_size_x", str(CAM_WIDTH))
    sem_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
    sem_bp.set_attribute("fov", str(CAM_FOV))
    sem_bp.set_attribute("sensor_tick", "0.0")
    sem_cam = world.spawn_actor(sem_bp, CAM_TRANSFORM, attach_to=ego)

    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", str(LIDAR_CHANNELS))
    lidar_bp.set_attribute("range", str(LIDAR_RANGE))
    lidar_bp.set_attribute("points_per_second", str(LIDAR_PPS))
    lidar_bp.set_attribute("rotation_frequency", str(LIDAR_ROT_FREQ))
    lidar_bp.set_attribute("sensor_tick", "0.0")
    lidar = world.spawn_actor(lidar_bp, LIDAR_TRANSFORM, attach_to=ego)

    # Sensor queues
    cam_q, depth_q, sem_q, lidar_q = queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue()
    cam.listen(lambda img: cam_q.put(img))
    depth_cam.listen(lambda img: depth_q.put(img))
    sem_cam.listen(lambda img: sem_q.put(img))
    lidar.listen(lambda data: lidar_q.put(data))
# Warm-up
    print(f"Warming up ({WARMUP_TICKS} ticks with physics ON)...")
    for _ in range(WARMUP_TICKS):
        world.tick()
        time.sleep(0.02)

    drain_queue(cam_q); drain_queue(depth_q); drain_queue(sem_q); drain_queue(lidar_q)
    world.tick()
    if (get_latest(cam_q, timeout=5.0) is None or
        get_latest(depth_q, timeout=5.0) is None or
        get_latest(sem_q, timeout=5.0) is None or
        get_latest(lidar_q, timeout=5.0) is None):
        raise RuntimeError("Sensors not responding after warm-up.")
    print("RGB/Depth/Semantic + LiDAR verified alive")

    # Disable physics for teleport mode (fallback if needed)
    ego.set_simulate_physics(False)
    world.tick()
    drain_queue(cam_q); drain_queue(depth_q); drain_queue(sem_q); drain_queue(lidar_q)
    world.tick()
    physics_enabled = False
    if (get_latest(cam_q, timeout=3.0) is None or
        get_latest(depth_q, timeout=3.0) is None or
        get_latest(sem_q, timeout=3.0) is None or
        get_latest(lidar_q, timeout=3.0) is None):
        print("WARNING: One or more sensors stop with physics OFF - keeping physics ON")
        ego.set_simulate_physics(True)
        physics_enabled = True
        for _ in range(5):
            world.tick()
    else:
        print("Sensors remain active with physics disabled; using direct pose replay")

    mkdir(FRAMES_DIR)

    samples = []
    skipped = 0
    frame_id = 0

    wz_actors = []
    wz_meta = []

    def destroy_wz():
        nonlocal wz_actors, wz_meta
        for a in wz_actors:
            try:
                a.destroy()
            except Exception:
                pass
        wz_actors = []
        wz_meta = []
        world.tick()

    try:
        # Build list of variants: props first, then meshes
        variants = []
        for (bp_id, log_name, yaw_override) in WORKZONE_OBJECT_OPTIONS:
            variants.append(
                {
                    "kind": "prop",
                    "bp_id": bp_id,
                    "name": log_name,
                    "yaw_override": yaw_override,
                }
            )
        for mesh_path in MESH_ASSET_PATHS:
            variants.append(
                {
                    "kind": "mesh",
                    "mesh_path": mesh_path,
                    "name": MESH_LOG_NAME_MAP.get(mesh_path, Path(mesh_path).stem),
                }
            )

        for v in variants:
            destroy_wz()
            if v["kind"] == "mesh":
                mesh_path = v["mesh_path"]
                wz_name = v["name"]
                yaw_used = float(MESH_YAW_MAP.get(mesh_path, MESH_YAW_DEG))
                print(f"\n{'='*60}")
                print(f"Mesh work-zone variant: {wz_name} (yaw={yaw_used:.1f} deg)")
                print(f"    {mesh_path}\n{'='*60}")
                wz_actors, wz_meta = spawn_mesh_variant(world, bp_lib, anchors, mesh_path, yaw_deg=yaw_used)
                print(f"Spawned meshes: {len(wz_actors)}/{len(anchors)} anchors")
            else:
                bp_id = v["bp_id"]
                wz_name = v["name"]
                yaw_override = v["yaw_override"]
                yaw_txt = f"{yaw_override:.1f} deg" if yaw_override is not None else "from-json/0 deg"
                print(f"\n{'='*60}\nProp work-zone variant: {wz_name} (bp={bp_id}, yaw={yaw_txt})\n{'='*60}")
                wz_actors, wz_meta = spawn_prop_variant(
                    world,
                    bp_lib,
                    anchors,
                    bp_id,
                    wz_name,
                    yaw_deg_override=yaw_override,
                )
                print(f"Spawned props: {len(wz_actors)}/{len(anchors)} anchors")
            world.tick()

            for p_idx, p in enumerate(poses):
                target_tf = pose_to_transform(p)
                ego.set_transform(target_tf)
                if physics_enabled:
                    ego.set_target_velocity(carla.Vector3D(0,0,0))
                    ego.set_target_angular_velocity(carla.Vector3D(0,0,0))

                drain_queue(cam_q); drain_queue(depth_q); drain_queue(sem_q); drain_queue(lidar_q)
                for _ in range(SETTLE_TICKS):
                    world.tick()

                for time_name in TIME_OPTIONS:
                    for rain_name in RAIN_OPTIONS:
                        apply_time_and_rain(world, time_name, rain_name)
                        for _ in range(2):
                            world.tick()

                        captured = False
                        for attempt in range(MAX_RETRIES):
                            drain_queue(cam_q); drain_queue(depth_q); drain_queue(sem_q); drain_queue(lidar_q)
                            for _ in range(SETTLE_TICKS):
                                world.tick()
                            world.tick()

                            img = get_latest(cam_q, timeout=SENSOR_TIMEOUT)
                            dep = get_latest(depth_q, timeout=SENSOR_TIMEOUT)
                            sem = get_latest(sem_q, timeout=SENSOR_TIMEOUT)
                            pcl = get_latest(lidar_q, timeout=SENSOR_TIMEOUT)
                            if img is None or dep is None or sem is None or pcl is None:

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
                            )[:, :, :3][:, :, ::-1]
                            pts = np.frombuffer(pcl.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]

                            frame_dir = FRAMES_DIR / f"{frame_id:06d}"
                            mkdir(frame_dir)
                            Image.fromarray(rgb).save(frame_dir / "rgb.png")

                            # Depth (viewable) + Semantic (CityScapes palette)
                            dep.save_to_disk(str(frame_dir / "depth.png"))

                            try:
                                sem.convert(cc.CityScapesPalette)
                            except Exception:
                                pass
                            sem.save_to_disk(str(frame_dir / "semantic.png"))

                            to_ply(frame_dir / "lidar.ply", pts)

                            actual_tf = ego.get_transform()
                            cam_tf = cam.get_transform()
                            lidar_tf = lidar.get_transform()

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
                                "wz_kind": v["kind"],
                                "wz_bp_id": (v.get("bp_id") if v["kind"]=="prop" else None),
                                "wz_mesh_path": (v.get("mesh_path") if v["kind"]=="mesh" else None),
                                    "wz_mesh_yaw_deg": (
                                        float(MESH_YAW_MAP.get(v.get("mesh_path", ""), MESH_YAW_DEG))
                                        if v["kind"] == "mesh"
                                        else None
                                    ),
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
                                "wz_kind": v["kind"],
                                "wz_bp_id": (v.get("bp_id") if v["kind"]=="prop" else None),
                                "wz_mesh_path": (v.get("mesh_path") if v["kind"]=="mesh" else None),
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
        try:
            cam.stop(); depth_cam.stop(); sem_cam.stop(); lidar.stop()
        except Exception:
            pass
        try:
            cam.destroy(); depth_cam.destroy(); sem_cam.destroy(); lidar.destroy()
        except Exception:
            pass
        try:
            ego.destroy()
        except Exception:
            pass
        try:
            destroy_wz()
        except Exception:
            pass

        world.tick()

        ego_log = {
            "frames": len(samples),
            "workzone_scenario_number": WORKZONE_SCENARIO_NUMBER,
            "max_ego_positions": MAX_EGO_POSITIONS,
            "Time": TIME_OPTIONS,
            "Rain": RAIN_OPTIONS,
            "Prop_objects": [x[1] for x in WORKZONE_OBJECT_OPTIONS],
            "Mesh_objects": list(MESH_LOG_NAME_MAP.values()),
            "wz_spawned_last_variant": wz_meta,
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
