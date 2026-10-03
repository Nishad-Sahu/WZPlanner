#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record an ego-vehicle route for a WAVE or legacy Town01 scenario.

The recorder stores raw ego poses and a distance-filtered route while retaining
the interactive camera and manual-control workflow.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
from pygame.locals import (
    K_a,
    K_c,
    K_d,
    K_ESCAPE,
    K_p,
    K_q,
    K_r,
    K_s,
    K_SPACE,
    K_t,
    K_w,
)

import carla
from carla import ColorConverter as cc


DEFAULT_LEGACY_ROOT = Path("legacy_inputs")
DEFAULT_OUTPUT_ROOT = Path("results")
LEGACY_EGO_Y_OFFSET = -2.0
SPAWN_SNAP_SEARCH_RADIUS_M = 8.0
SPAWN_HEIGHT_OFFSET_M = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None, help="Optional world to load before recording")
    parser.add_argument("--scenario-dir", type=Path, default=None)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--legacy-scenario-id", type=int, default=None)
    parser.add_argument(
        "--legacy-spawn-dir",
        type=Path,
        default=DEFAULT_LEGACY_ROOT / "Workzone_generation_code" / "ego_vehicle_spawn_filesV3",
    )
    parser.add_argument("--route-policy", choices=["open_lane", "blocked_merge"], default="open_lane")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--record-interval-s", type=float, default=0.1)
    parser.add_argument("--filter-distance-m", type=float, default=0.1)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--ego-vehicle-filter", default="vehicle.tesla.model3")
    return parser.parse_args()


def transform_to_dict(transform: carla.Transform) -> dict:
    return {
        "x": transform.location.x,
        "y": transform.location.y,
        "z": transform.location.z,
        "pitch": transform.rotation.pitch,
        "yaw": transform.rotation.yaw,
        "roll": transform.rotation.roll,
    }


def main() -> None:
    args = parse_args()
    scenario_dir, spawn_transform, seed_metadata = resolve_scenario_and_spawn(args)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    pygame.init()
    pygame.display.set_caption("CARLA Manual Path Recorder")
    screen = pygame.display.set_mode((1280, 720))
    font = pygame.font.SysFont("monospace", 18)
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    if args.town and not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    spawn_transform, spawn_debug = stabilize_spawn_transform(world, spawn_transform)
    seed_metadata["resolved_spawn_transform"] = transform_to_dict(spawn_transform)
    seed_metadata["spawn_resolution"] = spawn_debug

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.filter(args.ego_vehicle_filter)[0]
    ego = world.try_spawn_actor(ego_bp, spawn_transform)
    if ego is None:
        raise RuntimeError(f"Failed to spawn ego vehicle at {spawn_transform}")

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("sensor_tick", "0.05")
    cam_transform = carla.Transform(carla.Location(x=1.4, z=1.6))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
    camera_queue: list[carla.Image] = []
    camera.listen(lambda img: camera_queue.append(img))

    spectator = world.get_spectator()
    positions: list[dict] = []
    last_log_time = -1e9
    recording = args.auto_start
    paused = False
    reverse_mode = False

    print("Controls:")
    print("  W/A/S/D + Space -> drive vehicle")
    print("  R -> toggle reverse")
    print("  T -> start recording")
    print("  P -> pause recording")
    print("  C -> resume recording")
    print("  Q or ESC -> stop and save\n")
    if args.auto_start:
        print("Recording started automatically.\n")
    else:
        print("Press T to start recording.\n")

    try:
        running = True
        while running:
            world.tick()
            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_r:
                        reverse_mode = not reverse_mode
                        print("Reverse gear:", "ON" if reverse_mode else "OFF")
                    elif event.key == K_p and recording:
                        paused = True
                        print("Recording paused.")
                    elif event.key == K_c and recording:
                        paused = False
                        print("Recording resumed.")
                    elif event.key == K_q:
                        print("Stopping and saving data...")
                        running = False
                    elif event.key == K_t:
                        recording = True
                        paused = False
                        print("Recording started.")

            keys = pygame.key.get_pressed()
            throttle = 0.0
            steer = 0.0
            brake = 0.0
            hand_brake = False

            if keys[K_w]:
                throttle = 0.5
            if keys[K_s]:
                brake = 0.8
            if keys[K_a]:
                steer = -0.25
            if keys[K_d]:
                steer = 0.25
            if keys[K_SPACE]:
                hand_brake = True

            ego.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=brake,
                    hand_brake=hand_brake,
                    reverse=reverse_mode,
                )
            )

            if recording and not paused and (sim_time - last_log_time) >= args.record_interval_s:
                transform = ego.get_transform()
                positions.append(
                    {
                        "timestamp_sim": sim_time,
                        "timestamp_wall": datetime.now().isoformat(),
                        **transform_to_dict(transform),
                    }
                )
                last_log_time = sim_time

            if camera_queue:
                image = camera_queue[-1]
                image.convert(cc.Raw)
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                rgb = array[:, :, :3][:, :, ::-1]
                surface = pygame.surfarray.make_surface(np.rot90(rgb))
                screen.blit(surface, (0, 0))

            status = "STOPPED"
            if recording and not paused:
                status = "RECORDING"
            elif recording and paused:
                status = "PAUSED"
            gear = "REV" if reverse_mode else "FWD"
            label = font.render(
                f"{status} | Gear: {gear} | Poses: {len(positions)} | Q/ESC to quit",
                True,
                (255, 80, 80) if reverse_mode else (80, 255, 80),
            )
            screen.blit(label, (10, 10))
            pygame.display.flip()

            ego_transform = ego.get_transform()
            spectator.set_transform(
                carla.Transform(
                    ego_transform.location + carla.Location(z=2.5) + ego_transform.get_forward_vector() * -8,
                    ego_transform.rotation,
                )
            )
            clock.tick(30)
    finally:
        save_recording(
            scenario_dir=scenario_dir,
            positions=positions,
            filter_distance_m=args.filter_distance_m,
            seed_metadata=seed_metadata,
        )
        try:
            camera.stop()
        except RuntimeError:
            pass
        camera.destroy()
        ego.destroy()
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        pygame.quit()


def resolve_scenario_and_spawn(args: argparse.Namespace) -> tuple[Path, carla.Transform, dict]:
    if args.scenario_dir is not None:
        scenario_dir = args.scenario_dir.resolve()
        seed = load_seed_from_scenario_dir(scenario_dir)
        return scenario_dir, seed_to_transform(seed), {"source": str(scenario_dir), "mode": "scenario_dir"}

    if args.legacy_scenario_id is None:
        raise SystemExit("Provide either --scenario-dir or --legacy-scenario-id")

    scenario_name = args.scenario_name or f"wz_scenario_{args.legacy_scenario_id:03d}_manual_town01"
    scenario_dir = (args.output_root / scenario_name).resolve()
    scenario_dir.mkdir(parents=True, exist_ok=True)

    spawn_path = args.legacy_spawn_dir / f"ego_positions_wz_{args.legacy_scenario_id:03d}.json"
    payload = json.loads(spawn_path.read_text(encoding="utf-8"))
    route_points = extract_route_points(payload["spawn_positions"], args.route_policy)
    if not route_points:
        raise RuntimeError(f"No route points found in {spawn_path} for policy {args.route_policy}")
    seed = route_points[0]
    metadata = {
        "source": str(spawn_path),
        "mode": "legacy",
        "legacy_scenario_id": args.legacy_scenario_id,
        "route_policy": args.route_policy,
    }
    (scenario_dir / "scenario_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scenario_dir, legacy_seed_to_transform(seed), metadata


def load_seed_from_scenario_dir(scenario_dir: Path) -> dict:
    for candidate in ("ego_positions_seed.json", "ego_positions_filtered.json", "ego_positions.json"):
        path = scenario_dir / candidate
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        positions = payload.get("positions", [])
        if positions:
            return positions[0]
    raise FileNotFoundError(
        f"No ego_positions_seed.json / ego_positions_filtered.json / ego_positions.json found in {scenario_dir}"
    )


def extract_route_points(points: list[dict], route_policy: str) -> list[dict]:
    if route_policy == "open_lane":
        return [item for item in points if item.get("scenario_type") == "open_lane_driving"]
    if route_policy == "blocked_merge":
        return [
            item
            for item in points
            if item.get("scenario_type") in {"blocked_lane_pre_merge", "blocked_lane_merging"}
        ]
    raise KeyError(f"Unknown route_policy {route_policy}")


def legacy_seed_to_transform(seed: dict) -> carla.Transform:
    location = seed["spawn_location"]
    rotation = seed.get("spawn_rotation", {})
    return carla.Transform(
        carla.Location(
            x=float(location["x"]),
            y=-float(location["y"]) + LEGACY_EGO_Y_OFFSET,
            z=float(location.get("z", 0.3)) + SPAWN_HEIGHT_OFFSET_M,
        ),
        carla.Rotation(
            pitch=float(rotation.get("pitch", 0.0)),
            yaw=float(rotation.get("yaw", 0.0)),
            roll=float(rotation.get("roll", 0.0)),
        ),
    )


def seed_to_transform(seed: dict) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(seed["x"]),
            y=float(seed["y"]),
            z=float(seed.get("z", 0.3)),
        ),
        carla.Rotation(
            pitch=float(seed.get("pitch", 0.0)),
            yaw=float(seed.get("yaw", 0.0)),
            roll=float(seed.get("roll", 0.0)),
        ),
    )


def stabilize_spawn_transform(world: carla.World, requested: carla.Transform) -> tuple[carla.Transform, dict]:
    world_map = world.get_map()
    waypoint = world_map.get_waypoint(
        requested.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return requested, {"mode": "raw", "reason": "no_driving_waypoint_found"}

    snapped_location = waypoint.transform.location + carla.Location(z=SPAWN_HEIGHT_OFFSET_M)
    snapped_rotation = carla.Rotation(
        pitch=waypoint.transform.rotation.pitch,
        yaw=waypoint.transform.rotation.yaw,
        roll=waypoint.transform.rotation.roll,
    )
    snapped_transform = carla.Transform(snapped_location, snapped_rotation)
    snap_distance = requested.location.distance(waypoint.transform.location)

    debug = {
        "mode": "snapped_to_driving_waypoint",
        "snap_distance_m": snap_distance,
        "search_radius_m": SPAWN_SNAP_SEARCH_RADIUS_M,
    }
    return snapped_transform, debug


def filter_positions(positions: list[dict], threshold_m: float) -> list[dict]:
    if not positions:
        return []
    filtered = [positions[0]]
    last_x = positions[0]["x"]
    last_y = positions[0]["y"]
    for item in positions[1:]:
        dist = math.hypot(item["x"] - last_x, item["y"] - last_y)
        if dist >= threshold_m:
            filtered.append(item)
            last_x = item["x"]
            last_y = item["y"]
    return filtered


def save_recording(scenario_dir: Path, positions: list[dict], filter_distance_m: float, seed_metadata: dict) -> None:
    raw_path = scenario_dir / "ego_positions.json"
    filtered_path = scenario_dir / "ego_positions_filtered.json"
    raw_payload = {"positions": positions, "recording_metadata": seed_metadata}
    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    filtered_positions = filter_positions(positions, filter_distance_m)
    filtered_payload = {
        "positions": filtered_positions,
        "recording_metadata": {
            **seed_metadata,
            "filter_distance_m": filter_distance_m,
            "raw_position_count": len(positions),
            "filtered_position_count": len(filtered_positions),
        },
    }
    filtered_path.write_text(json.dumps(filtered_payload, indent=2), encoding="utf-8")
    print(f"Saved {len(positions)} raw poses to {raw_path}")
    print(f"Saved {len(filtered_positions)} filtered poses to {filtered_path}")


if __name__ == "__main__":
    main()
