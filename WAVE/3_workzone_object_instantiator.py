#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instantiate work-zone objects in CARLA.

Supports two input modes:
1. Existing scenario folder containing workzone_objects_*.json files.
2. A legacy Town01 configuration converted into a WAVE scenario directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import carla


BLUEPRINT_BY_OBJECT_TYPE = {
    "cone": "static.prop.trafficcone01",
    "grey_cone": "static.prop.trafficcone01",
    "barrel": "static.prop.barrel",
    "barrier": "static.prop.streetbarrier",
}
LEGACY_OBJECT_Z_OFFSET = 0.05

DEFAULT_LEGACY_ROOT = Path("legacy_inputs")
DEFAULT_LEGACY_CONFIG_JSON = (
    DEFAULT_LEGACY_ROOT
    / "Workzone_generation_code"
    / "traffic_cones_single_boundary_lane_blockage_100_configs.json"
)
DEFAULT_OUTPUT_ROOT = Path("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None, help="Optional world to load before spawning")
    parser.add_argument("--scenario-dir", type=Path, default=None)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=DEFAULT_LEGACY_ROOT,
        help="Root of an optional legacy scenario export",
    )
    parser.add_argument(
        "--legacy-config-json",
        type=Path,
        default=DEFAULT_LEGACY_CONFIG_JSON,
        help="Legacy configuration JSON used with --legacy-config-id",
    )
    parser.add_argument("--legacy-config-id", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--scenario-name",
        default=None,
        help="Optional output scenario folder name when using --legacy-config-id",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep process alive and destroy spawned objects on Ctrl+C. By default, spawn and exit.",
    )
    parser.add_argument(
        "--write-actor-ids",
        action="store_true",
        help="Write spawned actor ids into scenario_dir/spawned_actor_ids.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_dir, wz_files = resolve_workzone_files(args)
    if not wz_files:
        raise SystemExit("No workzone_objects_*.json files found or generated.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    if args.town and not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    bp_lib = world.get_blueprint_library()

    total_actors: list[carla.Actor] = []
    for wz_path in wz_files:
        data = json.loads(wz_path.read_text(encoding="utf-8"))
        print(f"\nLoading {len(data.get('objects', []))} objects from {wz_path}")
        spawned = 0
        for obj in data.get("objects", []):
            blueprint_name = BLUEPRINT_BY_OBJECT_TYPE.get(str(obj.get("type", "")).lower())
            if blueprint_name is None:
                print(f"Skipping unknown object type {obj.get('type')}")
                continue
            blueprint = bp_lib.find(blueprint_name)
            transform = carla.Transform(
                carla.Location(
                    x=float(obj["x"]),
                    y=float(obj["y"]),
                    z=float(obj.get("z", 0.0)),
                )
            )
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is None:
                print(f"Failed to spawn {obj.get('type')} at ({obj.get('x')}, {obj.get('y')})")
                continue
            total_actors.append(actor)
            spawned += 1
        print(f"Spawned {spawned} objects from {wz_path.name}")

    if scenario_dir and args.write_actor_ids:
        actor_ids_path = scenario_dir / "spawned_actor_ids.json"
        actor_ids_path.write_text(
            json.dumps({"actor_ids": [actor.id for actor in total_actors]}, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote actor ids to {actor_ids_path}")

    print(f"\nTotal spawned actors: {len(total_actors)}")
    if not args.hold:
        print("Objects left in the world. Start manual recording next.")
        return

    print("Press Ctrl+C to destroy the spawned props.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCleaning up spawned props...")
        for actor in total_actors:
            try:
                actor.destroy()
            except RuntimeError:
                pass
        print("Cleanup complete.")


def resolve_workzone_files(args: argparse.Namespace) -> tuple[Path | None, list[Path]]:
    if args.scenario_dir is not None:
        scenario_dir = args.scenario_dir.resolve()
        files = sorted(scenario_dir.glob("workzone_objects_*.json"))
        return scenario_dir, files

    if args.legacy_config_id is None:
        raise SystemExit("Provide either --scenario-dir or --legacy-config-id")

    scenario_name = args.scenario_name or f"wz_scenario_{args.legacy_config_id:03d}_manual_town01"
    scenario_dir = (args.output_root / scenario_name).resolve()
    scenario_dir.mkdir(parents=True, exist_ok=True)

    config_payload = load_legacy_config(args.legacy_config_json, args.legacy_config_id)
    objects_payload = convert_legacy_config_to_objects(config_payload)
    objects_path = scenario_dir / "workzone_objects_1.json"
    objects_path.write_text(json.dumps(objects_payload, indent=2), encoding="utf-8")

    metadata = {
        "legacy_config_json": str(args.legacy_config_json),
        "configuration_id": args.legacy_config_id,
        "town": "Town01",
        "num_objects": len(objects_payload["objects"]),
    }
    (scenario_dir / "scenario_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared scenario folder {scenario_dir}")
    return scenario_dir, [objects_path]


def load_legacy_config(config_json: Path, config_id: int) -> dict:
    configs = json.loads(config_json.read_text(encoding="utf-8"))
    for item in configs:
        if int(item["configuration_id"]) == int(config_id):
            return item
    raise KeyError(f"Configuration id {config_id} not found in {config_json}")


def convert_legacy_config_to_objects(config_payload: dict) -> dict:
    objects = []
    seen: set[tuple[float, float]] = set()

    cone_groups = []
    if "cone_locations" in config_payload:
        cone_groups.append(config_payload["cone_locations"])
    if "inner_boundary_cones" in config_payload:
        cone_groups.append(config_payload["inner_boundary_cones"])
    if "outer_boundary_cones" in config_payload:
        cone_groups.append(config_payload["outer_boundary_cones"])

    for cone_group in cone_groups:
        for x, y in cone_group:
            carla_x = float(x)
            carla_y = -float(y)
            key = (round(carla_x, 4), round(carla_y, 4))
            if key in seen:
                continue
            seen.add(key)
            objects.append(
                {
                    "type": "cone",
                    "x": carla_x,
                    "y": carla_y,
                    "z": LEGACY_OBJECT_Z_OFFSET,
                }
            )
    return {"objects": objects}


if __name__ == "__main__":
    main()
