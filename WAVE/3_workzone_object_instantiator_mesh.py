#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spawn one CARLA mesh type at every work-zone anchor in a scenario directory."""

import argparse
import glob
import json
import time
from pathlib import Path

import carla


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Folder containing workzone_objects_*.json files",
    )
    ap.add_argument(
        "--mesh-path",
        "--mesh_path",
        dest="mesh_path",
        type=str,
        required=True,
        help="Mesh asset path, e.g. /Game/Carla/Static/Fence/SM_WireFence.SM_WireFence",
    )
    ap.add_argument(
        "--yaw",
        type=float,
        required=True,
        help="Yaw in degrees to apply to each spawned mesh actor",
    )
    ap.add_argument(
        "--z-offset",
        "--z_offset",
        dest="z_offset",
        type=float,
        default=0.0,
        help="Optional z offset added to every spawn location (default: 0.0)",
    )
    ap.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA host (default: localhost)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA port (default: 2000)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Client timeout seconds (default: 10.0)",
    )
    ap.add_argument(
        "--role-name",
        "--role_name",
        dest="role_name",
        type=str,
        default="workzone_mesh",
        help="role_name attribute if supported by the blueprint (default: workzone_mesh)",
    )
    return ap.parse_args()


def load_locations_from_files(wz_files):
    """
    Expects each JSON to have either:
      - {"objects": [{"x":..,"y":..,"z":.., ...}, ...]}
    """
    locations = []
    for wz_path in wz_files:
        with open(wz_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        objs = data.get("objects", [])
        if not isinstance(objs, list):
            print(f"WARNING: {wz_path}: 'objects' is not a list - skipping file.")
            continue

        for obj in objs:
            if not isinstance(obj, dict):
                continue
            if not all(k in obj for k in ("x", "y", "z")):
                continue
            locations.append((float(obj["x"]), float(obj["y"]), float(obj["z"])))

    return locations


def main():
    args = parse_args()

    folder = args.folder.expanduser().resolve()
    pattern = str(folder / "workzone_objects_*.json")
    wz_files = sorted(glob.glob(pattern))

    if not wz_files:
        print(f"WARNING: No workzone_objects_*.json files found in: {folder}")
        return

    print(f"Folder: {folder}")
    print(f"Found {len(wz_files)} files matching: workzone_objects_*.json")

    locations = load_locations_from_files(wz_files)
    if not locations:
        print("WARNING: No valid (x,y,z) locations found across files - nothing to spawn.")
        return

    print(f"Total locations loaded: {len(locations)}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    # Optional clutter removal
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    bp_lib = world.get_blueprint_library()

    # Use mesh blueprint
    mesh_bp = bp_lib.find("static.prop.mesh")
    if mesh_bp is None:
        raise RuntimeError("Could not find blueprint: static.prop.mesh")

    # Set mesh path
    if mesh_bp.has_attribute("mesh_path"):
        mesh_bp.set_attribute("mesh_path", args.mesh_path)
    else:
        raise RuntimeError("static.prop.mesh blueprint does not have attribute 'mesh_path'")

    # Optional role_name
    if args.role_name and mesh_bp.has_attribute("role_name"):
        mesh_bp.set_attribute("role_name", args.role_name)

    total_actors = []
    spawned = 0
    failed = 0

    print(f"Spawning mesh: {args.mesh_path}")
    print(f"Yaw: {args.yaw} deg | z_offset: {args.z_offset}")

    for (x, y, z) in locations:
        tr = carla.Transform(
            carla.Location(x=x, y=y, z=z + args.z_offset),
            carla.Rotation(yaw=float(args.yaw)),
        )
        actor = world.try_spawn_actor(mesh_bp, tr)
        if actor:
            total_actors.append(actor)
            spawned += 1
        else:
            failed += 1

    print(f"\nSpawned meshes: {spawned}")
    print(f"WARNING: Failed spawns (collision/overlap): {failed}")
    print("Press Ctrl+C to destroy all spawned meshes.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCleaning up...")
        for a in total_actors:
            try:
                a.destroy()
            except Exception:
                pass
        print("All spawned mesh actors removed.")


if __name__ == "__main__":
    main()
