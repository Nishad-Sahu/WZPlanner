#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactively record lane-boundary polylines in CARLA world coordinates."""

import argparse
import json
import threading
import time
from pathlib import Path

import carla
from pynput import mouse, keyboard

# -----------------------
# Config
# -----------------------
PICK_DISTANCE_M = 10.0   # distance ahead of spectator
Z_HEIGHT = 0.0           # ground plane Z for saved points

# Debug drawing
POINTER_COLOR = carla.Color(255, 140, 0)  # orange
POINT_COLOR   = carla.Color(0, 255, 0)    # green
LINE_COLOR    = carla.Color(0, 200, 255)  # cyan-ish
UNDO_COLOR    = carla.Color(255, 0, 255)  # magenta

POINTER_SIZE = 0.15
POINT_SIZE   = 0.12
LINE_THICKNESS = 0.08

# Set to 0.0 for "persistent-ish" (still refreshed). Use small >0 for fade.
MARKER_LIFETIME = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("lane_boundary_1.json"),
        help="Destination JSON file",
    )
    return parser.parse_args()

# -----------------------
# Helpers
# -----------------------
def clear_existing_marks(world):
    """Quickly tickle debug layer (limited clearing in CARLA debug draw)."""
    loc = world.get_spectator().get_transform().location
    world.debug.draw_point(loc, size=0.01, color=carla.Color(0, 0, 0), life_time=0.05)
    time.sleep(0.05)

def pointer_location(spectator_tf: carla.Transform) -> carla.Location:
    fwd = spectator_tf.get_forward_vector()
    return carla.Location(
        x=spectator_tf.location.x + PICK_DISTANCE_M * fwd.x,
        y=spectator_tf.location.y + PICK_DISTANCE_M * fwd.y,
        z=Z_HEIGHT
    )

def update_pointer(world, spectator, stop_flag):
    debug = world.debug
    while not stop_flag[0]:
        tf = spectator.get_transform()
        loc = pointer_location(tf)
        debug.draw_point(
            carla.Location(loc.x, loc.y, loc.z + 0.25),
            size=POINTER_SIZE,
            color=POINTER_COLOR,
            life_time=MARKER_LIFETIME
        )
        time.sleep(0.04)

def draw_current_polyline(world, pts):
    """Draw points and connecting lines for the current polyline."""
    dbg = world.debug
    if not pts:
        return
    # points
    for p in pts:
        dbg.draw_point(carla.Location(p.x, p.y, p.z + 0.15),
                       size=POINT_SIZE, color=POINT_COLOR, life_time=MARKER_LIFETIME)
    # lines
    if len(pts) >= 2:
        for a, b in zip(pts[:-1], pts[1:]):
            dbg.draw_line(
                carla.Location(a.x, a.y, a.z + 0.12),
                carla.Location(b.x, b.y, b.z + 0.12),
                thickness=LINE_THICKNESS,
                color=LINE_COLOR,
                life_time=MARKER_LIFETIME
            )

def draw_finished_polylines(world, polylines):
    """Redraw finished polylines while placement remains active."""
    dbg = world.debug
    for pts in polylines:
        if len(pts) < 2:
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            dbg.draw_line(
                carla.Location(a.x, a.y, a.z + 0.10),
                carla.Location(b.x, b.y, b.z + 0.10),
                thickness=LINE_THICKNESS,
                color=carla.Color(120, 255, 120),
                life_time=MARKER_LIFETIME
            )

def locs_to_json_points(locs):
    return [[float(p.x), float(p.y), float(p.z)] for p in locs]

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    spectator = world.get_spectator()

    clear_existing_marks(world)

    print("\nLane-boundary polyline placement")
    print(f"Left click: add a point {PICK_DISTANCE_M:g} m ahead of the spectator")
    print("G: finalize the current polyline")
    print("Backspace: remove the latest point")
    print("C: clear all polylines and start placement")
    print("P: pause placement")
    print("Q: save and exit\n")

    # State
    finished = []     # list[list[carla.Location]]
    current = []      # list[carla.Location]
    active, stop = [False], [False]

    # Pointer thread
    pointer_thread = threading.Thread(target=update_pointer, args=(world, spectator, stop), daemon=True)
    pointer_thread.start()

    def finalize_polyline():
        nonlocal current, finished
        if len(current) >= 2:
            finished.append(current)
            print(f"Finalized polyline {len(finished)} with {len(current)} points.")
        else:
            print("WARNING: A polyline requires at least two points.")
        current = []

    def save_and_exit():
        nonlocal current, finished
        # Include a valid unfinished polyline before writing the output.
        if len(current) >= 2:
            finished.append(current)
            print(f"Included unfinished polyline {len(finished)} with {len(current)} points.")
            current = []

        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "map_name": world.get_map().name,
            "pick_distance_m": PICK_DISTANCE_M,
            "z_height": Z_HEIGHT,
            "polylines": [{"points_world": locs_to_json_points(poly)} for poly in finished]
        }
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {len(finished)} polylines to {args.output_file}\n")

        stop[0] = True
        return False  # stops keyboard listener

    # Keyboard listener
    def on_key_press(k):
        nonlocal current, finished
        try:
            key = k.char.lower()
        except AttributeError:
            key = None

        if key == "q":
            return save_and_exit()
        elif key == "c":
            print("Cleared all polylines. Left click to add points.")
            clear_existing_marks(world)
            finished.clear()
            current = []
            active[0] = True
        elif key == "p":
            active[0] = False
            print("Placement paused.")
        elif key == "g":
            finalize_polyline()
        elif k == keyboard.Key.backspace:
            if current:
                last = current.pop()
                world.debug.draw_point(carla.Location(last.x, last.y, last.z + 0.2),
                                       size=0.2, color=UNDO_COLOR, life_time=1.5)
                print(f"Removed point ({last.x:.2f}, {last.y:.2f}, {last.z:.2f})")
        return True

    key_listener = keyboard.Listener(on_press=on_key_press)
    key_listener.start()

    # Mouse listener
    def on_click(x, y, button, pressed):
        nonlocal current
        if pressed and active[0] and button == mouse.Button.left:
            tf = spectator.get_transform()
            loc = pointer_location(tf)

            current.append(loc)
            print(f"Point {len(current)}: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")

    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()

    # Main draw loop (redraw lines/points)
    try:
        while not stop[0]:
            # continuously redraw current and finished polylines so they stay visible
            draw_finished_polylines(world, finished)
            draw_current_polyline(world, current)
            time.sleep(0.04)
    finally:
        try:
            key_listener.stop()
            mouse_listener.stop()
        except Exception:
            pass
        stop[0] = True
        print("Lane-boundary placement closed.")

if __name__ == "__main__":
    main()
