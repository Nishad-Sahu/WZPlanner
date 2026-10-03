#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactively place work-zone objects from the CARLA spectator view.

Left click places a cone at the pointer, Backspace removes the latest cone,
C clears and starts placement, P pauses, and Q saves the result.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import carla
from pynput import mouse, keyboard

Z_HEIGHT = 0.0
POINTER_DISTANCE_M = 10.0

CONE_BP_NAME = "static.prop.trafficcone01"   # fallback blueprint
POINTER_COLOR = carla.Color(255, 140, 0)     # orange
UNDO_COLOR = carla.Color(0, 255, 255)        # cyan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("workzone_objects.json"),
        help="Destination JSON file",
    )
    return parser.parse_args()


def clear_existing_marks(world):
    """Quickly clear debug layer."""
    loc = world.get_spectator().get_transform().location
    world.debug.draw_point(loc, size=0.01, color=carla.Color(0, 0, 0), life_time=0.05)
    time.sleep(0.05)


def update_pointer(world, spectator, stop_flag):
    """Continuously move a debug sphere 10 m ahead of spectator."""
    debug = world.debug
    while not stop_flag[0]:
        tf = spectator.get_transform()
        fwd = tf.get_forward_vector()
        pointer_loc = carla.Location(
            x=tf.location.x + POINTER_DISTANCE_M * fwd.x,
            y=tf.location.y + POINTER_DISTANCE_M * fwd.y,
            z=Z_HEIGHT + 0.25
        )
        debug.draw_point(pointer_loc, size=0.15,
                         color=POINTER_COLOR, life_time=0.1)
        time.sleep(0.05)


def main():
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    spectator = world.get_spectator()
    blueprint_lib = world.get_blueprint_library()

    clear_existing_marks(world)

    print("\nWork-zone object placement")
    print(f"Left click: place a cone {POINTER_DISTANCE_M:g} m ahead of the spectator")
    print("Backspace: remove the latest cone")
    print("C: clear and start placement")
    print("P: pause placement")
    print("Q: save and exit\n")

    # State
    points, actors = [], []
    active, stop = [False], [False]

    # Start pointer thread
    pointer_thread = threading.Thread(target=update_pointer,
                                      args=(world, spectator, stop),
                                      daemon=True)
    pointer_thread.start()

    # ---- Keyboard listener ----
    def on_key_press(k):
        try:
            key = k.char.lower()
        except AttributeError:
            key = None

        if key == "q":
            stop[0] = True
            return False
        elif key == "c":
            print("Placement started. Left click to add cones.")
            clear_existing_marks(world)
            for a in actors:
                try:
                    a.destroy()
                except RuntimeError:
                    pass
            points.clear()
            actors.clear()
            active[0] = True
        elif key == "p":
            active[0] = False
            print("Placement paused.")
        elif k == keyboard.Key.backspace:
            if actors:
                last_actor = actors.pop()
                last_loc = points.pop()
                try:
                    last_actor.destroy()
                    world.debug.draw_point(last_loc, size=0.2,
                                           color=UNDO_COLOR, life_time=2.0)
                except RuntimeError:
                    pass
                print(f"Removed cone at ({last_loc.x:.2f}, {last_loc.y:.2f})")

    key_listener = keyboard.Listener(on_press=on_key_press)
    key_listener.start()

    # ---- Mouse listener ----
    def on_click(x, y, button, pressed):
        if pressed and active[0] and button == mouse.Button.left:
            tf = spectator.get_transform()
            fwd = tf.get_forward_vector()
            loc = carla.Location(
                x=tf.location.x + POINTER_DISTANCE_M * fwd.x,
                y=tf.location.y + POINTER_DISTANCE_M * fwd.y,
                z=Z_HEIGHT
            )
            rot = carla.Rotation(pitch=0.0, yaw=tf.rotation.yaw, roll=0.0)
            tr = carla.Transform(loc, rot)
            bp = blueprint_lib.find(CONE_BP_NAME)
            actor = world.spawn_actor(bp, tr)
            actors.append(actor)
            points.append(loc)
            print(f"Cone {len(points)}: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")

    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()

    # ---- Main loop ----
    while not stop[0]:
        time.sleep(0.05)

    # ---- Save JSON ----
    data = {"objects": [
        {"type": "cone", "x": p.x, "y": p.y, "z": p.z}
        for p in points
    ]}
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(points)} cones to {args.output_file}\n")

    # Cleanup
    key_listener.stop()
    mouse_listener.stop()
    stop[0] = True
    print("Placement tool closed.")


if __name__ == "__main__":
    main()

