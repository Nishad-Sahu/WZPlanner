#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove consecutive ego poses separated by less than a distance threshold."""

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Default: <input stem>_filtered.json",
    )
    parser.add_argument("--distance-threshold", type=float, default=0.1, help="Minimum 2D spacing in metres")
    return parser.parse_args()

def main():
    args = parse_args()
    output_file = args.output_file or args.input_file.with_name(f"{args.input_file.stem}_filtered.json")

    with args.input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "positions" not in data:
        raise ValueError("JSON format not recognized: missing 'positions' key")

    positions = data["positions"]
    print(f"Loaded {len(positions)} positions.")

    if not positions:
        print("No positions found; no output was written.")
        return

    # Always keep the first position
    filtered = [positions[0]]
    last_x, last_y = positions[0]["x"], positions[0]["y"]

    for p in positions[1:]:
        dx = p["x"] - last_x
        dy = p["y"] - last_y
        dist = math.hypot(dx, dy)
        if dist >= args.distance_threshold:
            filtered.append(p)
            last_x, last_y = p["x"], p["y"]

    removed = len(positions) - len(filtered)
    point_label = "point" if removed == 1 else "points"
    print(
        f"Retained {len(filtered)} positions "
        f"({removed} {point_label} closer than {args.distance_threshold} m removed)."
    )

    # Save result
    data["positions"] = filtered
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote filtered poses to {output_file}")

if __name__ == "__main__":
    main()

