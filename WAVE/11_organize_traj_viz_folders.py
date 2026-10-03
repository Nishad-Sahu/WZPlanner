#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organize trajectory overlay images into folders inferred from file names.

Input file pattern:
  wz_<scenario_id>_<frame_id>_<time>_<rain>_<work_object>_camera_traj_viz.png

Creates folders inside input dir:
  scenario_<scenario_id>/<time>/<rain>/<work_object>/

By default, creates hard links to avoid duplicating file data.
"""

import argparse
import glob
import json
import os
import shutil

SUFFIX = "_camera_traj_viz.png"


def parse_name(filename):
    if not filename.endswith(SUFFIX):
        return None
    stem = filename[: -len(SUFFIX)]
    parts = stem.split("_")
    # Expected minimum:
    # wz, <scenario>, <frame>, <time>, <rain_part_1>, <rain_part_2>, <work_object...>
    if len(parts) < 7:
        return None
    if parts[0] != "wz":
        return None

    scenario_id = parts[1]
    frame_id = parts[2]
    time_of_day = parts[3]
    rain = "_".join(parts[4:6])
    work_object = "_".join(parts[6:])
    return scenario_id, frame_id, time_of_day, rain, work_object


def ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def place_file(src, dst, mode):
    if os.path.exists(dst):
        return "exists"

    ensure_parent(dst)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "move":
        shutil.move(src, dst)
        return "move"
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_camera_traj_viz.png files.",
    )
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy", "move"],
        default="hardlink",
        help="How to place files in organized folders.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    pattern = os.path.join(input_dir, "wz_*" + SUFFIX)
    files = sorted(glob.glob(pattern))

    placed = 0
    skipped_bad_name = 0
    already_exists = 0
    by_scenario = {}
    mode_counts = {}

    for src in files:
        name = os.path.basename(src)
        parsed = parse_name(name)
        if parsed is None:
            skipped_bad_name += 1
            continue
        scenario_id, _frame_id, time_of_day, rain, work_object = parsed
        dst = os.path.join(
            input_dir,
            f"scenario_{scenario_id}",
            time_of_day,
            rain,
            work_object,
            name,
        )

        result = place_file(src, dst, mode=args.mode)
        if result == "exists":
            already_exists += 1
            continue
        placed += 1
        by_scenario[scenario_id] = by_scenario.get(scenario_id, 0) + 1
        mode_counts[result] = mode_counts.get(result, 0) + 1

    summary = {
        "input_dir": input_dir,
        "mode": args.mode,
        "files_found": len(files),
        "placed": placed,
        "already_exists": already_exists,
        "skipped_bad_name": skipped_bad_name,
        "mode_counts": mode_counts,
        "by_scenario": dict(sorted(by_scenario.items(), key=lambda kv: int(kv[0]))),
    }

    summary_path = os.path.join(input_dir, "_organized_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
