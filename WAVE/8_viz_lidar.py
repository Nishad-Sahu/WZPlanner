#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Display a LiDAR PLY file with Open3D."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path, help="Input LiDAR PLY file")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("open3d is required: python -m pip install open3d") from exc

    pcd = o3d.io.read_point_cloud(str(args.ply))

    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {args.ply}")

    print(pcd)

    # Apply a neutral color so uncolored point clouds remain visible.
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="LiDAR PLY Viewer",
        width=1280,
        height=720,
        point_show_normal=False
    )


if __name__ == "__main__":
    main()

