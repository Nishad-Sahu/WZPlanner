#!/usr/bin/env python3
"""
Visualize master annotation boundaries overlaid on the LiDAR point cloud.
Uses Open3D for 3D interactive visualization.

Usage:
    python viz_lidar_annotations.py <lidar.ply> <master_annotation.json> [--save <output.ply>]

Coordinate conventions:
    PLY (CARLA sensor-local, left-handed): x=forward, y=right, z=up
    Vehicle frame (annotations):           x=forward, y=right, z=up
    Sensor offset from vehicle origin:     x=+1.6, y=0, z=+1.7

    To convert vehicle to sensor-local:
        x_ply = x_veh - 1.6
        y_ply = y_veh
        z_ply = z_veh - 1.7
"""

import argparse
import json
import numpy as np

o3d = None

# -- sensor mount (same for LiDAR and camera in this setup) --
LIDAR_X, LIDAR_Y, LIDAR_Z = 1.6, 0.0, 1.7

# -- colours (RGB 0-1) --
COLORS = {
    "lane_boundary":       [1.0, 0.70, 0.0],   # orange
    "workzone_boundary":   [0.0, 0.84, 1.0],   # cyan
    "driving_center_line": [1.0, 0.0,  0.0],   # red
}

POINT_CLOUD_COLOR = [0.55, 0.55, 0.55]  # neutral grey for raw cloud


def vehicle_to_lidar(pts_vehicle: np.ndarray) -> np.ndarray:
    """Convert Nx3 points from vehicle frame to PLY sensor-local frame."""
    out = np.empty_like(pts_vehicle)
    out[:, 0] = pts_vehicle[:, 0] - LIDAR_X       # forward
    out[:, 1] = pts_vehicle[:, 1] - LIDAR_Y      # right
    out[:, 2] = pts_vehicle[:, 2] - LIDAR_Z       # up
    return out


def boundary_to_lidar_pts(coeffs, x_range, num_pts=300):
    """Sample a polynomial boundary in vehicle frame, return points in lidar frame."""
    x_vals = np.linspace(x_range[0], x_range[1], num=num_pts)
    y_vals = np.polyval(coeffs, x_vals)
    z_vals = np.zeros_like(x_vals)            # annotations lie on the ground plane
    pts_veh = np.column_stack([x_vals, y_vals, z_vals])
    return vehicle_to_lidar(pts_veh)


def build_tube(pts, color, radius=0.12, resolution=8):
    """Create a 'tube' mesh around a polyline so boundaries are clearly visible."""
    meshes = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        seg = p1 - p0
        length = np.linalg.norm(seg)
        if length < 1e-6:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=resolution, split=1)
        # Align cylinder (default axis = Z) with segment direction
        direction = seg / length
        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        if s < 1e-8:
            R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
        mid = (p0 + p1) / 2.0
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = mid
        cyl.transform(T)
        cyl.paint_uniform_color(color)
        meshes.append(cyl)
    return meshes


def main():
    global o3d

    parser = argparse.ArgumentParser(
        description="Overlay annotation boundaries on a LiDAR point cloud (Open3D)")
    parser.add_argument("ply", help="Path to lidar.ply")
    parser.add_argument("annotation", help="Path to master_annotation.json")
    parser.add_argument("--save", default=None,
                        help="Save coloured point cloud (boundaries as extra points) to this .ply path")
    parser.add_argument("--tube-radius", type=float, default=0.12,
                        help="Radius of the tube meshes drawn for each boundary")
    args = parser.parse_args()

    try:
        import open3d as open3d_module
    except ImportError as exc:
        raise SystemExit("open3d is required: python -m pip install open3d") from exc
    o3d = open3d_module

    # -- load point cloud --
    pcd = o3d.io.read_point_cloud(args.ply)
    n_pts = len(pcd.points)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(POINT_CLOUD_COLOR, (n_pts, 1)))
    print(f"Loaded point cloud: {n_pts} points")

    # -- load annotations --
    with open(args.annotation) as f:
        ann = json.load(f)

    geometries = [pcd]
    all_boundary_pts = []
    all_boundary_colors = []

    for boundary in ann["boundaries"]:
        b_type = boundary["boundary_type"]
        coeffs = boundary["coefficients"]
        x_range = boundary["y_range"]
        color = COLORS.get(b_type, [0.0, 1.0, 0.0])

        pts_3d = boundary_to_lidar_pts(coeffs, x_range)

        # Tube mesh for interactive view
        tubes = build_tube(pts_3d, color, radius=args.tube_radius)
        geometries.extend(tubes)

        # Also add as coloured points (for the --save option)
        all_boundary_pts.append(pts_3d)
        all_boundary_colors.append(np.tile(color, (len(pts_3d), 1)))

        print(f"  {b_type}: {len(pts_3d)} pts, "
              f"y_range [{x_range[0]:.1f}, {x_range[1]:.1f}]")

    # -- optional: save a merged coloured PLY --
    if args.save:
        merged_pts = np.vstack([np.asarray(pcd.points)] + all_boundary_pts)
        merged_clr = np.vstack([np.asarray(pcd.colors)] + all_boundary_colors)
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(merged_pts)
        merged.colors = o3d.utility.Vector3dVector(merged_clr)
        o3d.io.write_point_cloud(args.save, merged)
        print(f"Saved merged cloud -> {args.save}")

    # -- visualise --
    try:
        print("\nOpening viewer  (press Q or Esc to close) ...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="LiDAR + Annotations", width=1280, height=720)
        for g in geometries:
            vis.add_geometry(g)

        # Set a reasonable viewpoint: look from above-behind
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.3, -1.0])
        ctr.set_lookat([20.0, 0.0, -1.7])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.15)

        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.array([0.05, 0.05, 0.1])
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"\nWARNING: Could not open interactive viewer ({e}).")
        if not args.save:
            fallback = str(Path(args.ply).with_suffix("")) + "_annotated.ply"
            # Save merged cloud as fallback
            merged_pts = np.vstack([np.asarray(pcd.points)] + all_boundary_pts)
            merged_clr = np.vstack([np.asarray(pcd.colors)] + all_boundary_colors)
            merged = o3d.geometry.PointCloud()
            merged.points = o3d.utility.Vector3dVector(merged_pts)
            merged.colors = o3d.utility.Vector3dVector(merged_clr)
            o3d.io.write_point_cloud(fallback, merged)
            print(f"   Saved coloured PLY instead -> {fallback}")
        print("   Open the .ply file in CloudCompare, MeshLab, or Open3D on a GUI machine.")


if __name__ == "__main__":
    main()
