# WAVE Data Generation Toolkit

WAVE is the CARLA-based data-generation pipeline used to create the simulated portion of WorkZonePlan. The workflow places work-zone objects and lane boundaries, records an ego route, replays that route under controlled conditions, and produces synchronized camera, LiDAR, and boundary annotations.

## Workflow

1. Place work-zone objects.
2. Place lane boundaries.
3. Instantiate the scenario in CARLA.
4. Drive the route manually and record ego poses.
5. Filter closely spaced ego poses.
6. Replay the route and capture RGB and LiDAR data.
7. Generate per-frame annotations.
8. Inspect the generated camera and LiDAR outputs.

The scripts are stored directly in this directory. Run the commands below from `WAVE/`.

## Requirements

- CARLA 0.9.16, including a matching Python API installation
- Python 3.10 or a Python version supported by the installed CARLA package
- A graphical desktop session for the interactive placement, driving, and visualization tools

Install the remaining Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The `carla` package is intentionally not listed in `requirements.txt` because it must match the CARLA server installation. Install it from the CARLA 0.9.16 Python API package and verify it with:

```bash
python -c "import carla; print(carla.__file__)"
```

## Scenario Layout

Use one directory per scenario:

```text
results/
  wz_scenario_201/
    workzone_objects_1.json
    lane_boundary_1.json
    ego_positions.json
    ego_positions_filtered.json
    frames/
    ego_log.json
    images/
    annotations/
    images_viz/
```

The commands below use `results/wz_scenario_201` as an example. Replace `201` with the desired scenario identifier.

## Core Commands

### 1. Place Work-Zone Objects

```bash
python 1_gen_wz.py \
  --host localhost \
  --port 2000 \
  --output-file results/wz_scenario_201/workzone_objects_1.json
```

Controls: left click adds a cone; `Backspace` removes the latest cone; `C` clears and starts placement; `P` pauses; `Q` saves and exits.

### 2. Place Lane Boundaries

```bash
python 2_gen_lane.py \
  --host localhost \
  --port 2000 \
  --output-file results/wz_scenario_201/lane_boundary_1.json
```

Controls: left click adds a point; `G` finalizes the current polyline; `Backspace` removes the latest point; `C` clears all polylines; `P` pauses; `Q` saves and exits.

### 3. Spawn Work-Zone Objects

```bash
python 3_workzone_object_instantiator.py \
  --host localhost \
  --port 2000 \
  --scenario-dir results/wz_scenario_201
```

The process spawns the scenario objects and exits. Add `--hold` to keep the process active and remove the spawned actors on `Ctrl+C`.

### 4. Record the Ego Route

```bash
python 4_manual_path_recorder.py \
  --host localhost \
  --port 2000 \
  --scenario-dir results/wz_scenario_201
```

Controls: `W/A/S/D` drive; `Space` applies the hand brake; `R` toggles reverse; `T` starts recording; `P` pauses; `C` resumes; `Q` or `Esc` saves and exits.

### 5. Filter Ego Poses

```bash
python 5_ego_positions_filter.py \
  --input-file results/wz_scenario_201/ego_positions.json
```

The default output is `ego_positions_filtered.json` in the same directory. Use `--output-file` or `--distance-threshold` to override the defaults.

### 6. Replay and Capture Sensor Data

The canonical prop-based pipeline captures RGB, depth, and LiDAR data:

```bash
python 6_path_replay_data_generator.py \
  --host localhost \
  --port 2000 \
  --dataset-dir results/wz_scenario_201 \
  --workzone-scenario-number 201
```

For mesh-enabled scenarios without depth capture:

```bash
python 6_path_replay_data_generator_with_meshes_v2_without_depth.py \
  --host localhost \
  --port 2000 \
  --dataset-dir results/wz_scenario_201 \
  --workzone-scenario-number 201
```

`6_path_replay_data_generator_with_meshes_v2.py` is the corresponding mesh-enabled depth-capture variant. It accepts the same command-line options.

All replay commands process the complete filtered route by default. Pass `--max-ego-positions N` to limit a diagnostic run to the first `N` poses.

### 7. Generate Annotations

For captures without depth filtering:

```bash
python 7_generate_master_annotations_without_depth_comparison.py \
  --host localhost \
  --port 2000 \
  --dataset-root results/wz_scenario_201
```

Use `7_generate_master_annotations.py` with the same options when each frame contains a raw `depth.png` image and depth-based occlusion filtering is required.

### 8. Visualize Camera Annotations

```bash
python 8_viz_camera_annotations.py \
  --images-dir results/wz_scenario_201/images \
  --annotations-dir results/wz_scenario_201/annotations \
  --output-dir results/wz_scenario_201/images_viz
```

### 9. Visualize LiDAR

```bash
python 8_viz_lidar.py results/wz_scenario_201/frames/000000/lidar.ply
```

Overlay annotation boundaries on the point cloud with:

```bash
python 8_viz_lidar_annotations.py \
  results/wz_scenario_201/frames/000000/lidar.ply \
  results/wz_scenario_201/annotations/<frame_base>.json
```

## Script Variants

- `3_workzone_object_instantiator.py` is the primary object-instantiation entry point. The `_prop` and `_mesh` scripts expose the lower-level prop-only and mesh-only loaders used during dataset development.
- `6_path_replay_data_generator.py` is the primary replay path for prop-based scenarios.
- `6_path_replay_data_generator_with_meshes_v2.py` adds mesh variants and depth capture.
- `6_path_replay_data_generator_with_meshes_v2_without_depth.py` adds mesh variants without depth capture.
- `7_generate_master_annotations.py` performs depth-aware filtering; the `_without_depth_comparison` variant does not require depth images.

The primary instantiator and manual recorder also retain optional `--legacy-*` arguments for importing Town01 scenarios from an earlier dataset layout. Run either script with `--help` for the required legacy paths.

## Evaluation Visualization Utilities

- `9_viz_all_eval_predictions.py` renders all predicted boundary types from an evaluation JSON file.
- `10_viz_trajectory_predictions.py` renders trajectory predictions after mode ordering and duplicate-trajectory filtering.
- `11_organize_traj_viz_folders.py` organizes rendered trajectory images by scenario and environmental condition.

Each utility provides complete command-line documentation through `python <script> --help`.

## Generated Files

Generated scenarios, images, point clouds, and annotations are excluded from version control by the repository `.gitignore`. Keep permanent dataset releases in external storage rather than committing generated captures to this source repository.
