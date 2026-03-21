# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management
- Use `uv` for all Python dependency management
- Install dependencies: `uv sync`
- Run scripts: `uv run python racing_tools/run.py --help`

## Code Style
- Write modular functions (max 50 lines)
- Prefer composition over inheritance
- Use type hints for all function signatures
- Keep files under 300 lines when possible
- Follow PEP 8 conventions
- Do not insert imports in functions
- Do not use try except blocks when whis is not necessary
- Use ruff format

## Project Overview

Racing telemetry tools for processing, analyzing, and overlaying telemetry data from racing sessions. Supports multiple telemetry formats (AIM MyChron5, Alfano, MoTeC) and combines them with video footage.

### Core Architecture

- **`session/`** - Telemetry data processing and normalization
  - `session.py`: Core `Session` class that loads telemetry from various formats (AIM, Alfano, CSV)
  - `predictive.py`: `PredictiveLapModel` for lap time prediction based on distance
  - `convert.py`: CLI for converting telemetry formats to MoTeC .ld files
  - `channel_mapping.json`: Channel name/units normalization for different devices

- **`track/`** - Track geometry and mapping
  - `track.py`: `Track` class for loading track boundaries, centerline, bestline from GeoJSON files
  - `utils.py`: Track utility functions (normalize_angle, compute_centerline, etc.)
  - `segmentation.py`: Track segmentation into straights/turns
  - Supports WGS84 → Web Mercator and UTM projections for accurate distance calculations

- **`run.py`** - Main video processing pipeline
  - Fisheye undistortion using camera calibration
  - Video stabilization (vid.stab)
  - Telemetry synchronization (piecewise linear mapping via lap crossings)
  - ASS subtitle generation for gauges, track map, lap stats
  - Hardware-accelerated encoding (NVENC/AV1)

- **`camera/`** - Camera calibration utilities (checkerboard calibration, fisheye model)

- **`video/`** - Video processing utilities
  - `undistort.py`, `trim.py`, `transcode.py`, `split.py`: Video manipulation tools
  - `stab.py`: Video stabilization using vidstab filters
  - `overlay.py`: Overlay rendering functions (track map, gauges)
  - `ass.py`: ASS subtitle generation for video overlays
  - `video_info.py`: Video metadata probing

### Telemetry Processing Pipeline

1. Load telemetry: `Session.load(path)` auto-detects format (AIM/Alfano)
2. Normalize channels via `channel_mapping.json` (applies scale/offset transforms)
3. Compute lap crossings from GPS track or existing Lap column
4. Build `PiecewiseSync` mapping from video crossings to telemetry crossings
5. Resample telemetry to video frame timestamps via `VideoSession.resample_to_video()`

### Session Data Model

- `Session.table`: pandas DataFrame with normalized telemetry
- Required columns: `Time`, `Distance`, `LapNumber`
- Speed columns: `GPS Speed` (km/h), `Wheel Speed` (km/h)
- GPS: `GPS Latitude`, `GPS Longitude` (WGS84 deg)

### Track Directory Structure

Track data lives inside the package at `racing_tools/track/data/`.

```
racing_tools/track/data/RIMSportKarting/
├── geometry/
│   ├── track-inner.geojson           # Inner track boundary (required)
│   ├── track-outer.geojson           # Outer track boundary (required)
│   ├── start-finish.geojson          # Lap timing line (optional)
│   ├── centerline.geojson            # Centerline (optional, computed if missing)
│   ├── bestline.geojson              # Optimal racing line (optional)
│   ├── kerbs.geojson                 # Kerb geometry (optional)
│   └── strips.geojson                # Strip geometry (optional)
├── track_config.json                 # Track metadata (UTM zone, name)
└── export/                           # Format exports (MoTeC, Alfano)
```

## Common Commands

### Convert telemetry to MoTeC format

```bash
# Single session
uv run python racing_tools/session/convert.py aim path/to/session_folder

# Batch process entire directory
uv run python racing_tools/session/convert.py batch path/to/logs

# Alfano Excel export
uv run python racing_tools/session/convert.py alfano-excel path/to/alfano_excel
```

### Generate video with telemetry overlay

```bash
# Full pipeline with telemetry sync
uv run python racing_tools/run.py \
  --in input_video.mp4 \
  --telemetry path/to/telemetry_folder \
  --track-dir racing_tools/track/data/RIMSportKarting \
  --out output.mp4

# Telemetry-only mode (no video)
uv run python racing_tools/run.py \
  --telemetry path/to/telemetry_folder \
  --track-dir racing_tools/track/data/RIMSportKarting
```

### Video utilities

```bash
# Trim video (interactive mode)
uv run python racing_tools/video/trim.py input.mp4

# Transcode to AV1 with NVENC
uv run python racing_tools/video/transcode.py input.mp4 -o output.mp4

# Split video by laps
uv run python racing_tools/video/split.py input.mp4 --crossings 10.5 25.3 40.1
```

### Camera calibration

```bash
# Find camera intrinsics from checkerboard images
uv run python racing_tools/camera/find_intrinsics.py path/to/checkboard_images/

# Undistort images using calibration
uv run python racing_tools/camera/undistort.py --intrinsics camera.csv input.jpg
```

## Testing

Test data is located in `data/test/`. Create 10-second clips from the middle of videos for testing:

```bash
# Create test clip from middle of video (e.g., at 387 seconds of 784s video)
ffmpeg -y -i data/17-03-2026/17-23/2026-03-17_17-23-13.mp4 -ss 387 -t 10 -c copy data/test/test_10sec.mp4

# Test undistortion
uv run python racing_tools/video/undistort.py \
  data/test/test_10sec.mp4 \
  racing_tools/camera/intrinsics_fisheye.csv \
  --output data/test/test_undistorted.mp4

# Test stabilization
uv run python racing_tools/video/stab.py data/test/test_10sec.mp4 --overwrite
```

## Git Workflow

For fixes and features, use a branch-based workflow:

```bash
# 1. Create a feature/fix branch from main
git checkout -b fix/short-description

# 2. Make changes, commit
git add <files>
git commit -m "fix: description of the fix"

# 3. Sync with main via rebase (keep history linear)
git fetch origin main
git rebase origin/main

# 4. Squash commits into logical units before merge
git rebase -i origin/main
# Mark commits to squash using 'squash' or 's' keyword

# 5. Merge back to main
git checkout main
git merge fix/short-description

# 6. Delete the branch
git branch -d fix/short-description
```

### Rebase Policy
- Always rebase onto `main` before merging to keep history linear
- Use `git rebase -i` to squash WIP/debugging commits into meaningful units
- Never force-push to `main` or shared branches

### Commit Squashing
- Combine related commits (e.g., "fix bug", "address review" → "fix: resolve issue X")
- Final merge commit should represent a complete, shippable change
- Use meaningful commit messages: `type: description` format (fix:, feat:, refactor:, etc.)

## Development Notes

- Coordinate reference systems: WGS84 for GPS, Web Mercator for mapping, UTM for accurate distance
- Channel mappings in `session/channel_mapping.json` apply scale/offset transforms automatically
- Video-telemetry sync uses lap crossing times matched by interval pattern (RMSE minimization)
- ASS subtitles are preferred over per-frame PNG rendering for performance
- Hardware acceleration auto-detected: NVENC if CUDA available, else CPU SVT-AV1