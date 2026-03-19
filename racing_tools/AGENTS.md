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
  - `video_info.py`: Video metadata extraction using ffprobe
  - `convert.py`: CLI for converting telemetry formats to MoTeC .ld files
  - `channel_mapping.json`: Channel name/units normalization for different devices

- **`track/`** - Track geometry and mapping
  - `models.py`: `Track` and `TrackGeometry` classes for loading shapefiles, GPS coordinates
  - Supports WGS84 → Web Mercator and UTM projections for accurate distance calculations
  - Loads track sectors/segments from GeoJSON files

- **`run.py`** - Main video processing pipeline
  - Fisheye undistortion using camera calibration
  - Video stabilization (vid.stab)
  - Telemetry synchronization (piecewise linear mapping via lap crossings)
  - ASS subtitle generation for gauges, track map, lap stats
  - Hardware-accelerated encoding (NVENC/AV1)

- **`overlay.py`** - Legacy overlay renderer (being migrated to run.py)

- **`camera/`** - Camera calibration utilities (checkerboard calibration, fisheye model, undistortion)

- **`transcode.py`**, **`trim.py`**, **`video_split.py`** - Video utilities

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

```
data/tracks/RIMSportKarting/
├── centerline/centerline.shp         # Main track geometry
├── start-finish/start-finish.shp     # Lap timing line
├── sectors.geojson                   # Optional sector definitions
└── bestline.geojson                  # Optional bestline GPS trajectory
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
  --track-dir data/tracks/RIMSportKarting \
  --out output.mp4

# Telemetry-only mode (no video)
uv run python racing_tools/run.py \
  --telemetry path/to/telemetry_folder \
  --track-dir data/tracks/RIMSportKarting
```

### Video utilities

```bash
# Trim video (interactive mode)
uv run python racing_tools/trim.py input.mp4

# Transcode to AV1 with NVENC
uv run python racing_tools/transcode.py input.mp4 -o output.mp4

# Split video by laps
uv run python racing_tools/video_split.py input.mp4 --crossings 10.5 25.3 40.1
```

### Camera calibration

```bash
# Find camera intrinsics from checkerboard images
uv run python racing_tools/camera/find_intrinsics.py path/to/checkboard_images/

# Undistort images using calibration
uv run python racing_tools/camera/undistort.py --intrinsics camera.csv input.jpg
```

## Development Notes

- Coordinate reference systems: WGS84 for GPS, Web Mercator for mapping, UTM for accurate distance
- Channel mappings in `session/channel_mapping.json` apply scale/offset transforms automatically
- Video-telemetry sync uses lap crossing times matched by interval pattern (RMSE minimization)
- ASS subtitles are preferred over per-frame PNG rendering for performance
- Hardware acceleration auto-detected: NVENC if CUDA available, else CPU SVT-AV1