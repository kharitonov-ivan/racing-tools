# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management
- Use `uv` for all Python dependency management
- Install dependencies: `uv sync`
- Run scripts: `uv run python racing_tools/run.py --help`

## Git Workflow
For fixes and features, use a branch-based workflow:
```bash
git checkout -b fix/short-description
git add <files>
git commit -m "fix: desc"
git fetch origin main
git rebase origin/main

# 4. Squash commits into logical units before merge
git rebase -i origin/main
# Mark commits to squash using 'squash' or 's' keyword
# Always review code with separate agent on this point
# Always run tests and run example to verify that everything works

# 5. Merge back to main - Always ask user to review before merging
git checkout main
git merge fix/short-description
git branch -d fix/short-description
```

### Rebase Policy
- Always rebase onto `main` before merging (linear history)
- Use `git rebase -i` to squash WIP commits
- Never force-push to `main` or shared branches

### Commit Squashing
- Combine related commits. Final commit: complete, shippable change.
- Format: `type: description` (`fix:`, `feat:`, `refactor:`)

## Development Notes
- CRS: WGS84 (GPS), Web Mercator (map), UTM (dist).
- `session/channel_mapping.json` auto-applies scale/offset.
- Video-telemetry sync: match lap crossings via interval pattern (RMSE min).
- ASS subtitles preferred over PNG for performance.
- HW acceleration: NVENC (CUDA) else CPU SVT-AV1.

## Code Style
- Modular functions (max 50 lines)
- Prefer composition over inheritance
- Type hints for all signatures
- Files under 300 lines
- PEP 8, `ruff format`
- No imports in functions
- No unnecessary `try/except` blocks

## Project Overview
Telemetry tools processing, analyzing, overlaying (AIM, Alfano, MoTeC) with video.

### Core Architecture
- **`session/`** - Telemetry & normalization
  - `session.py`: `Session` loads AIM/Alfano/CSV.
  - `predictive.py`: `PredictiveLapModel` (dist-based predict).
  - `convert.py`: CLI MoTeC exporter.
  - `channel_mapping.json`: Normalization.
- **`track/`** - Geometry & mapping
  - `track.py`: `Track` (GeoJSON loading).
  - `utils.py`, `segmentation.py`: Utils.
  - WGS84 → Mercator/UTM projections.
- **`run.py`** - Video processing pipeline
  - Fisheye undistort, vid.stab, syn matching crossings, ASS overlays, NVENC/AV1.
- **`camera/`** - Checkerboard calibration, fisheye model.
- **`video/`** - Processing utilities
  - `undistort.py`, `trim.py`, `transcode.py`, `split.py`, `stab.py`, `overlay.py`, `ass.py`, `video_info.py`.

### Telemetry Processing Pipeline
1. `Session.load(path)` (AIM/Alfano)
2. Normalize via `channel_mapping.json`
3. Compute lap crossings (GPS track or Lap col)
4. Build `PiecewiseSync` (video crossings -> telemetry crossings)
5. `VideoSession.resample_to_video()`

### Session Data Model
- `Session.table`: pandas DataFrame
- Required cols: `Time`, `Distance`, `LapNumber`
- Speed cols: `GPS Speed`, `Wheel Speed` (km/h)
- GPS: `GPS Latitude`, `GPS Longitude` (WGS84 deg)

### Track Directory Structure
```text
racing_tools/track/data/<Name>/
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
uv run python racing_tools/session/convert.py aim <session_folder>
uv run python racing_tools/session/convert.py batch <logs_dir>
uv run python racing_tools/session/convert.py alfano-excel <alfano_excel>
```

### Generate video with telemetry overlay
```bash
uv run python racing_tools/run.py --in in.mp4 --telemetry <tel_dir> --track-dir <track_dir> --out output.mp4
uv run python racing_tools/run.py --telemetry <tel_dir> --track-dir <track_dir> # telemetry-only
```

### Video utilities
```bash
uv run python racing_tools/video/trim.py in.mp4
uv run python racing_tools/video/transcode.py in.mp4 -o out.mp4
uv run python racing_tools/video/split.py in.mp4 --crossings 10.5 25.3
```

### Camera calibration
```bash
uv run python racing_tools/camera/find_intrinsics.py <imgs_dir>
uv run python racing_tools/camera/undistort.py --intrinsics cam.csv in.jpg
```

## Testing
`data/test/` clips for testing:
```bash
ffmpeg -y -i data/<vid> -ss 387 -t 10 -c copy data/test/10s.mp4
uv run python racing_tools/video/undistort.py data/test/10s.mp4 <cam.csv> --output out.mp4
uv run python racing_tools/video/stab.py data/test/10s.mp4 --overwrite
```
