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
uv run python racing_tools/run.py --in in.mp4 --telemetry <tel_dir> --track <track_dir> --out output.mp4
uv run python racing_tools/run.py --telemetry <tel_dir> --track <track_dir> # telemetry-only
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
## Current Technical Assignment Prompt

Use this prompt to resume the current workstream:

```text
You are continuing the lap-analysis work in this repository. Do not revert unrelated user changes already present in the worktree.

Goal:
Make the local analysis UI load real telemetry correctly from local files/folders and render sane lap data, especially for Alfano ZIP input.

Primary files already involved:
- racing_web/analysis.html
- racing_web/analysis.css
- racing_web/analysis.js
- racing_web/analysis_server.py
- racing_tools/session/session.py
- racing_tools/session/alfano/* (only if the root cause is there)

Constraints:
- Reuse the existing Session/Track pipeline. Do not add a parallel loader.
- Fix data corruption at the source. Do not hide it in the UI with clamps or fake fallbacks.
- Do not touch racing_tools/video/transcode.py unless strictly required; it has separate in-progress changes.
- If you change exported symbols, run reference checks first.
- Add regression tests for the real failing path; do not use mocks.

Known reproduced problems:
1. `build_session_info(...)` for
   `experiments/alfano-log-zip-format/data/ALFANO7_LAP_SN1061_170326_16H32_SG__P__A_13_6309.zip`
   returns duplicate lap choices for the same lap ids. Example: lap 1 appears multiple times with
   `85.460`, `85.470`, `85.480`; total returned entries were 40.
2. `build_analysis_payload(...)` for that same ZIP produced a best lap with an impossible total distance:
   `12709215495.83142` meters, with sectors `[0.0, 12709215495.83142]`.
3. Direct inspection showed the root telemetry table for that ZIP had a broken `Distance` scale:
   `Distance.max() == 177726681957.05258`.
4. The paired Excel CSV sample
   `experiments/alfano-log-zip-format/data/Excel_SN1061_170326_16H32_SG__P__A_13_6309.csv`
   produced a sane distance scale instead: `Distance.max() == 1294.01`.
5. `prepare_session_laps()` currently has to cope with `0 GPS crossings`; the fix must still return usable lap lists and payloads.

What to do:
1. Trace where the Alfano ZIP distance/lap data becomes invalid.
2. Fix the real source of the bad lap stats and/or bad distance series.
3. Ensure session-info returns one logical option per lap.
4. Ensure analysis payload distances and sector boundaries are monotonic, realistic, and derived from valid telemetry.
5. Verify the page can load the sample track `racing_tools/track/data/RIMSportKarting` together with the sample Alfano ZIP through the existing analysis server flow.

Acceptance criteria:
- `build_session_info()` on the sample ZIP returns unique lap ids, with a stable best lap.
- `build_analysis_payload()` on the sample ZIP returns realistic lap distance (kart-scale, not billions), valid monotonic sectors, and non-empty points.
- The analysis UI still supports loading 1-6 laps and keeps the current compare behavior: first two laps drive sector-gap and delta views; extra laps remain overlaid.
- Add/update automated tests in `tests/` for the regression.
- Run the targeted tests you changed and a direct sample-data verification against the real Alfano ZIP before finishing.

Useful reproduction snippets:
- `from racing_web.analysis_server import build_session_info, build_analysis_payload`
- track path: `racing_tools/track/data/RIMSportKarting`
- sample zip: `experiments/alfano-log-zip-format/data/ALFANO7_LAP_SN1061_170326_16H32_SG__P__A_13_6309.zip`
- sample csv: `experiments/alfano-log-zip-format/data/Excel_SN1061_170326_16H32_SG__P__A_13_6309.csv`

Observed evidence from the current session that you should preserve as the baseline:
- ZIP best lap previously resolved to `01:03.060`.
- ZIP payload previously returned `721` resampled points.
- The failure is not “no data”; it is corrupted/incorrectly interpreted data.
```
