# Telemetry Overlay Tool

This directory contains `overlay.py`, a tool to render telemetry overlays onto racing videos.
It combines video footage with telemetry data (MyChron5, Alfano, etc.) to create a composite video with gauges, track maps, and lap timers.

## Features

- **Gauges**: Speed, RPM, G-Force (Lat/Long).
- **Track Map**: Live position on the track map (requires shapefiles).
- **Lap Timer**: Automatic lap estimation using start/finish line coordinates.
- **Synchronization**: Adjust time offsets between video and telemetry.
- **Hardware Acceleration**: Support for NVENC (NVIDIA) encoding.

## Usage

The tool uses `uv` to manage dependencies. Run it from the repository root:

uv run --project converter python render/overlay.py --video path/to/video.mp4 --telemetry path/to/telemetry_folder --track-dir data/tracks/YourTrack --output output_video.mp4
```

### Example

```bash
uv run --project converter python render/overlay.py --video render/session-01.mp4 --telemetry data/telemetry/2025-11-18-mychron5-RIMSportKarting-RotaxMax-KharitonovIvan-session-01 --track-dir data/tracks/RIMSportKarting --start 60 --duration 30 --time-shift 1.5 --output render/overlay_output.mp4

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--video` | Path to input video file | `render/session-01.mp4` |
| `--telemetry` | Path to telemetry folder | (Example path) |
| `--output` | Path to output video file | `render/session-01-overlay-sample.mp4` |
| `--start` | Start time in seconds | `0.0` |
| `--duration` | Duration to process in seconds | `Full video` |
| `--time-shift` | Time offset (seconds) to sync telemetry | `0.0` |
| `--track-dir` | Directory containing track shapefiles | `data/tracks/RIMSportKarting` |
| `--hwaccel-cuda` | Enable CUDA hardware acceleration (Auto-enabled if NVIDIA GPU detected) | `Auto` |
| `--video-codec` | Video codec (e.g., `libx264`, `h264_nvenc`) | `h264_nvenc` (if CUDA) else `libx264` |
| `--keep-frames` | Keep intermediate PNG frames | `False` |

## Track Maps

To enable the track map and lap timer, provide a `--track-dir` containing:
- `centerline/centerline.shp`: The track path.
- `start-finish-line/start-finish-line.shp`: The start/finish line for lap timing.
