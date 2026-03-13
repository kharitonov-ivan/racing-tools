#!/usr/bin/env python3
"""
TODO: 
1) сли есть телеметрия, сохранять trim_info после загрузки и синхронизации,
      чтобы автоматически определять start/end на основе telemetry duration

2) [av1_nvenc @ 0x601fa6a04cc0] The selected preset is deprecated. Use p1 to p7 + -tune or fast/medium/slow. 
"""
import argparse
import atexit
import math
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import cv2
import ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from racing_tools.camera.model import CameraModel
from racing_tools.overlay import (
    PredictiveLapModel,
    build_track_overlay,
    draw_track_static,
    format_duration,
    get_gradient_color,
)
from racing_tools.session.session import DISTANCE_AIM, SPEED_AIM, PiecewiseSync, Session, VideoSession, ensure_distance
from racing_tools.session.video_info import VideoInfo, probe_video
from racing_tools.sync_ui import run_manual_lap_marking
from racing_tools.track.models import WGS84_TO_WEBMERC, Track
from racing_tools.trim import (
    VideoSidecar,  # Manual lap marking mode - use VideoSidecar for persistence
    get_trim_info,
)
from racing_tools.utils import check_cuda_availability


class Pipeline(NamedTuple):
    """Video and audio stream pair for FFmpeg pipeline."""

    video: ffmpeg.Stream
    audio: ffmpeg.Stream


def create_session_from_crossings(video_info: VideoInfo, crossing_times: list[float]) -> Session:
    """
    Create a Session object from video info and manual crossing times (no telemetry).

    Creates one row per video frame with:
    - Time: frame_number / fps
    - Duration: cumulative time
    - LapNumber: assigned from crossing_times (before first crossing = Lap 0)

    Args:
        video_info: VideoInfo with fps, nb_frames, etc.
        crossing_times: Sorted list of lap crossing times in seconds.

    Returns:
        Session object with minimal frame-based data.
    """
    from bisect import bisect_right

    import pandas as pd

    fps = video_info.fps
    nb_frames = video_info.nb_frames

    # Create time array for each frame
    times = np.arange(nb_frames) / fps

    # Assign lap numbers based on crossing times
    # Everything before first crossing is Lap 0
    # crossing_times[0] -> start of Lap 1, etc.
    sorted_crossings = sorted(crossing_times) if crossing_times else []

    def get_lap_number(t: float) -> int:
        if not sorted_crossings:
            return 0
        return bisect_right(sorted_crossings, t)

    lap_numbers = np.array([get_lap_number(t) for t in times])

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Time": times,
            "Duration": times,  # Cumulative time (same as Time for video-based)
            "LapNumber": lap_numbers,
        }
    )

    return Session(table=df)


def write_pgm_u16(path: Path, img_u16: np.ndarray) -> None:
    # Write big-endian PGM (P5) with maxval 65535
    if img_u16.dtype != np.uint16 or img_u16.ndim != 2:
        raise ValueError("img_u16 must be uint16 2D array")
    h, w = img_u16.shape
    header = f"P5\n{w} {h}\n65535\n".encode("ascii")
    path.write_bytes(header + img_u16.byteswap().tobytes())


def _compute_undistort_maps(w: int, h: int, model: CameraModel, balance: float = 1.0, fov_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute fisheye undistortion remap coordinates."""
    K = model.matrix.astype(np.float64)
    D = model.dist_coeffs.astype(np.float64).reshape(4, 1)
    dim = (w, h)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance, fov_scale=fov_scale)
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_32FC1)


def _create_remap_arrays(mapx: np.ndarray, mapy: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert float maps to uint16 arrays with validity mask."""
    mapx_i = np.rint(mapx).astype(np.int32)
    mapy_i = np.rint(mapy).astype(np.int32)
    valid = (mapx_i >= 0) & (mapx_i < w) & (mapy_i >= 0) & (mapy_i < h)

    x_u16, y_u16, mask_u16 = (np.zeros((h, w), dtype=np.uint16) for _ in range(3))

    x_u16[valid] = mapx_i[valid].astype(np.uint16)
    y_u16[valid] = mapy_i[valid].astype(np.uint16)
    mask_u16[valid] = np.uint16(65535)

    return x_u16, y_u16, mask_u16


def make_fisheye_remap_maps(
    w: int, h: int, model: CameraModel, out_xmap: Path, out_ymap: Path, out_mask: Path, balance: float = 1.0, fov_scale: float = 1.0
) -> None:
    """Generate fisheye undistortion remap PGM files for FFmpeg."""
    mapx, mapy = _compute_undistort_maps(w, h, model, balance, fov_scale)
    x_u16, y_u16, mask_u16 = _create_remap_arrays(mapx, mapy, w, h)

    for path, data in ((out_xmap, x_u16), (out_ymap, y_u16), (out_mask, mask_u16)):
        write_pgm_u16(path, data)


def build_opener(input_path: str | Path, hwaccel: str | None = None) -> Pipeline:
    v = ffmpeg.input(str(input_path), **({} if hwaccel is None else {"hwaccel": hwaccel}))
    return Pipeline(v, v.audio)


def build_trimer(pipe: Pipeline, ss: float, to: float) -> Pipeline:
    return Pipeline(
        pipe.video.video.filter("trim", start=ss, duration=to - ss).filter("setpts", "PTS-STARTPTS"),
        pipe.audio.filter("atrim", start=ss, duration=to - ss).filter("asetpts", "PTS-STARTPTS"),
    )


def _load_remap_stream(path: Path, fps: float) -> ffmpeg.Stream:
    """Load a PGM remap file as a looped FFmpeg stream."""
    return ffmpeg.input(str(path), loop=1, framerate=fps).video.filter("setpts", "PTS-STARTPTS")


def build_undistorter(pipe: Pipeline, camera_model: CameraModel, balance: float, fov_scale: float, video_info: VideoInfo) -> Pipeline:
    tmp = Path(tempfile.gettempdir())
    xmap, ymap, mask = tmp / "xmap.pgm", tmp / "ymap.pgm", tmp / "mask.pgm"

    make_fisheye_remap_maps(video_info.width, video_info.height, camera_model, xmap, ymap, mask, balance, fov_scale)

    return Pipeline(
        ffmpeg.filter([pipe.video, _load_remap_stream(xmap, video_info.fps), _load_remap_stream(ymap, video_info.fps)], "remap"), pipe.audio
    )


def build_transform_estimator(pipe: Pipeline, transform_path: Path, shakiness: int, accuracy: int, stepsize: int) -> Pipeline:
    v = pipe.video.filter("vidstabdetect", shakiness=shakiness, accuracy=accuracy, stepsize=stepsize, result=str(transform_path))
    return Pipeline(v, pipe.audio)


def build_stabilizer(
    pipe: Pipeline,
    transform_path: Path,
    smoothing: int = 10,
    zoom: int = 0,
    optzoom: int = 0,
    crop: str = "keep",
    interpol: str = "bilinear",
    unsharp: bool = True,
) -> Pipeline:
    v = pipe.video.filter(
        "vidstabtransform", input=str(transform_path), zoom=zoom, smoothing=smoothing, optzoom=optzoom, crop=crop, interpol=interpol
    )
    if unsharp:
        v = v.filter("unsharp", lx=5, ly=5, la=0.8, cx=3, cy=3, ca=0.4)
    return Pipeline(v, pipe.audio)


def build_ov(pipe: Pipeline, overlay_stream: ffmpeg.Stream | None = None) -> Pipeline:
    v = pipe.video
    if overlay_stream:
        v = v.overlay(overlay_stream, x=0, y=0, eof_action="pass")

    v = v.filter("drawtext", text="%{n}", x=50, y=50, fontsize=48, fontcolor="white", box=1, boxcolor="black@0.5", boxborderw=5)
    return Pipeline(v, pipe.audio)


def build_lap_stats_ov(pipe: Pipeline, session: "VideoSession") -> Pipeline:
    """
    Renders the lap list using ASS subtitles for better alignment and performance.
    Dynamically displays only columns that have data.
    """
    v = pipe.video

    lap_stats = session.get_lap_stats()
    crossings = session.crossings
    best_lap = session.best_lap
    total_duration = session.info.duration
    width = session.info.width
    height = session.info.height

    if not lap_stats:
        return pipe

    # Build columns dynamically from available data
    sample = lap_stats[0]
    columns = []  # [(header, key, width)]
    
    # Always show lap id and time first
    columns.append(("Lap", "id", 60))
    columns.append(("Video", "time", 120))

    # Add GPS time column if available
    if sample.get("gps_time") is not None:
        columns.append(("GPS", "gps_time", 120))

    # Add any other numeric columns that have data
    for key, value in sample.items():
        if key in ("id", "time", "gps_time"):
            continue
        if value is not None:
            columns.append((key.replace("_", " ").title(), key, 100))

    # --- Configuration ---
    margin_right = 50
    col_gap = 20
    total_width = sum(c[2] for c in columns) + col_gap * (len(columns) - 1)
    base_x = width - total_width - margin_right

    start_y = 50
    row_h = 40

    # Calculate column x positions
    col_positions = []
    x = 0
    for _, _, w in columns:
        col_positions.append(x)
        x += w + col_gap

    # Best lap ID
    best_lap_id = best_lap["id"] if best_lap else -1
    sorted_stats = sorted(lap_stats, key=lambda x: x["id"])

    # Create temp ASS file
    fd, temp_ass_path = tempfile.mkstemp(suffix=".ass")
    os.close(fd)

    def cleanup():
        if os.path.exists(temp_ass_path):
            os.unlink(temp_ass_path)

    atexit.register(cleanup)

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Header,Arial,20,&H00AAAAAA,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1
Style: Row,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1
Style: RowGold,Arial,24,&H0000D7FF,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(temp_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)

        # Write headers
        h_y = start_y
        for i, (header, _, _) in enumerate(columns):
            x_pos = base_x + col_positions[i]
            f.write(f"Dialogue: 0,0:00:00.00,99:59:59.99,Header,,0,0,0,,{{\\pos({x_pos},{h_y})}}{header}\n")

        data_start_y = start_y + 40

        # Write data rows
        for row_idx, s in enumerate(sorted_stats):
            y_pos = data_start_y + row_idx * row_h
            style = "RowGold" if s["id"] == best_lap_id else "Row"
            common = f"Dialogue: 0,0:00:00.00,99:59:59.99,{style},,0,0,0,,"

            for col_idx, (_, key, _) in enumerate(columns):
                x_pos = base_x + col_positions[col_idx]
                value = s.get(key)
                # Format based on key type
                if key == "id":
                    text = f"{value}*" if value == best_lap_id else str(value)
                elif key in ("time", "gps_time"):
                    text = format_duration(value, decimals=3) if value else "-"
                elif value is None:
                    text = "-"
                elif isinstance(value, float):
                    text = f"{value:.1f}"
                else:
                    text = str(int(value))
                f.write(f"{common}{{\\pos({x_pos},{y_pos})}}{text}\n")

        # Dynamic Pointer >
        times = [0.0] + list(crossings) + [total_duration]

        def fmt_ass(t):
            h = int(t / 3600)
            m = int((t % 3600) / 60)
            sec = int(t % 60)
            cs = int((t * 100) % 100)
            return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"

        for i in range(len(times) - 1):
            range_start = times[i]
            range_end = times[i + 1]
            if range_end <= range_start:
                continue

            current_lap_num = i
            idx = next((idx for idx, s in enumerate(sorted_stats) if s["id"] == current_lap_num), -1)

            if idx != -1:
                y_pos = data_start_y + idx * row_h
                s_str = fmt_ass(range_start)
                e_str = fmt_ass(range_end)
                ptr_x = base_x - 30
                f.write(f"Dialogue: 1,{s_str},{e_str},Row,,0,0,0,,{{\\pos({ptr_x},{y_pos})}}>\n")

    v = v.filter("subtitles", filename=temp_ass_path)
    return Pipeline(v, pipe.audio)


def build_gauge_overlay(pipe: Pipeline, session: "VideoSession") -> Pipeline:
    """
    Builder for gauge overlay.
    Generates ASS subtitle file and attaches 'subtitles' filter.
    """
    v = pipe.video
    
    # Extract data from session
    width = session.info.width
    height = session.info.height
    fps = session.info.fps
    lap_stats = session.get_lap_stats()
    best_lap = session.best_lap
    session_table = session.table
    
    # Find speed column (may have different names)
    speed_col = None
    for name in ["Speed", "GPS Speed", "Wheel Speed"]:
        if name in session_table.columns:
            speed_col = name
            break
    
    if speed_col is None:
        print(f"[gauge] No speed column found in {list(session_table.columns)}, skipping gauge overlay")
        return pipe
    
    # Find RPM column
    rpm_col = None
    for name in ["RPM", "Engine RPM"]:
        if name in session_table.columns:
            rpm_col = name
            break

    # 1. Generate ASS File
    # Create temp file
    fd, temp_ass_path = tempfile.mkstemp(suffix=".ass")
    os.close(fd)

    # Schedule cleanup on exit
    def cleanup():
        if os.path.exists(temp_ass_path):
            os.unlink(temp_ass_path)

    atexit.register(cleanup)

    # ASS Header
    # Font Size increased to 48 per user request
    # BorderStyle=3 (Opaque Box) for better contrast
    # Define styles for Gauge and Delta
    # Delta style: Top Center, bigger font? Or similar.
    # We use \an8 (Top Center) override in the event or define a style.
    # Let's define a Delta style.
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Gauge,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,3,0,0,2,10,10,50,1
Style: Delta,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,8,10,10,20,1
Style: LapTimer,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,1,10,10,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Prepare Predictive Model
    predictive_model = None
    lap_start_distances = {}

    print(f"[gauge] Building predictive model: lap_stats={bool(lap_stats)}, best_lap={best_lap}")
    print(f"[gauge] Columns: {list(session_table.columns)}")
    
    if lap_stats and session_table is not None and best_lap:
        best_lap_id = best_lap["id"]

        # Extract Best Lap Data
        if "Distance" in session_table.columns and "LapNumber" in session_table.columns:
            # Group distances by lap
            lap_start_distances = session_table.groupby("LapNumber")["Distance"].min().to_dict()
            print(f"[gauge] Lap start distances: {lap_start_distances}")

            # Build model
            mask = session_table["LapNumber"] == best_lap_id
            bl_df = session_table[mask].copy()
            print(f"[gauge] Best lap {best_lap_id} data: {len(bl_df)} rows")
            if not bl_df.empty:
                start_dist = bl_df["Distance"].min()
                # Use elapsed lap time if available, or calc from LapTime column
                dists = (bl_df["Distance"] - start_dist).values
                
                if "LapTime" in bl_df.columns:
                    times = bl_df["LapTime"].values
                else:
                    print("[gauge] No LapTime column, cannot build predictive model")
                    times = None
                
                if times is not None:
                    # Check valid
                    valid = ~np.isnan(dists) & ~np.isnan(times)
                    dists = dists[valid]
                    times = times[valid]

                    print(f"[gauge] Valid data points: {len(dists)}")
                    if len(dists) > 10:
                        predictive_model = PredictiveLapModel(list(zip(dists, times)))
                        print(f"[gauge] Built Predictive Model from Lap {best_lap_id} ({len(dists)} points)")
        else:
            print(f"[gauge] Missing Distance or LapNumber columns")

    with open(temp_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)

        speeds = session_table[speed_col].fillna(0).values
        rpms = session_table[rpm_col].fillna(0).values if rpm_col else np.zeros(len(session_table))

        # New columns for Delta
        has_delta = False
        if predictive_model and "Distance" in session_table.columns and "LapNumber" in session_table.columns and "LapTime" in session_table.columns:
            dists = session_table["Distance"].values
            laps = session_table["LapNumber"].values
            l_times = session_table["LapTime"].values
            has_delta = True
            
            # Debug: Print model range and sample delta calculations
            print(f"[gauge] Predictive model distances: min={predictive_model.dists.min():.2f}, max={predictive_model.dists.max():.2f}")
            print(f"[gauge] Predictive model times: min={predictive_model.times.min():.2f}, max={predictive_model.times.max():.2f}")
            
            # Sample calculation for first few valid points
            debug_count = 0
            for i in range(min(100, len(dists))):
                curr_lap = laps[i]
                if curr_lap in lap_start_distances:
                    start_d = lap_start_distances[curr_lap]
                    lap_dist = dists[i] - start_d
                    if lap_dist >= 0 and debug_count < 3:
                        pred_time = predictive_model.get_time(lap_dist)
                        curr_time = l_times[i]
                        delta = curr_time - pred_time
                        print(f"[gauge] Sample delta (frame {i}): lap={curr_lap}, lap_dist={lap_dist:.2f}, curr_time={curr_time:.3f}, pred_time={pred_time:.3f}, delta={delta:+.3f}")
                        debug_count += 1

        max_rpm = 14000
        if rpm_col:
            m = session_table[rpm_col].max()
            if not np.isnan(m):
                max_rpm = int(np.ceil(m / 1000) * 1000)

        total_frames = len(session_table)

        # Precompute crossings for lap elapsed time
        display_crossings: list[float] = getattr(session, "crossings", []) or []

        def fmt_time(t):
            h = int(t / 3600)
            m = int((t % 3600) / 60)
            s = int(t % 60)
            cs = int((t * 100) % 100)
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        def fmt_lap_time(t: float) -> str:
            """Format elapsed lap time as M:SS.mmm."""
            m = int(t / 60)
            s = t - m * 60
            return f"{m}:{s:06.3f}"

        # Generate per-frame events
        for i in range(total_frames):
            t_start = i / fps
            t_end = (i + 1) / fps
            s_str = fmt_time(t_start)
            e_str = fmt_time(t_end)

            # Simple text generation
            speed = speeds[i]
            rpm = rpms[i]

            # RPM color: red < 7500, yellow 7500-8500, green > 8500
            if rpm < 7500:
                rpm_color = "&H000000FF"  # Red (BGR)
            elif rpm < 8500:
                rpm_color = "&H0000FFFF"  # Yellow (BGR)
            else:
                rpm_color = "&H0000FF00"  # Green (BGR)

            bar_len = 20
            filled = int((rpm / max_rpm) * bar_len)
            filled = max(0, min(bar_len, filled))
            bar_text = "[" + "|" * filled + " " * (bar_len - filled) + "]"

            # Apply color to RPM section
            text = f"{{\\c&H00FFFFFF&}}{int(speed):3d} km/h   {{\\c{rpm_color}&}}{bar_text}   {int(rpm):5d} RPM"

            f.write(f"Dialogue: 0,{s_str},{e_str},Gauge,,0,0,0,,{text}\n")

            # Lap timer + frame number
            video_t = t_start
            lap_elapsed = video_t
            if display_crossings:
                last_crossing = 0.0
                for cx in display_crossings:
                    if cx <= video_t:
                        last_crossing = cx
                    else:
                        break
                lap_elapsed = video_t - last_crossing
            timer_text = f"Lap {fmt_lap_time(lap_elapsed)}  |  Frame {i}"
            f.write(f"Dialogue: 0,{s_str},{e_str},LapTimer,,0,0,0,,{timer_text}\n")

            # Delta Overlay
            if has_delta:
                try:
                    curr_dist = dists[i]
                    curr_lap = laps[i]
                    curr_time = l_times[i]

                    if curr_lap in lap_start_distances:
                        start_d = lap_start_distances[curr_lap]
                        lap_dist = curr_dist - start_d

                        if lap_dist >= 0:
                            pred_time = predictive_model.get_time(lap_dist)
                            delta = curr_time - pred_time

                            # Skip if delta is huge (out lap vs flying lap mismatch)
                            if abs(delta) < 20.0:
                                d_str = f"{delta:+.2f}"
                                color_code = "&H0000FF00" if delta < 0 else "&H000000FF"

                                # Delta Bar Calculation
                                scale_sec = 2.0
                                half_width = 10
                                # Clamp delta
                                val = max(-scale_sec, min(scale_sec, delta))
                                # Calculate ratio (-1.0 to 1.0)
                                ratio = val / scale_sec

                                # Construct bar
                                # Center is |
                                # Left side (negative/green)
                                # Right side (positive/red)

                                fill_char = "|"
                                empty_char = " "

                                num_fill = int(abs(ratio) * half_width)

                                if ratio < 0:
                                    # Fill left: [   |||||   ]
                                    left_part = empty_char * (half_width - num_fill) + fill_char * num_fill
                                    right_part = empty_char * half_width
                                else:
                                    # Fill right: [   |||||   ]
                                    left_part = empty_char * half_width
                                    right_part = fill_char * num_fill + empty_char * (half_width - num_fill)

                                delta_bar = f"[{left_part}|{right_part}]"

                                # Stack text: Value BELOW bar
                                # Use \N for hard line break in ASS
                                delta_text = "{\\c" + color_code + "}" + f"{delta_bar}\\N{d_str}"
                                f.write(f"Dialogue: 1,{s_str},{e_str},Delta,,0,0,0,,{delta_text}\n")
                except:
                    pass

    # 2. Attach Filter
    v = v.filter("subtitles", filename=temp_ass_path)

    return Pipeline(v, pipe.audio)


def calculate_segment_stats(session_table, lap_id, raw_segments):
    """
    Calculate min/max speed for each segment for a specific lap.
    """
    stats = {}

    speed_col = next((c for c in ("GPS Speed", "Speed", "Wheel Speed") if c in session_table.columns), None)
    if "Distance" not in session_table.columns or speed_col is None:
        print("[stats] Missing Distance or Speed columns.")
        return stats

    lap_data = session_table[session_table["LapNumber"] == lap_id]
    if lap_data.empty:
        print(f"[stats] No data for lap {lap_id}")
        return stats

    current_dist = 0.0
    lap_start_dist = lap_data["Distance"].min()
    print(f"[stats] Lap {lap_id} Start Dist: {lap_start_dist:.2f}")

    for idx, seg in enumerate(raw_segments):
        pts = np.array(seg["points"])
        if len(pts) < 2:
            seg_len = 0
            stats[idx] = (0, 0)
        else:
            # Check if points are WGS84 (degrees) and project if needed
            # Simple heuristic: x between -180 and 180
            if np.max(np.abs(pts[:, 0])) <= 180 and np.max(np.abs(pts[:, 1])) <= 90:
                xs, ys = WGS84_TO_WEBMERC.transform(pts[:, 0], pts[:, 1])
                projected_pts = np.column_stack((xs, ys))
            else:
                projected_pts = pts

            # Calculate length of this segment in METERS
            diffs = projected_pts[1:] - projected_pts[:-1]
            dists = np.linalg.norm(diffs, axis=1)
            seg_len = np.sum(dists)

        start_d = current_dist
        end_d = current_dist + seg_len
        current_dist += seg_len

        # Map to lap data (relative to lap start)
        mask = (lap_data["Distance"] >= (lap_start_dist + start_d)) & (lap_data["Distance"] < (lap_start_dist + end_d))

        seg_samples = lap_data[mask]

        if not seg_samples.empty:
            spd = seg_samples[speed_col]
            s_min, s_max = spd.min(), spd.max()
            stats[idx] = (s_min, s_max)
        else:
            pass

    print(f"[stats] Calculated stats for {len(stats)}/{len(raw_segments)} segments.")
    return stats


def draw_full_track_stats(draw, drawing_area, track_overlay_data):
    """
    Custom drawer to include stats for BOTH straights and turns.
    """
    inner_x, inner_y, width, height = drawing_area
    segments = track_overlay_data.get("segments")

    if segments:
        stats_by_lap = track_overlay_data.get("segment_stats", {})
        seg_stats_map = stats_by_lap

        for seg_idx, seg in enumerate(segments):
            points = seg["points"]
            scaled = [
                (
                    inner_x + max(0.0, min(1.0, pt[0])) * width,
                    inner_y + max(0.0, min(1.0, pt[1])) * height,
                )
                for pt in points
            ]

            if not scaled:
                continue

            stats = seg_stats_map.get(seg_idx)
            if stats:
                min_spd, max_spd = stats

                mid_idx = len(scaled) // 2
                mid_pt = scaled[mid_idx]

                txt = f"{int(min_spd)}/{int(max_spd)}"

                # Font handling
                try:
                    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
                except:
                    fnt = ImageFont.load_default()

                # Position text
                x, y = mid_pt[0], mid_pt[1]

                # Draw text with outline
                for off in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text((x + off[0], y + off[1]), txt, font=fnt, fill="black", anchor="mm")

                draw.text((x, y), txt, font=fnt, fill="#ffffff", anchor="mm")


def calculate_sector_stats_for_lap(session_table, lap_id, sectors):
    """
    Calculate min/max speed for each sector for a specific lap.
    
    Computes segment distances from points if start_dist/end_dist not present.
    """
    stats = {}
    
    speed_col = next((c for c in ("GPS Speed", "Speed", "Wheel Speed") if c in session_table.columns), None)
    if "Distance" not in session_table.columns or speed_col is None:
        print(f"[sector-stats] Missing Distance or Speed columns")
        return stats

    lap_data = session_table[session_table["LapNumber"] == lap_id]
    if lap_data.empty:
        print(f"[sector-stats] No data for lap {lap_id}")
        return stats

    lap_start_dist = lap_data["Distance"].min()
    
    # Calculate cumulative distance for each segment from points
    cumulative_dist = 0.0
    
    for idx, sector in enumerate(sectors):
        # Get segment distance range
        if "start_dist" in sector and "end_dist" in sector:
            start_d = sector["start_dist"]
            end_d = sector["end_dist"]
        else:
            # Calculate from points
            pts = sector.get("points", [])
            if len(pts) < 2:
                continue
            
            pts_arr = np.array(pts)
            
            # Project to meters if WGS84
            if np.max(np.abs(pts_arr[:, 0])) <= 180 and np.max(np.abs(pts_arr[:, 1])) <= 90:
                xs, ys = WGS84_TO_WEBMERC.transform(pts_arr[:, 0], pts_arr[:, 1])
                pts_m = np.column_stack((xs, ys))
            else:
                pts_m = pts_arr
            
            # Segment length
            diffs = np.linalg.norm(pts_m[1:] - pts_m[:-1], axis=1)
            seg_len = np.sum(diffs)
            
            start_d = cumulative_dist
            end_d = cumulative_dist + seg_len
            cumulative_dist = end_d
        
        # Match telemetry distance (relative to lap start)
        mask = (lap_data["Distance"] >= (lap_start_dist + start_d)) & \
               (lap_data["Distance"] < (lap_start_dist + end_d))
        
        seg_samples = lap_data[mask]
        
        if not seg_samples.empty:
            spd = seg_samples[speed_col]
            stats[idx] = (spd.min(), spd.max())
            if idx < 3:
                print(f"[sector-stats] Sector {idx}: dist {start_d:.0f}-{end_d:.0f}m, "
                      f"matched {len(seg_samples)} samples, speed {spd.min():.0f}-{spd.max():.0f}")
        else:
            if idx < 3:
                print(f"[sector-stats] Sector {idx}: dist {start_d:.0f}-{end_d:.0f}m, NO SAMPLES")
    
    print(f"[sector-stats] Lap {lap_id}: calculated stats for {len(stats)}/{len(sectors)} sectors")
    return stats


def draw_sectors_with_stats(draw, drawing_area, sectors, sector_stats):
    """
    Draw sector speed stats on track map.
    """
    inner_x, inner_y, width, height = drawing_area
    
    for idx, sector in enumerate(sectors):
        points = sector.get("points", [])
        if not points:
            continue
            
        # Scale points to drawing area
        scaled = [
            (
                inner_x + max(0.0, min(1.0, pt[0])) * width,
                inner_y + max(0.0, min(1.0, pt[1])) * height,
            )
            for pt in points
        ]
        
        stats = sector_stats.get(idx)
        if stats and scaled:
            min_spd, max_spd = stats
            mid_idx = len(scaled) // 2
            mid_pt = scaled[mid_idx]
            
            txt = f"{int(min_spd)}/{int(max_spd)}"
            
            try:
                fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
            except:
                fnt = ImageFont.load_default()
            
            x, y = mid_pt[0], mid_pt[1]
            
            # Draw text with outline
            for off in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + off[0], y + off[1]), txt, font=fnt, fill="black", anchor="mm")
            draw.text((x, y), txt, font=fnt, fill="#ffffff", anchor="mm")


def build_per_lap_track_maps(
    pipe: Pipeline,
    track: Track,
    session_table,
    lap_stats: list,
    crossings: list,
    width: int,
    height: int,
    fps: float,
) -> Pipeline:
    """
    Build per-lap track map overlays with sector statistics.
    
    Optimized: 1 static track PNG + ASS subtitles for per-lap stats.
    Much faster than multiple overlay filters.
    """
    v = pipe.video
    
    print(f"[per-lap-map] Starting: track={track is not None}, lap_stats={len(lap_stats) if lap_stats else 0}")
    
    if not track or not lap_stats:
        print("[per-lap-map] Missing track or lap_stats, skipping")
        return pipe
    
    # Use track.segments (auto-generated)
    sectors = track.segments or []
    print(f"[per-lap-map] Found {len(sectors)} sectors")
    
    if not sectors:
        print("[per-lap-map] No sectors defined, skipping")
        return pipe
    
    # Normalize bounds
    bounds = track.bounds
    min_x, max_x, min_y, max_y = bounds
    x_range = max_x - min_x or 1
    y_range = max_y - min_y or 1
    
    # Map dimensions and position
    map_w, map_h = 600, 600
    map_x, map_y = 30, 30
    padding = 20
    
    # Create static track image (just the lines, no stats)
    img = Image.new("RGBA", (map_w, map_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    for poly in track.polylines:
        norm_poly = [
            ((p[0] - min_x) / x_range, 1.0 - (p[1] - min_y) / y_range)
            for p in poly
        ]
        scaled = [(padding + p[0] * (map_w - 2 * padding), padding + p[1] * (map_h - 2 * padding)) for p in norm_poly]
        if len(scaled) > 1:
            draw.line(scaled, fill="#888888", width=4)
    
    # Draw sector boundaries (dots at start of each sector)
    sector_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
    for idx, sector in enumerate(sectors):
        pts = sector.get("points", [])
        if pts:
            # Start point of sector
            start_pt = pts[0]
            norm_x = (start_pt[0] - min_x) / x_range
            norm_y = 1.0 - (start_pt[1] - min_y) / y_range
            sx = padding + norm_x * (map_w - 2 * padding)
            sy = padding + norm_y * (map_h - 2 * padding)
            
            color = sector_colors[idx % len(sector_colors)]
            # Draw sector start marker (circle)
            r = 6
            draw.ellipse([(sx - r, sy - r), (sx + r, sy + r)], fill=color, outline="#000000")
    
    # Save static track image
    fd, static_track_path = tempfile.mkstemp(suffix="_track.png")
    os.close(fd)
    img.save(static_track_path)
    print(f"[per-lap-map] Saved static track: {static_track_path}")
    
    # Calculate normalized sector midpoints for ASS positioning
    sector_positions = []  # (screen_x, screen_y) for each sector
    for sector in sectors:
        pts = sector.get("points", [])
        if pts:
            mid_idx = len(pts) // 2
            mid_pt = pts[mid_idx]
            norm_x = (mid_pt[0] - min_x) / x_range
            norm_y = 1.0 - (mid_pt[1] - min_y) / y_range
            screen_x = map_x + padding + norm_x * (map_w - 2 * padding)
            screen_y = map_y + padding + norm_y * (map_h - 2 * padding)
            sector_positions.append((screen_x, screen_y))
        else:
            sector_positions.append((0, 0))
    
    # Generate ASS file for per-lap stats
    fd_ass, stats_ass_path = tempfile.mkstemp(suffix="_track_stats.ass")
    os.close(fd_ass)
    
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TrackStat,DejaVu Sans,14,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1
Style: LapLabel,DejaVu Sans,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def fmt_time(t):
        h = int(t / 3600)
        m = int((t % 3600) / 60)
        s = int(t % 60)
        cs = int((t * 100) % 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    
    with open(stats_ass_path, "w") as f:
        f.write(ass_header)
        
        for lap in lap_stats:
            lap_id = lap["id"]
            
            # Skip lap 0 (out-lap)
            if lap_id == 0:
                continue
            
            # Determine time range
            if lap_id - 1 < len(crossings):
                t_start = crossings[lap_id - 1]
            else:
                continue
            
            if lap_id < len(crossings):
                t_end = crossings[lap_id]
            else:
                t_end = session_table["Time"].max() if "Time" in session_table.columns else t_start + 120
            
            # Calculate stats for this lap
            sector_stats = calculate_sector_stats_for_lap(session_table, lap_id, sectors)
            
            if not sector_stats:
                continue
            
            s_str = fmt_time(t_start)
            e_str = fmt_time(t_end)
            
            # Write stat events for each sector
            for idx, (sx, sy) in enumerate(sector_positions):
                if idx in sector_stats:
                    min_spd, max_spd = sector_stats[idx]
                    txt = f"{int(min_spd)}/{int(max_spd)}"
                    f.write(f"Dialogue: 0,{s_str},{e_str},TrackStat,,0,0,0,,{{\\pos({sx:.0f},{sy:.0f})}}{txt}\n")
            
            # Lap label at bottom of map
            label_x = map_x + map_w // 2
            label_y = map_y + map_h - 10
            f.write(f"Dialogue: 1,{s_str},{e_str},LapLabel,,0,0,0,,{{\\pos({label_x},{label_y})}}Lap {lap_id}\n")
            
            print(f"[per-lap-map] Lap {lap_id}: {t_start:.1f}s - {t_end:.1f}s, {len(sector_stats)} sectors")
    
    # Cleanup registration
    def cleanup():
        for path in [static_track_path, stats_ass_path]:
            if os.path.exists(path):
                os.unlink(path)
    atexit.register(cleanup)
    
    # Apply: 1 static overlay + 1 subtitles filter
    track_input = ffmpeg.input(static_track_path, loop=1, framerate=fps)
    v = v.overlay(track_input, x=map_x, y=map_y, eof_action="pass")
    v = v.filter("subtitles", filename=stats_ass_path)
    
    return Pipeline(v, pipe.audio)


def build_track_map_overlay(
    pipe: Pipeline, track_overlay_data_obj, resampled_df, width, height, fps, lap_stats, seg_stats, best_lap=None
) -> Pipeline:
    """
    Generates static track map with stats and dynamic ASS events for position.
    """
    v = pipe.video

    if not track_overlay_data_obj:
        return pipe

    track_overlay_data = {
        "segments": track_overlay_data_obj.segments,
        "normalized_lines": track_overlay_data_obj.normalized_lines,
        "start_finish_normalized": track_overlay_data_obj.start_finish_normalized,
        "positions": track_overlay_data_obj.positions,
    }

    # Best lap ID
    best_lap_id = best_lap["id"] if best_lap else -1

    if best_lap_id != -1 and seg_stats:
        track_overlay_data["segment_stats"] = seg_stats
        track_overlay_data["current_lap"] = best_lap_id

    # 3. Create Static Map Image
    map_w, map_h = 400, 400
    map_x, map_y = 50, 50

    img = Image.new("RGBA", (map_w, map_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    map_box = (0, 0, map_w, map_h)

    drawing_area = draw_track_static(draw, map_box, track_overlay_data)
    draw_full_track_stats(draw, drawing_area, track_overlay_data)

    fd, static_map_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(static_map_path)

    # 4. Generate Dynamic ASS
    if "NormalizedX" in resampled_df.columns and "NormalizedY" in resampled_df.columns:
        fd_ass, dot_ass_path = tempfile.mkstemp(suffix=".ass")
        os.close(fd_ass)

        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dot,Arial,60,&H000000FF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        with open(dot_ass_path, "w") as f:
            f.write(ass_header)

            # Map coords
            inner_x, inner_y, area_w, area_h = drawing_area
            origin_x = map_x + inner_x
            origin_y = map_y + inner_y

            norm_x = resampled_df["NormalizedX"].fillna(0).values
            norm_y = resampled_df["NormalizedY"].fillna(0).values
            total_frames = len(resampled_df)

            def fmt_time(t):
                h = int(t / 3600)
                m = int((t % 3600) / 60)
                s = int(t % 60)
                cs = int((t * 100) % 100)
                return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

            for i in range(total_frames):
                nx = norm_x[i]
                ny = norm_y[i]

                # Check bounds/validity (0..1)
                # If 0,0 it might be invalid...

                start_t = i / fps
                end_t = (i + 1) / fps
                s_str = fmt_time(start_t)
                e_str = fmt_time(end_t)

                sx = origin_x + nx * area_w
                sy = origin_y + ny * area_h

                f.write(f"Dialogue: 0,{s_str},{e_str},Dot,,0,0,0,,{{\\pos({sx:.1f},{sy:.1f})}}.\n")

        v = v.filter("subtitles", filename=dot_ass_path)

    # Overlay Static Image - USE loop=1 !!
    v = v.overlay(ffmpeg.input(static_map_path, loop=1, framerate=fps), x=map_x, y=map_y, eof_action="pass")

    return Pipeline(v, pipe.audio)


def build_writer(pipe: Pipeline, output_path, vcodec=None, preset=None, crf=None, bitrate=None):
    # Force YUV420P for compatibility with most HW encoders (especially NVENC)
    v = pipe.video.filter("format", "yuv420p")

    kwargs = {}
    if vcodec:
        kwargs["vcodec"] = vcodec

    # Handle NVENC specific settings for better quality/defaults
    if vcodec and "nvenc" in vcodec:
        # NVENC uses -cq for quality (like CRF) and requires -rc vbr
        if preset:
            kwargs["preset"] = preset
        kwargs["tune"] = "hq"

        if crf is not None:
            kwargs["cq"] = crf
            kwargs["rc"] = "vbr"  # Variable bitrate mode
            kwargs["b:v"] = "0"  # Let VBR handle bitrate
        elif bitrate:
            kwargs["b:v"] = bitrate
        
        # Keyframe interval for proper seeking (NVENC uses -g)
        kwargs["g"] = 60  # Keyframe every ~1 second at 60fps
    elif vcodec and "svtav1" in vcodec:
        # SVT-AV1 specific settings
        if preset:
            kwargs["preset"] = preset  # 0-13 (0=slowest/best)
        if crf is not None:
            kwargs["crf"] = crf
        if bitrate:
            kwargs["b:v"] = bitrate
        
        # Keyframe interval for proper seeking (SVT-AV1 uses -g)
        kwargs["g"] = 60  # Keyframe every ~1 second at 60fps
    else:
        # Standard CPU encoders (libx264, libx265)
        if preset:
            kwargs["preset"] = preset
        if crf is not None:
            kwargs["crf"] = crf
        if bitrate:
            kwargs["video_bitrate"] = bitrate
        
        # Keyframe interval for proper seeking
        kwargs["g"] = 60  # Keyframe every ~1 second at 60fps

    return ffmpeg.output(v, pipe.audio, output_path, **kwargs).overwrite_output()


def export_best_lap(
    output_video: str | Path,
    video_session: "VideoSession",
    video_duration: float,
    trim_start: float = 0.0,
    buffer_seconds: float = 3.0,
) -> None:
    """
    Export the best lap from the output video as a separate file.

    Uses video_session.crossings which are already in display time (video - trim).
    """
    best_lap = video_session.best_lap
    if not best_lap:
        print("[Best Lap Export] No valid best lap found.")
        return

    best_lap_id = best_lap["id"]
    print(f"\n[Best Lap Export] Exporting Lap {best_lap_id} ({best_lap['time']:.3f}s)...")

    # Crossings are already in display time (video time minus trim)
    crossings = getattr(video_session, 'crossings', []) or []

    if not crossings or best_lap_id < 1 or best_lap_id > len(crossings):
        print(f"[Best Lap Export] Could not determine crossing times for Lap {best_lap_id}")
        return

    clip_start = crossings[best_lap_id - 1] - buffer_seconds
    clip_end = (crossings[best_lap_id] if best_lap_id < len(crossings) else crossings[-1] + best_lap['time']) + buffer_seconds
    
    # Clamp to valid range (output video is from 0 to output_duration)
    output_duration = video_duration - trim_start
    clip_start = max(0.0, clip_start)
    clip_end = min(output_duration, clip_end)
    clip_duration = clip_end - clip_start
    
    print(f"[Best Lap Export] Debug: output clip: start={clip_start:.2f}, end={clip_end:.2f}, duration={clip_duration:.2f}")
    
    if clip_duration <= 0:
        print("[Best Lap Export] Invalid clip duration, skipping.")
        return
    
    output_path = Path(output_video)
    best_lap_output = output_path.with_name(f"{output_path.stem}-best-lap{output_path.suffix}")
    
    print(f"[Best Lap Export] Range: {clip_start:.2f}s to {clip_end:.2f}s (duration: {clip_duration:.2f}s)")
    
    try:
        # Use ffmpeg to extract the clip from the OUTPUT video
        (
            ffmpeg
            .input(str(output_video), ss=clip_start, t=clip_duration)
            .output(str(best_lap_output), vcodec='copy', acodec='copy')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"[Best Lap Export] Saved to: {best_lap_output}")
    except ffmpeg.Error as e:
        print(f"[Best Lap Export] Failed: {e}")


def main() -> int:
    p = argparse.ArgumentParser(description="fisheye undistort + vid.stab + overlay with one final encode")
    p.add_argument("--in", dest="inp", required=True, help="input video path")
    p.add_argument("--out", dest="out", help="output video path")
    p.add_argument("--telemetry", dest="telemetry", default=None, help="path to telemetry session")
    p.add_argument("--track_dir", dest="track_dir", default=None, help="path to track directory")
    p.add_argument(
        "--overlay",
        default=None,
        help="overlay image/video path (supports alpha if format has it)",
    )

    p.add_argument("--balance", type=float, default=1.0, help="fisheye balance (0..1)")
    p.add_argument("--fov-scale", type=float, default=1.0, help="fisheye fov_scale (>1 wider)")

    p.add_argument("--stabilise", default=False, action="store_true", help="stabilise video")

    p.add_argument("--shakiness", type=int, default=10, help="vidstabdetect shakiness")
    p.add_argument("--accuracy", type=int, default=15, help="vidstabdetect accuracy")
    p.add_argument("--stepsize", type=int, default=4, help="vidstabdetect stepsize")
    p.add_argument("--smoothing", type=int, default=10, help="vidstabtransform smoothing")
    p.add_argument("--optzoom", type=int, default=0, help="vidstabtransform optzoom")
    p.add_argument("--zoom", type=float, default=0.1, help="vidstabtransform zoom")
    p.add_argument("--crop", default="keep", choices=["black", "keep"], help="Border cropping")
    p.add_argument(
        "--interpol",
        default="bilinear",
        choices=["no", "linear", "bilinear", "bicubic"],
        help="Interpolation",
    )

    p.add_argument("--vcodec", default=None, help="output video codec (default: auto-detect)")
    p.add_argument("--preset", default="7", help="encoder preset (SVT-AV1: 0-13, NVENC: p1-p7)")
    p.add_argument("--crf", type=int, default=28, help="CRF/CQ for AV1 (lower=better, 20-35 recommended)")

    p.add_argument("--intrinsics", help="path to camera intrinsics CSV")
    p.add_argument("--no-interactive", action="store_true", help="no interactive trim selection")

    p.add_argument(
        "--dynamic-overlay",
        action="store_true",
        help="Generate overlay dynamically via pipe",
    )
    p.add_argument(
        "--no-export-best-lap",
        dest="export_best_lap",
        action="store_false",
        default=True,
        help="Disable exporting best lap (enabled by default)",
    )

    args = p.parse_args()
    # Get trim info
    inp_path = Path(args.inp)
    trim_info = get_trim_info(inp_path, args.no_interactive)

    # Determine Codec and HW Accel
    hwaccel = None
    if args.vcodec is None:
        if check_cuda_availability():
            # Use AV1 NVENC for best quality/compression
            args.vcodec = "av1_nvenc"
            hwaccel = "cuda"
            # NVENC AV1 uses p1-p7 presets (p7=slowest/best)
            if args.preset == "7":  # If user didn't change default
                args.preset = "p7"
            print(f"CUDA detected: Using {args.vcodec} with hwaccel={hwaccel}")
        else:
            args.vcodec = "libsvtav1"  # CPU AV1 encoder
            print(f"Using CPU encoder: {args.vcodec}")
    else:
        # If user specified something, check if it implies cuda (simple heuristic or manual)
        if "nvenc" in args.vcodec and check_cuda_availability():
            hwaccel = "cuda"

    if not args.out:
        inp_path = Path(args.inp)
        args.out = str(inp_path.with_name(f"{inp_path.stem}_output{inp_path.suffix}"))

    video_info = probe_video(Path(args.inp))  # Detect video info

    session, track = None, None

    if args.track_dir:
        track = Track.load(args.track_dir)

    # --- Step 1: Always get video crossings via manual lap marking ---
    crossings_sidecar = VideoSidecar.load(Path(args.inp), "crossings")
    if crossings_sidecar.exists:
        print(f"[Crossings] Found saved video crossings: {len(crossings_sidecar.get('times', []))} laps")
        if not args.no_interactive:
            if input("Regenerate lap markings? [y/N]: ").strip().lower() == "y":
                crossings_sidecar.exists = False

    if not crossings_sidecar.exists and not args.no_interactive:
        times = run_manual_lap_marking(args.inp, start_time=trim_info.start if trim_info else 0.0)
        if times:
            crossings_sidecar.save({"times": times})

    crossings_video: list[float] = crossings_sidecar.get("times", [])

    # --- Step 2: Load telemetry (if any) and build piecewise sync ---
    sync_mapping: PiecewiseSync | None = None

    if args.telemetry:
        session = Session.load(args.telemetry)
        if track:
            session.track = track.geometry
            session.detect_crossings()

        # Fallback: infer crossings from Lap column transitions
        if not session.crossings and "Lap" in session.table.columns:
            import pandas as pd
            laps = pd.to_numeric(session.table["Lap"], errors="coerce").ffill().fillna(1)
            t_vals = session.table["Time"].values
            inferred = [float(t_vals[i]) for i in range(1, len(laps)) if laps.iloc[i] != laps.iloc[i - 1]]
            if inferred:
                session.crossings = inferred
                print(f"[Crossings] Inferred {len(inferred)} crossings from Lap column")

        session.add_lap_numbers()
        crossings_telem: list[float] = session.crossings or []

        # Export to MoTeC .ld format
        motec_output = Path(args.telemetry).with_suffix(".ld")
        session.to_motec(output=motec_output, frequency=10.0)
        print(f"[MoTeC] Exported to {motec_output}")

        # Build piecewise sync from matched crossing pairs
        n_pairs = min(len(crossings_video), len(crossings_telem))
        if n_pairs >= 1:
            anchors = list(zip(crossings_video[:n_pairs], crossings_telem[:n_pairs]))
            sync_mapping = PiecewiseSync(anchors=anchors)
            print(f"[Sync] Built piecewise mapping with {n_pairs} anchor points:")
            for i, (v, t) in enumerate(anchors):
                print(f"  Crossing {i+1}: video={v:.3f}s <-> telem={t:.3f}s (offset={t-v:.3f}s)")
            if len(crossings_video) != len(crossings_telem):
                print(f"[Sync] Warning: crossing count mismatch — video={len(crossings_video)}, telem={len(crossings_telem)}")
        else:
            print("[Sync] Warning: no matching crossings, using offset=0")
            sync_mapping = PiecewiseSync.from_offset(0.0)
    else:
        # No telemetry — create session from video crossings
        session = create_session_from_crossings(video_info, crossings_video)
        session.crossings = crossings_video
        session.add_lap_numbers()

    # --- Step 3: Build video session and resample ---
    video_session = VideoSession.from_session(session, Path(args.inp))
    trim_start_time = trim_info.start if trim_info else 0.0

    if args.telemetry and sync_mapping is not None:
        video_session.table = video_session.resample_to_video(
            fps=video_info.fps,
            trim_start=trim_start_time,
            duration=((trim_info.end - trim_info.start) if trim_info and trim_info.end else video_info.duration),
            sync=sync_mapping,
        )

    # Use video crossings (adjusted for trim) as display crossings
    if crossings_video:
        video_session.crossings = [max(0.0, c - trim_start_time) for c in crossings_video]
        print(f"[Crossings] Display crossings (trim={trim_start_time:.2f}): {video_session.crossings[:3]}...")
    # Store GPS crossings for comparison in lap table
    if args.telemetry and sync_mapping is not None:
        video_session.crossings_gps = crossings_telem

    pipeline = build_opener(Path(args.inp), hwaccel=hwaccel)

    if trim_info:
        pipeline = build_trimer(pipeline, trim_info.start, trim_info.end)

    if args.intrinsics:
        camera_model = CameraModel.load(Path(args.intrinsics))  # Load intrinsics
        pipeline = build_undistorter(pipeline, camera_model, args.balance, args.fov_scale, video_info)

    if args.stabilise:
        # Transform file should be in same folder as video with .trf extension
        transforms_filepath = Path(args.inp).with_suffix(".trf")
        
        # Determine if we need to run Pass 1 (transform estimation)
        run_pass1 = True
        if transforms_filepath.exists():
            print(f"Found existing transform file: {transforms_filepath}")
            regenerate = input("Regenerate? [y/N]: ").strip().lower()
            run_pass1 = (regenerate == "y")
            if not run_pass1:
                print("Using existing transform file")
        
        if run_pass1:
            print(f"Pass 1: Detecting stability (shakiness={args.shakiness})...")
            t_pipe = build_transform_estimator(
                pipeline,
                transforms_filepath,
                args.shakiness,
                args.accuracy,
                args.stepsize,
            )
            # Run to null output - this generates the transform file as a side effect
            ffmpeg.output(t_pipe.video, os.devnull, format="null").run(overwrite_output=True)
            print(f"Transform file saved: {transforms_filepath}")

        print(f"Pass 2: Stabilizing (smoothing={args.smoothing})...")
        pipeline = build_stabilizer(
            pipeline,
            transforms_filepath,
            smoothing=args.smoothing,
            zoom=args.zoom,
            optzoom=args.optzoom,
            crop=args.crop,
            interpol=args.interpol,
        )
    # Apply Lap List Overlay
    pipeline = build_lap_stats_ov(pipeline, video_session)

    if args.telemetry:
        pipeline = build_gauge_overlay(pipeline, video_session)
        
        # Per-lap track map overlay
        if track:
            crossings = getattr(session, 'crossings', []) or []
            pipeline = build_per_lap_track_maps(
                pipe=pipeline,
                track=track,
                session_table=video_session.table,
                lap_stats=video_session.get_lap_stats(),
                crossings=crossings,
                width=video_info.width,
                height=video_info.height,
                fps=video_info.fps,
            )

    output = build_writer(pipeline, args.out, vcodec=args.vcodec, preset=args.preset, crf=args.crf)
    
    # Ignore SIGINT in Python so FFmpeg can handle Ctrl+C gracefully
    import signal
    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        output.run()
    except ffmpeg.Error:
        # FFmpeg was interrupted by SIGINT - this is expected, output should be saved
        print(f"\n[Interrupted] Output saved to: {args.out}")
        return 0
    finally:
        signal.signal(signal.SIGINT, original_handler)
    
    # Export best lap if enabled
    if args.export_best_lap:
        export_best_lap(
            output_video=args.out,
            video_session=video_session,
            video_duration=video_info.duration,
            trim_start=trim_info.start if trim_info else 0.0,
        )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
