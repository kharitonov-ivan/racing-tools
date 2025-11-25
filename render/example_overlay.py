#!/usr/bin/env python3
"""
Example helper that renders a short telemetry overlay demo.

The script trims a 2–3 minute segment from the source video, resamples the
matching telemetry using the converter package, draws a lightweight overlay for
each frame, and finally asks ffmpeg to composite both streams.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import os

import numpy as np
import pandas as pd
import shapefile
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor, as_completed


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_SRC = REPO_ROOT / "converter" / "converter"
if str(CONVERTER_SRC) not in sys.path:
    sys.path.append(str(CONVERTER_SRC))

from convert import (  # type: ignore  # pylint: disable=import-error
    aim_session,
    alfano_session,
    alfano_excel_session,
    detect,
)


WGS84_TO_WEBMERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
WEBMERC_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float


@dataclass
class TrackLayout:
    polylines: list[list[tuple[float, float]]]
    bounds: tuple[float, float, float, float]


@dataclass
class TrackGeometry:
    layout: TrackLayout
    start_finish_webmerc: list[tuple[float, float]] | None = None
    start_finish_wgs84: list[tuple[float, float]] | None = None


@dataclass
class TrackOverlay:
    normalized_lines: list[list[tuple[float, float]]]
    positions: np.ndarray | None
    start_finish_normalized: list[tuple[float, float]] | None = None
    start_finish_wgs84: list[tuple[float, float]] | None = None


def to_float(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_duration(value: float | int | None, decimals: int = 1) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "--:--"
    total = max(0.0, float(value))
    minutes = int(total // 60)
    seconds = total - minutes * 60
    seconds_fmt = f"{seconds:04.{decimals}f}"
    return f"{minutes}:{seconds_fmt}"


def probe_video(path: Path) -> VideoInfo:
    """Use ffprobe to collect geometry/fps info."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout or "{}")
    stream = data.get("streams", [{}])[0]
    fmt = data.get("format", {})
    rate = stream.get("r_frame_rate", "0/1")
    num, _, den = rate.partition("/")
    fps = to_float(num) / max(to_float(den), 1.0)
    return VideoInfo(
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=fps or 30.0,
        duration=to_float(fmt.get("duration", 0.0)),
    )


def load_session(folder: Path, frequency: float, normalize: bool = True):
    """Detect telemetry type and return a normalized session object."""
    kind = detect(folder)
    if kind == "aim":
        return aim_session(folder, frequency, normalize)
    if kind == "alfano":
        return alfano_session(folder, normalize)
    if kind == "alfano_excel":
        session, _freq = alfano_excel_session(folder, normalize)
        return session
    raise ValueError(f"Unsupported telemetry folder {folder}")


def pick_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def resample_telemetry(
    session,
    *,
    fps: float,
    duration: float,
    time_shift: float,
    clip_start: float,
) -> pd.DataFrame:
    """Align telemetry samples with the video timeline."""
    table = session.table.copy()
    if "Time" not in table.columns:
        raise ValueError("Telemetry data does not contain Time column")

    table["Time"] = pd.to_numeric(table["Time"], errors="coerce")
    table = table.dropna(subset=["Time"]).set_index("Time").sort_index()
    table = table.infer_objects(copy=False)

    total_frames = max(1, int(math.ceil(duration * fps)))
    relative_video_times = np.arange(total_frames, dtype=float) / fps
    absolute_video_times = clip_start + relative_video_times
    telemetry_times = absolute_video_times + time_shift

    min_time = float(table.index.min())
    max_time = float(table.index.max())
    telemetry_times = np.clip(telemetry_times, min_time, max_time)
    index = pd.Index(telemetry_times, name="Time")

    interpolated = (
        table.reindex(table.index.union(index))
        .sort_index()
        .interpolate(method="index")
        .ffill()
        .bfill()
    )
    aligned = interpolated.loc[index].reset_index(drop=False)
    aligned["VideoTime"] = relative_video_times
    if "LapNumber" in aligned.columns:
        lap_series = (
            pd.to_numeric(aligned["LapNumber"], errors="coerce")
            .round()
            .ffill()
            .bfill()
        )
        if lap_series.notna().any():
            aligned["LapNumber"] = lap_series.astype(int)
    if "LapTime" in aligned.columns:
        lap_time = pd.to_numeric(aligned["LapTime"], errors="coerce")
        aligned["LapTime"] = lap_time.clip(lower=0.0)
    return aligned


def ensure_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Pick a readable font; gracefully fallback to Pillow's default."""
    candidates = [
        REPO_ROOT / "render" / "fonts" / ("Inter-SemiBold.ttf" if bold else "Inter-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/SFNSDisplay.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_bar(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    ratio: float,
    fill,
    outline,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=outline, width=3)
    usable = max(x1 - x0 - 6, 1)
    filled = int(usable * max(0.0, min(1.0, ratio)))
    draw.rectangle((x0 + 3, y0 + 3, x0 + 3 + filled, y1 - 3), fill=fill)


def load_track_layout(track_dir: Path) -> TrackLayout:
    shp_path = track_dir / "centerline" / "centerline.shp"
    if not shp_path.is_file():
        raise FileNotFoundError(f"centerline shapefile missing in {track_dir}")
    reader = shapefile.Reader(str(shp_path))
    polylines: list[list[tuple[float, float]]] = []
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    for shape in reader.shapes():
        points = shape.points
        parts = list(shape.parts) + [len(points)]
        for i in range(len(parts) - 1):
            segment = points[parts[i] : parts[i + 1]]
            if not segment:
                continue
            polylines.append(segment)
            xs, ys = zip(*segment)
            min_x = min(min_x, min(xs))
            max_x = max(max_x, max(xs))
            min_y = min(min_y, min(ys))
            max_y = max(max_y, max(ys))
    if not polylines:
        raise ValueError(f"No shapes found in {shp_path}")
    bounds = (min_x, max_x, min_y, max_y)
    return TrackLayout(polylines=polylines, bounds=bounds)


def load_polyline(path: Path) -> list[tuple[float, float]] | None:
    if not path.is_file():
        return None
    reader = shapefile.Reader(str(path))
    shapes = reader.shapes()
    if not shapes:
        return None
    shape = shapes[0]
    start = shape.parts[0] if shape.parts else 0
    end = shape.parts[1] if len(shape.parts) > 1 else len(shape.points)
    return shape.points[start:end]


def load_track_geometry(track_dir: Path) -> TrackGeometry:
    layout = load_track_layout(track_dir)
    sf_path = track_dir / "start-finish-line" / "start-finish-line.shp"
    start_finish = load_polyline(sf_path)
    start_finish_wgs84 = None
    if start_finish:
        xs, ys = zip(*start_finish)
        lons, lats = WEBMERC_TO_WGS84.transform(np.array(xs), np.array(ys))
        start_finish_wgs84 = list(zip(lons, lats))
    return TrackGeometry(
        layout=layout,
        start_finish_webmerc=start_finish,
        start_finish_wgs84=start_finish_wgs84,
    )


def normalize_track_polylines(layout: TrackLayout) -> list[list[tuple[float, float]]]:
    min_x, max_x, min_y, max_y = layout.bounds
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    normalized: list[list[tuple[float, float]]] = []
    for line in layout.polylines:
        scaled = []
        for x, y in line:
            nx = (x - min_x) / span_x
            ny = 1.0 - (y - min_y) / span_y
            scaled.append((nx, ny))
        normalized.append(scaled)
    return normalized


def normalize_polyline(points: list[tuple[float, float]] | None, bounds: tuple[float, float, float, float]):
    if not points:
        return None
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    return [
        ((x - min_x) / span_x, 1.0 - (y - min_y) / span_y)
        for x, y in points
    ]


def normalize_track_positions(points: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    if points.size == 0:
        return points
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    xs = (points[:, 0] - min_x) / span_x
    ys = 1.0 - (points[:, 1] - min_y) / span_y
    stacked = np.column_stack(
        [np.clip(xs, 0.0, 1.0), np.clip(ys, 0.0, 1.0)],
    )
    return stacked


def build_track_overlay(geometry: TrackGeometry, samples: pd.DataFrame) -> TrackOverlay | None:
    layout = geometry.layout

    lat_col = pick_column(samples, ["GPS Latitude", "Latitude"])
    lon_col = pick_column(samples, ["GPS Longitude", "Longitude"])
    if not lat_col or not lon_col:
        print("[overlay] Telemetry lacks GPS Latitude/Longitude; track map disabled")
        return None

    lat = pd.to_numeric(samples[lat_col], errors="coerce").interpolate().ffill().bfill()
    lon = pd.to_numeric(samples[lon_col], errors="coerce").interpolate().ffill().bfill()
    if lat.isna().all() or lon.isna().all():
        print("[overlay] Unable to derive numeric GPS coordinates; track map disabled")
        return None

    xs, ys = WGS84_TO_WEBMERC.transform(lon.to_numpy(), lat.to_numpy())
    positions = np.column_stack([xs, ys])
    normalized_positions = normalize_track_positions(positions, layout.bounds)
    normalized_lines = normalize_track_polylines(layout)
    start_finish_norm = normalize_polyline(geometry.start_finish_webmerc, layout.bounds)
    return TrackOverlay(
        normalized_lines=normalized_lines,
        positions=normalized_positions,
        start_finish_normalized=start_finish_norm,
        start_finish_wgs84=geometry.start_finish_wgs84,
    )


def draw_track_map(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    overlay: TrackOverlay,
    frame_index: int,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=(0, 0, 0, 140), outline="#214d66")
    pad = 18
    inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
    width = max(inner[2] - inner[0], 1)
    height = max(inner[3] - inner[1], 1)

    for line in overlay.normalized_lines:
        if len(line) < 2:
            continue
        scaled = [
            (
                inner[0] + max(0.0, min(1.0, pt[0])) * width,
                inner[1] + max(0.0, min(1.0, pt[1])) * height,
            )
            for pt in line
        ]
        draw.line(scaled, fill="#5ad2ff", width=5, joint="curve")

    if overlay.start_finish_normalized and len(overlay.start_finish_normalized) >= 2:
        sf_points = [
            (
                inner[0] + max(0.0, min(1.0, pt[0])) * width,
                inner[1] + max(0.0, min(1.0, pt[1])) * height,
            )
            for pt in overlay.start_finish_normalized
        ]
        draw.line(sf_points, fill="#ffd479", width=6)

    if overlay.positions is None or frame_index >= len(overlay.positions):
        return
    pos = overlay.positions[frame_index]
    if np.isnan(pos).any():
        return
    px = inner[0] + float(np.clip(pos[0], 0.0, 1.0)) * width
    py = inner[1] + float(np.clip(pos[1], 0.0, 1.0)) * height
    draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill="#ff7272", outline="white", width=2)


def render_track_frames(
    data: pd.DataFrame,
    *,
    session,
    video: VideoInfo,
    track_overlay: TrackOverlay,
    output_dir: Path,
) -> tuple[int, tuple[int, int]]:
    frames = len(data)
    if frames == 0 or track_overlay.positions is None:
        return 0, (0, 0)

    map_size = min(int(video.width * 0.28), 420)
    overlay_width = map_size + 120
    overlay_height = map_size + 140
    output_dir.mkdir(parents=True, exist_ok=True)

    title_font = ensure_font(38, bold=True)
    meta_font = ensure_font(28, bold=False)

    map_box = (60, 90, 60 + map_size, 90 + map_size)

    for index, row in data.reset_index(drop=True).iterrows():
        img = Image.new("RGBA", (overlay_width, overlay_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, overlay_width, overlay_height), radius=26, fill=(10, 14, 20, 200))

        header = session.venue or "Track"
        draw.text((32, 24), f"{header} · Track Map", font=title_font, fill="#e8f6ff")

        draw_track_map(draw, box=map_box, overlay=track_overlay, frame_index=index)

        video_time = row.get("VideoTime", 0.0)
        minutes = int(video_time // 60)
        seconds = video_time % 60
        draw.text((32, overlay_height - 60), f"T+{minutes:02d}:{seconds:04.1f}", font=meta_font, fill="#9ad5ff")

        target = output_dir / f"{index:05d}.png"
        img.save(target)

    return frames, (overlay_width, overlay_height)


def render_info_frame(
    index: int,
    row: pd.Series,
    *,
    session,
    overlay_width: int,
    overlay_height: int,
    font_large,
    font_medium,
    font_small,
    heading_values,
    lap_numbers,
    lap_time_values,
    speed_values,
    rpm_values,
    max_speed: float,
    max_rpm: float,
    lat_values,
    lon_values,
    distance_values,
) -> tuple[int, Image.Image]:
    img = Image.new("RGBA", (overlay_width, overlay_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, overlay_width, overlay_height), radius=28, fill=(12, 12, 12, 200))

    header = f"{session.driver or 'Driver'} · {session.venue or session.session or 'Session'}"
    draw.text((32, 24), header, font=font_large, fill="white")

    video_time = row.get("VideoTime", 0.0)
    minutes = int(video_time // 60)
    seconds = video_time % 60
    timer_text = f"T+{minutes:02d}:{seconds:04.1f}"
    draw.text((overlay_width - 210, 28), timer_text, font=font_medium, fill="#9ad5ff")

    lap_y = 88
    if lap_numbers is not None and not pd.isna(lap_numbers.iloc[index]):
        lap_val = int(lap_numbers.iloc[index])
        lap_label = "PIT" if lap_val < 0 else "OUT" if lap_val == 0 else str(lap_val)
        draw.text((32, lap_y), f"Lap {lap_label}", font=font_medium, fill="#99f0ff")
    if lap_time_values is not None:
        lap_time_text = format_duration(lap_time_values.iloc[index])
        draw.text((overlay_width - 260, lap_y), f"{lap_time_text}", font=font_medium, fill="#ffffff")

    heading_y = lap_y + 42
    if heading_values is not None:
        heading = float(heading_values.iloc[index])
        draw.text((overlay_width - 210, heading_y), f"{heading:6.1f}°", font=font_medium, fill="#ffd1a6")

    y_offset = heading_y + 32
    if speed_values is not None:
        speed = float(speed_values.iloc[index])
        draw.text((32, y_offset - 8), f"Speed {speed:6.1f} km/h", font=font_medium, fill="#fbdc6c")
        draw_bar(
            draw,
            box=(32, y_offset + 32, overlay_width - 32, y_offset + 68),
            ratio=speed / max(max_speed, 1.0),
            fill=(251, 220, 108, 230),
            outline="#857f51",
        )
        y_offset += 84

    if rpm_values is not None:
        rpm = float(rpm_values.iloc[index])
        draw.text((32, y_offset - 8), f"RPM {rpm:7.0f}", font=font_medium, fill="#96fca8")
        draw_bar(
            draw,
            box=(32, y_offset + 32, overlay_width - 32, y_offset + 68),
            ratio=rpm / max(max_rpm, 1.0),
            fill=(150, 252, 168, 230),
            outline="#4c7c58",
        )
        y_offset += 84

    metrics = []
    if distance_values is not None:
        metrics.append(f"Distance {distance_values.iloc[index] / 1000.0:5.2f} km")
    if lat_values is not None and lon_values is not None:
        metrics.append(f"G (Lat/Long) {lat_values.iloc[index]:+4.2f}/{lon_values.iloc[index]:+4.2f}")
    elif lat_values is not None:
        metrics.append(f"G (Lat) {lat_values.iloc[index]:+4.2f}")
    elif lon_values is not None:
        metrics.append(f"G (Long) {lon_values.iloc[index]:+4.2f}")
    if heading_values is not None:
        metrics.append(f"Heading {heading_values.iloc[index]:6.1f}°")

    footer = " · ".join(metrics) if metrics else (session.event_date or "")
    if footer:
        draw.text((32, overlay_height - 60), footer, font=font_small, fill="#d2d2d2")
    return index, img


def render_info_frames(
    data: pd.DataFrame,
    *,
    session,
    video: VideoInfo,
    output_dir: Path,
) -> tuple[int, tuple[int, int]]:
    """Render semi-transparent overlay frames for ffmpeg."""
    data = data.reset_index(drop=True).copy()
    overlay_width = max(int(video.width * 0.38), 520)
    overlay_height = max(int(video.height * 0.22), 220)
    output_dir.mkdir(parents=True, exist_ok=True)

    font_large = ensure_font(52, bold=True)
    font_medium = ensure_font(36, bold=True)
    font_small = ensure_font(28, bold=False)

    speed_col = pick_column(data, ["GPS Speed", "Speed", "Wheel Speed"])
    rpm_col = pick_column(data, ["RPM", "Engine RPM"])
    lat_col = pick_column(data, ["GPS Accel Lat", "IMU Accel Lat", "IMU Accel Lat Filtered"])
    lon_col = pick_column(data, ["GPS Accel Long", "IMU Accel Long", "IMU Accel Long Filtered"])
    heading_col = pick_column(data, ["Heading", "GPS Heading"])
    lap_col = "LapNumber" if "LapNumber" in data.columns else None
    lap_time_col = "LapTime" if "LapTime" in data.columns else None
    distance_col = pick_column(data, ["Distance"])

    def numeric_series(column: str | None):
        if not column or column not in data.columns:
            return None
        return pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    speed_values = numeric_series(speed_col)
    rpm_values = numeric_series(rpm_col)
    lat_values = numeric_series(lat_col)
    lon_values = numeric_series(lon_col)
    distance_values = numeric_series(distance_col)
    heading_values = numeric_series(heading_col)
    lap_time_values = numeric_series(lap_time_col)
    lap_numbers = (
        pd.to_numeric(data[lap_col], errors="coerce").round().astype("Int64") if lap_col else None
    )

    max_speed = float(speed_values.quantile(0.995)) if speed_values is not None else 0.0
    max_rpm = float(rpm_values.quantile(0.995)) if rpm_values is not None else 0.0

    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [
            executor.submit(
                render_info_frame,
                index,
                row,
                session=session,
                overlay_width=overlay_width,
                overlay_height=overlay_height,
                font_large=font_large,
                font_medium=font_medium,
                font_small=font_small,
                heading_values=heading_values,
                lap_numbers=lap_numbers,
                lap_time_values=lap_time_values,
                speed_values=speed_values,
                rpm_values=rpm_values,
                max_speed=max_speed,
                max_rpm=max_rpm,
                lat_values=lat_values,
                lon_values=lon_values,
                distance_values=distance_values,
            )
            for index, row in data.iterrows()
        ]
        for future in as_completed(futures):
            results.append(future.result())

    for index, img in sorted(results, key=lambda pair: pair[0]):
        target = output_dir / f"{index:05d}.png"
        img.save(target)

    return len(data), (overlay_width, overlay_height)


def run_ffmpeg(
    *,
    video_path: Path,
    overlay_specs: list[dict[str, str]],
    fps: float,
    start: float,
    duration: float,
    output_path: Path,
    hwaccel_cuda: bool = False,
    video_codec: str = "libx264",
) -> None:
    """Invoke ffmpeg to trim, overlay, and export."""
    cmd = ["ffmpeg", "-y"]
    if hwaccel_cuda:
        cmd.extend(["-hwaccel", "cuda"])
    cmd.extend([
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video_path),
    ])
    for spec in overlay_specs:
        cmd.extend([
            "-framerate",
            f"{fps:.5f}",
            "-i",
            spec["pattern"],
        ])

    video_map = "0:v"
    if overlay_specs:
        lines: list[str] = []
        current = "[0:v]"
        for idx, spec in enumerate(overlay_specs):
            overlay_stream = f"[{idx + 1}:v]"
            out_label = "[vout]" if idx == len(overlay_specs) - 1 else f"[tmp{idx}]"
            lines.append(f"{current}{overlay_stream}overlay={spec['x']}:{spec['y']}:format=auto{out_label}")
            current = out_label
        cmd.extend(["-filter_complex", ";".join(lines)])
        video_map = current

    codec = video_codec.lower()
    quality_flags: list[str] = []
    if "nvenc" in codec:
        quality_flags = ["-cq", "19", "-b:v", "0"]
    else:
        quality_flags = ["-crf", "18"]

    cmd.extend([
        "-map",
        video_map,
        "-map",
        "0:a?",
        "-c:v",
        video_codec,
        "-preset",
        "medium",
        *quality_flags,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ])
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a telemetry overlay demo clip.")
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "render" / "session-01.mp4")
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "telemetry"
        / "2025-11-18-mychron5-RIMSportKarting-RotaxMax-KharitonovIvan-session-01",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "render" / "session-01-overlay-sample.mp4")
    parser.add_argument("--start", type=float, default=60.0, help="Clip start time (seconds).")
    parser.add_argument("--duration", type=float, default=150.0, help="Clip duration in seconds.")
    parser.add_argument(
        "--time-shift",
        type=float,
        default=0.0,
        help="Telemetry time offset relative to the trimmed video (seconds).",
    )
    parser.add_argument("--telemetry-frequency", type=float, default=20.0, help="Samplerate hint for AIM sessions.")
    parser.add_argument(
        "--track-dir",
        type=Path,
        default=REPO_ROOT / "data" / "tracks" / "RustaviKarting",
        help="Track shape directory (expects centerline shapefile).",
    )
    parser.add_argument("--keep-frames", action="store_true", help="Do not delete generated overlay PNG frames.")
    parser.add_argument("--hwaccel-cuda", action="store_true", help="Add '-hwaccel cuda' to the ffmpeg command.")
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libx264",
        help="Video codec passed to ffmpeg (e.g. libx264, h264_nvenc).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    telemetry_dir = args.telemetry.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not video_path.is_file():
        raise SystemExit(f"Video {video_path} is missing")
    if not telemetry_dir.is_dir():
        raise SystemExit(f"Telemetry folder {telemetry_dir} is missing")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe_video(video_path)
    available = max(0.0, info.duration - args.start)
    duration = max(1.0, min(args.duration, available))

    track_geometry = None
    if args.track_dir:
        track_dir = args.track_dir.expanduser().resolve()
        try:
            track_geometry = load_track_geometry(track_dir)
        except FileNotFoundError:
            print(f"[overlay] Track directory {track_dir} missing, skipping map")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[overlay] Failed to load track layout: {exc}")

    print(f"[overlay] Loading telemetry from {telemetry_dir.name}")
    session = load_session(telemetry_dir, args.telemetry_frequency, True)
    if track_geometry and track_geometry.start_finish_wgs84:
        session.estimate_laps(track_geometry.start_finish_wgs84)
    samples = resample_telemetry(
        session,
        fps=info.fps,
        duration=duration,
        time_shift=args.time_shift,
        clip_start=args.start,
    )
    track_overlay = None
    if track_geometry:
        track_overlay = build_track_overlay(track_geometry, samples)

    video_codec = args.video_codec
    if args.hwaccel_cuda and args.video_codec == "libx264":
        video_codec = "h264_nvenc"

    tmp_root = Path(tempfile.mkdtemp(prefix="telemetry_overlay_"))
    try:
        info_dir = tmp_root / "info"
        print(f"[overlay] Drawing info overlay frames in {info_dir}")
        frame_count, info_size = render_info_frames(
            samples,
            session=session,
            video=info,
            output_dir=info_dir,
        )
        print(f"[overlay] Generated {frame_count} info frames ({info_size[0]}x{info_size[1]})")

        overlay_specs = [
            {
                "pattern": str(info_dir / "%05d.png"),
                "x": f"main_w-{info_size[0]}-60",
                "y": f"main_h-{info_size[1]}-60",
            }
        ]

        if track_overlay:
            track_dir = tmp_root / "track"
            track_count, track_size = render_track_frames(
                samples,
                session=session,
                video=info,
                track_overlay=track_overlay,
                output_dir=track_dir,
            )
            if track_count:
                print(f"[overlay] Generated {track_count} track-map frames ({track_size[0]}x{track_size[1]})")
                overlay_specs.insert(
                    0,
                    {
                        "pattern": str(track_dir / "%05d.png"),
                        "x": "60",
                        "y": "60",
                    },
                )

        print(f"[overlay] Calling ffmpeg for compositing -> {output_path.name}")
        run_ffmpeg(
            video_path=video_path,
            overlay_specs=overlay_specs,
            fps=info.fps,
            start=args.start,
            duration=duration,
            output_path=output_path,
            hwaccel_cuda=args.hwaccel_cuda,
            video_codec=video_codec,
        )
    finally:
        if args.keep_frames:
            print(f"[overlay] Keeping frames under {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
