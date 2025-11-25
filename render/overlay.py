#!/usr/bin/env python3
"""
Overlay Program
===============

Renders a telemetry overlay for racing videos.
Features:
- Track map with position
- Speed, RPM, G-Force gauges
- Lap timer (current lap time)
- Lap counter
- Automatic synchronization with video

Usage:
    python3 render/overlay.py --video <video.mp4> --telemetry <folder> --output <out.mp4>
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
from tqdm import tqdm

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_SRC = REPO_ROOT / "converter" / "converter"
if str(CONVERTER_SRC) not in sys.path:
    sys.path.append(str(CONVERTER_SRC))

from convert import (  # type: ignore
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
    segments: list[dict] | None = None


class CenterlineProjector:
    def __init__(self, points: np.ndarray):
        self.points = points
        # Calculate cumulative distance along polyline
        dists = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)
        self.cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
        self.total_length = self.cumulative_dists[-1]
        
        # Pre-calculate segment vectors
        self.segments = self.points[1:] - self.points[:-1]
        self.segment_lengths_sq = np.sum(self.segments**2, axis=1)
        
    def project(self, point: np.ndarray) -> float:
        """Project point onto nearest segment and return distance along centerline."""
        # Vector from segment start to point
        v_start_point = point - self.points[:-1]
        
        # Project v_start_point onto segment vector
        # t = dot(v_start_point, segment) / |segment|^2
        # Handle zero-length segments to avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.sum(v_start_point * self.segments, axis=1) / self.segment_lengths_sq
            t = np.nan_to_num(t) # Treat 0/0 as 0
        
        # Clamp t to [0, 1]
        t_clamped = np.clip(t, 0, 1)
        
        # Find closest point on each segment
        closest_points = self.points[:-1] + self.segments * t_clamped[:, np.newaxis]
        
        # Distance from query point to closest point on each segment
        dists_sq = np.sum((closest_points - point)**2, axis=1)
        
        # Find index of closest segment
        min_idx = np.argmin(dists_sq)
        
        # Calculate distance along centerline
        seg_len = np.sqrt(self.segment_lengths_sq[min_idx])
        dist_along = self.cumulative_dists[min_idx] + t_clamped[min_idx] * seg_len
        
        return float(dist_along)


class PredictiveLapModel:
    def __init__(self, distance_time_map: list[tuple[float, float]]):
        # distance_time_map is list of (distance, time)
        # Sort by distance
        data = np.array(distance_time_map)
        # Sort by distance
        order = np.argsort(data[:, 0])
        self.dists = data[order, 0]
        self.times = data[order, 1]
        
        # Remove duplicates
        unique_indices = np.unique(self.dists, return_index=True)[1]
        self.dists = self.dists[unique_indices]
        self.times = self.times[unique_indices]
        
    def get_time(self, distance: float) -> float:
        # Linear interpolation
        return float(np.interp(distance, self.dists, self.times))


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
    # Width: 2 digits + 1 dot + decimals (e.g. 05.2f for 09.50)
    width = 3 + decimals
    seconds_fmt = f"{seconds:0{width}.{decimals}f}"
    return f"{minutes:02d}:{seconds_fmt}"


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
    # Remove duplicate timestamps to avoid combinatorial explosion during reindex/loc
    table = table[~table.index.duplicated(keep="first")]
    table = table.infer_objects(copy=False)

    total_frames = max(1, int(math.ceil(duration * fps)))
    relative_video_times = np.arange(total_frames, dtype=float) / fps
    absolute_video_times = clip_start + relative_video_times
    telemetry_times = absolute_video_times + time_shift

    min_time = float(table.index.min())
    max_time = float(table.index.max())
    telemetry_times = np.clip(telemetry_times, min_time, max_time)
    index = pd.Index(telemetry_times, name="Time")

    # Resample to fixed FPS
    # We use 'index' interpolation to respect the time gaps
    # Ensure the union index is unique to avoid combinatorial explosion in reindex/loc
    union_index = table.index.union(index).unique().sort_values()
    
    interpolated = (
        table.reindex(union_index)
        .sort_index()
        .infer_objects(copy=False)
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
            .fillna(0)
            .astype(int)
        )
        aligned["LapNumber"] = lap_series

    return aligned
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


def calculate_heading(p1, p2):
    """Calculate heading between two points in degrees."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = np.arctan2(dy, dx)
    deg = np.degrees(rads)
    return deg


def normalize_angle(angle):
    """Normalize angle to -180 to 180."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def segment_track(polylines: list[list[tuple[float, float]]], turn_threshold: float = 0.8) -> list[dict]:
    """
    Split track into segments based on curvature.
    turn_threshold: degrees of heading change per point to consider a turn.
    """
    # Flatten polylines into a single list of points (assuming single loop)
    points = []
    for poly in polylines:
        points.extend(poly)
    
    if not points:
        return []

    # Convert to numpy for easier handling
    points_arr = np.array(points)
    
    # Calculate headings
    headings = []
    for i in range(len(points_arr) - 1):
        h = calculate_heading(points_arr[i], points_arr[i+1])
        headings.append(h)
    
    # Calculate curvature (change in heading)
    curvatures = []
    for i in range(len(headings) - 1):
        diff = normalize_angle(headings[i+1] - headings[i])
        curvatures.append(abs(diff))
    
    curvatures = [0] + curvatures + [0] 
    
    segments = []
    current_type = None # 'straight' or 'turn'
    current_points = []
    
    # Window size for smoothing (tuned value)
    window = 12
    
    for i in range(len(points_arr) - 1):
        # Simple smoothing
        start = max(0, i - window)
        end = min(len(curvatures), i + window + 1)
        avg_curv = np.mean(curvatures[start:end])
        
        segment_type = 'turn' if avg_curv > turn_threshold else 'straight'
        
        if segment_type != current_type:
            if current_points:
                segments.append({"type": current_type, "points": current_points})
            current_type = segment_type
            current_points = [points_arr[i]]
        else:
            current_points.append(points_arr[i])
            
    # Add last segment
    if current_points:
        current_points.append(points_arr[-1])
        segments.append({"type": current_type, "points": current_points})
    
    # Merge small segments
    min_points = 10 # Minimum points to be a valid segment
    
    if len(segments) > 1:
        cleaned_segments = []
        cleaned_segments.append(segments[0])
        
        for i in range(1, len(segments)):
            seg = segments[i]
            last = cleaned_segments[-1]
            
            if len(seg["points"]) < min_points:
                # Too small, merge into last
                last["points"].extend(seg["points"])
            else:
                # If type matches last (because we absorbed something), merge
                if seg["type"] == last["type"]:
                    last["points"].extend(seg["points"])
                else:
                    cleaned_segments.append(seg)
        
        segments = cleaned_segments

    return segments


@dataclass
class TrackLayout:
    polylines: list[list[tuple[float, float]]]
    bounds: tuple[float, float, float, float]
    segments: list[dict] | None = None


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
    
    # Calculate segments
    # Note: polylines here are in raw coordinates (likely Web Mercator for this track)
    # We should check if they are in meters before segmenting, but segment_track works on relative angles so scale doesn't matter much,
    # provided x/y are consistent.
    segments = segment_track(polylines, turn_threshold=0.8)
    
    return TrackLayout(polylines=polylines, bounds=bounds, segments=segments)


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
    start_finish_webmerc = None
    if start_finish:
        # Shapefile is already in WGS84 (lon, lat) based on .prj file
        start_finish_wgs84 = start_finish
        # Convert to Web Mercator for visualization
        lons, lats = zip(*start_finish)
        xs, ys = WGS84_TO_WEBMERC.transform(np.array(lons), np.array(lats))
        start_finish_webmerc = list(zip(xs, ys))
    return TrackGeometry(
        layout=layout,
        start_finish_webmerc=start_finish_webmerc,
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


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> tuple[bool, float]:
    """
    Check if line segment p1-p2 intersects with q1-q2.
    Returns (intersects, t) where t is the intersection parameter on p1-p2 (0..1).
    """
    px, py = p1
    rx, ry = p2[0] - px, p2[1] - py
    qx, qy = q1
    sx, sy = q2[0] - qx, q2[1] - qy

    cross = rx * sy - ry * sx
    if abs(cross) < 1e-9:
        return False, 0.0

    t = ((qx - px) * sy - (qy - py) * sx) / cross
    u = ((qx - px) * ry - (qy - py) * rx) / cross

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, t
    return False, 0.0


def calculate_laps(
    telemetry: pd.DataFrame,
    geometry: TrackGeometry,
) -> pd.DataFrame:
    """
    Calculate LapNumber and LapTime based on start-finish line crossings.
    """
    if geometry.start_finish_wgs84 is None or len(geometry.start_finish_wgs84) < 2:
        print("[overlay] No start-finish line defined; skipping lap calculation")
        return telemetry

    # Ensure we have GPS coordinates
    lat_col = pick_column(telemetry, ["GPS Latitude", "Latitude"])
    lon_col = pick_column(telemetry, ["GPS Longitude", "Longitude"])
    if not lat_col or not lon_col:
        return telemetry

    df = telemetry.copy()
    df = df.sort_index()
    
    # Create line segments from telemetry points (WGS84 coordinates)
    lats = pd.to_numeric(df[lat_col], errors="coerce").values
    lons = pd.to_numeric(df[lon_col], errors="coerce").values
    
    # Use Time column if available, otherwise index
    if "Time" in df.columns:
        times = pd.to_numeric(df["Time"], errors="coerce").values
    else:
        print("[overlay] Warning: 'Time' column not found, using index as time base")
        times = df.index.values
    
    # Start-finish line in WGS84 coordinates
    # Use first and last unique points to avoid degenerate line
    sf_line = geometry.start_finish_wgs84
    unique_sf_points = list(dict.fromkeys(sf_line))  # Remove duplicates while preserving order
    if len(unique_sf_points) < 2:
        print("[overlay] Warning: Start-finish line has less than 2 unique points")
        df["LapNumber"] = 0
        df["LapTime"] = times - times[0]
        return df
    
    sf_p1 = unique_sf_points[0]
    sf_p2 = unique_sf_points[-1]  # Use last unique point
    
    print(f"[overlay] Start-finish line: {sf_p1} to {sf_p2}")

    lap_number = 0
    lap_start_time = times[0]
    
    lap_numbers = np.zeros(len(df), dtype=int)
    lap_times = np.zeros(len(df), dtype=float)
    
    # Find all crossings of the start-finish line
    crossings = []

    for i in range(len(df) - 1):
        p1 = (lons[i], lats[i])
        p2 = (lons[i+1], lats[i+1])
        
        # Skip degenerate segments (same point)
        if p1 == p2:
            continue
        
        intersects, t = segments_intersect(p1, p2, sf_p1, sf_p2)
        
        if intersects:
            # Interpolate time of crossing
            t_cross = times[i] + t * (times[i+1] - times[i])
            crossings.append(t_cross)
    
    if len(crossings) == 0:
        print("[overlay] Warning: No start-finish line crossings detected")
        # Return telemetry with lap 0 (out lap) for all samples
        df["LapNumber"] = 0
        df["LapTime"] = times - times[0]
        return df

    print(f"[overlay] Detected {len(crossings)} lap crossing(s)")

    # Assign lap numbers and times
    current_lap = 0  # 0 = Out lap
    current_crossing_idx = 0
    last_crossing_time = times[0]
    
    
    for i in range(len(df)):
        t = times[i]
        
        # Check if we passed a crossing
        if current_crossing_idx < len(crossings) and t >= crossings[current_crossing_idx]:
            current_lap += 1
            last_crossing_time = crossings[current_crossing_idx]
            current_crossing_idx += 1
            
        lap_numbers[i] = current_lap
        lap_times[i] = t - last_crossing_time

    df["LapNumber"] = lap_numbers
    df["LapTime"] = lap_times
    
    # Calculate exact lap durations
    # Lap 0 duration = crossings[0] - times[0]
    # Lap N duration = crossings[N] - crossings[N-1]
    lap_durations = {}
    if crossings:
        # Lap 0
        lap_durations[0] = crossings[0] - times[0]
        
        # Subsequent laps
        for i in range(1, len(crossings)):
            lap_durations[i] = crossings[i] - crossings[i-1]
            
    return df, lap_durations, crossings


def calculate_lap_stats(df: pd.DataFrame, lap_durations: dict[int, float]) -> list[dict]:
    """
    Calculate statistics for each completed lap.
    Returns a list of dicts with: id, time, min_rpm, max_rpm, min_speed, max_speed
    """
    stats = []
    if "LapNumber" not in df.columns:
        return stats
        
    # Identify columns
    speed_col = pick_column(df, ["GPS Speed", "Speed", "Vitesse"])
    rpm_col = pick_column(df, ["RPM", "Régime"])
    
    # Group by lap
    laps = df["LapNumber"].unique()
    laps.sort()
    
    for lap_idx in laps:
        # if lap_idx == 0:
        #    continue # User wants Lap 0 (Outlap) included
            
        lap_data = df[df["LapNumber"] == lap_idx]
        if lap_data.empty:
            continue
            
        # Lap Time: use exact duration if available (for completed laps), else max of running time
        lap_id = int(lap_idx)
        if lap_id in lap_durations:
            lap_time = lap_durations[lap_id]
        else:
            # For the current/incomplete lap, we don't have a duration yet.
            # But calculate_lap_stats is usually for COMPLETED laps (or we want stats for current so far?)
            # If it's the last lap in the file and not completed, this gives running time so far.
            lap_time = lap_data["LapTime"].max()
        
        # Speed/RPM stats
        min_speed = 0.0
        max_speed = 0.0
        if speed_col:
            s = pd.to_numeric(lap_data[speed_col], errors="coerce")
            min_speed = s.min()
            max_speed = s.max()
            
        min_rpm = 0.0
        max_rpm = 0.0
        if rpm_col:
            r = pd.to_numeric(lap_data[rpm_col], errors="coerce")
            min_rpm = r.min()
            max_rpm = r.max()
            
        stats.append({
            "id": int(lap_idx),
            "time": lap_time,
            "min_speed": min_speed,
            "max_speed": max_speed,
            "min_rpm": min_rpm,
            "max_rpm": max_rpm,
        })
        
    return stats




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
    
    normalized_segments = []
    if layout.segments:
        for seg in layout.segments:
            # seg["points"] might be numpy arrays, normalize_polyline expects iterable of (x,y)
            # normalize_polyline iterates and unpacks x, y. Numpy array of shape (2,) unpacks fine.
            norm_points = normalize_polyline(seg["points"], layout.bounds)
            normalized_segments.append({
                "type": seg["type"],
                "points": norm_points
            })

    return TrackOverlay(
        normalized_lines=normalized_lines,
        positions=normalized_positions,
        start_finish_normalized=start_finish_norm,
        start_finish_wgs84=geometry.start_finish_wgs84,
        segments=normalized_segments,
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


def render_track_frame(
    index: int,
    video_time: float,
    *,
    venue: str,
    overlay_width: int,
    overlay_height: int,
    font_paths: dict[str, str],
    map_box: tuple[int, int, int, int],
    track_overlay_data: dict,
    output_dir: Path,
) -> int:
    """Render a single track frame.

    The ``track_overlay_data`` dictionary now contains a ``segment_stats`` entry
    mapping ``lap -> segment_index -> [min_speed, max_speed]``.  The current lap
    number is supplied via the ``current_lap`` key.  This function draws the
    segment colour and, when stats are available, annotates each segment with
    ``min/max`` speed values.
    """
    """Render a single track frame in a worker process."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    img = Image.new("RGBA", (overlay_width, overlay_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, overlay_width, overlay_height), radius=26, fill=(10, 14, 20, 200))

    # Load fonts inside the worker process
    title_font = ImageFont.truetype(font_paths["title"], 38) if font_paths.get("title") else ImageFont.load_default()
    meta_font = ImageFont.truetype(font_paths["meta"], 28) if font_paths.get("meta") else ImageFont.load_default()

    header = venue or "Track"
    draw.text((32, 24), f"{header} · Track Map", font=title_font, fill="#e8f6ff")

    # Reconstruct track overlay for drawing
    normalized_lines = track_overlay_data["normalized_lines"]
    positions = track_overlay_data["positions"]
    start_finish_normalized = track_overlay_data["start_finish_normalized"]
    
    # Draw track map
    x0, y0, x1, y1 = map_box
    draw.rounded_rectangle(map_box, radius=18, fill=(0, 0, 0, 140), outline="#214d66")
    pad = 18
    inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
    width = max(inner[2] - inner[0], 1)
    height = max(inner[3] - inner[1], 1) # Moved up for scope
    segments = track_overlay_data.get("segments")
    if segments:
        # Draw segments
        # Enumerate segments to know their index
        for seg_idx, seg in enumerate(segments):
            points = seg["points"]
            if len(points) < 2:
                continue
            
            scaled = [
                (
                    inner[0] + max(0.0, min(1.0, pt[0])) * width,
                    inner[1] + max(0.0, min(1.0, pt[1])) * height,
                )
                for pt in points
            ]
            
            # Color based on type
            color = "#4a90e2"  # Blue for straight
            if seg["type"] == "turn":
                color = "#e24a4a"  # Red for turn
            draw.line(scaled, fill=color, width=6)

            # Draw min/max speed for this segment if available for the current lap
            current_lap = track_overlay_data.get("current_lap")
            stats_by_lap = track_overlay_data.get("segment_stats", {})
            if current_lap is not None and seg["type"] == "straight":
                seg_stats = stats_by_lap.get(int(current_lap), {}).get(seg_idx)
                if seg_stats:
                    min_spd, max_spd = seg_stats
                    # Place the text near the middle of the segment
                    mid_idx = len(scaled) // 2
                    mid_pt = scaled[mid_idx]
                    txt = f"{int(min_spd)}/{int(max_spd)}"
                    small_font = ensure_font(16)
                    draw.text((mid_pt[0] + 5, mid_pt[1] - 12), txt, font=small_font, fill="#ffffff")
    else:
        # Fallback to drawing lines
        normalized_lines = track_overlay_data.get("normalized_lines", [])
        for line in normalized_lines:
            if len(line) < 2:
                continue
            scaled = [
                (
                    inner[0] + max(0.0, min(1.0, pt[0])) * width,
                    inner[1] + max(0.0, min(1.0, pt[1])) * height,
                )
                for pt in line
            ]
            draw.line(scaled, fill="#4a90e2", width=6)

    start_finish_normalized = track_overlay_data.get("start_finish_normalized")
    if start_finish_normalized and len(start_finish_normalized) >= 2:
        sf_points = [
            (
                inner[0] + max(0.0, min(1.0, pt[0])) * width,
                inner[1] + max(0.0, min(1.0, pt[1])) * height,
            )
            for pt in start_finish_normalized
        ]
        draw.line(sf_points, fill="#ffd479", width=6)

    if positions is not None and index < len(positions):
        pos = positions[index]
        if not np.isnan(pos).any():
            px = inner[0] + float(np.clip(pos[0], 0.0, 1.0)) * width
            py = inner[1] + float(np.clip(pos[1], 0.0, 1.0)) * height
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill="#ff7272", outline="white", width=2)

    minutes = int(video_time // 60)
    seconds = video_time % 60
    draw.text((32, overlay_height - 60), f"T+{minutes:02d}:{seconds:04.1f}", font=meta_font, fill="#9ad5ff")

    target = output_dir / f"{index:05d}.png"
    img.save(target)
    return index


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

    map_box = (60, 90, 60 + map_size, 90 + map_size)
    
    font_paths = {
        "title": get_font_path(bold=True),
        "meta": get_font_path(bold=False),
    }
    
    # Prepare track overlay data for serialization
    # Prepare track overlay data for serialization
    track_overlay_data = {
        "normalized_lines": track_overlay.normalized_lines,
        "positions": track_overlay.positions.tolist() if track_overlay.positions is not None else None,
        "start_finish_normalized": track_overlay.start_finish_normalized,
        "segments": track_overlay.segments,
    }

    # Compute per‑lap segment speed statistics (min / max) for the current lap
    # Determine which segment each telemetry sample belongs to by nearest point
    speed_col = pick_column(data, ["GPS Speed", "Speed", "Vitesse"])
    lap_col = "LapNumber"
    if speed_col is None or lap_col not in data.columns:
        segment_stats = {}
    else:
        positions_arr = np.array(track_overlay_data["positions"]) if track_overlay_data["positions"] is not None else None
        segments_pts = [np.array(seg["points"]) for seg in track_overlay.segments] if track_overlay.segments else []
        # Helper to find nearest segment index for a position
        def nearest_segment(pos):
            best_idx = None
            best_dist = float('inf')
            for idx, pts in enumerate(segments_pts):
                if pts.size == 0:
                    continue
                dists = np.linalg.norm(pts - pos, axis=1)
                min_d = dists.min()
                if min_d < best_dist:
                    best_dist = min_d
                    best_idx = idx
            return best_idx
        # Build nested dict: lap -> segment -> [min, max]
        segment_stats = {}
        for _, row in data.iterrows():
            lap = row.get(lap_col)
            speed = row.get(speed_col)
            if pd.isna(lap) or pd.isna(speed) or positions_arr is None:
                continue
            idx = int(row.name)
            pos = positions_arr[idx]
            seg_idx = nearest_segment(pos)
            if seg_idx is None:
                continue
            lap_dict = segment_stats.setdefault(int(lap), {})
            seg_dict = lap_dict.setdefault(seg_idx, [float('inf'), -float('inf')])
            seg_dict[0] = min(seg_dict[0], float(speed))
            seg_dict[1] = max(seg_dict[1], float(speed))
    # Attach stats to overlay data for rendering
    track_overlay_data["segment_stats"] = segment_stats
    
    venue = session.venue
    data = data.reset_index(drop=True)
    
    tasks = []
    for index, row in data.iterrows():
        video_time = row.get("VideoTime", 0.0)
        # Copy overlay data and inject current lap number
        overlay_data = dict(track_overlay_data)
        overlay_data["current_lap"] = row.get(lap_col)
        tasks.append({
            "index": index,
            "video_time": video_time,
            "venue": venue,
            "overlay_width": overlay_width,
            "overlay_height": overlay_height,
            "font_paths": font_paths,
            "map_box": map_box,
            "track_overlay_data": overlay_data,
            "output_dir": output_dir,
        })
    
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(render_track_frame, **task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering Track Frames"):
            future.result()  # Raise exception if any occurred

    return frames, (overlay_width, overlay_height)


def render_info_frame(
    index: int,
    video_time: float,
    *,
    driver: str,
    venue: str,
    session_name: str,
    event_date: str,
    overlay_width: int,
    overlay_height: int,
    font_paths: dict[str, str],
    heading: float | None,
    lap_number: int | None,
    lap_time: float | None,
    speed: float | None,
    rpm: float | None,
    max_speed: float,
    max_rpm: float,
    lat_g: float | None,
    lon_g: float | None,
    distance: float | None,
    lap_stats: list[dict],
    output_dir: Path,
    absolute_time: pd.Timestamp | None = None,
    predictive_model: PredictiveLapModel | None = None,
    projector: CenterlineProjector | None = None,
    current_pos: tuple[float, float] | None = None,
) -> int:
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new("RGBA", (overlay_width, overlay_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Helper to load fonts
    def load_font(key: str, size: int):
        path = font_paths.get(key)
        if path:
            return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    # --- 0. Predictive Delta Bar (Top Center) ---
    if predictive_model and projector and current_pos and lap_time is not None:
        # Calculate delta
        try:
            dist = projector.project(np.array(current_pos))
            predicted_time = predictive_model.get_time(dist)
            delta = lap_time - predicted_time
            
            # Draw Bar
            bar_w = 600
            bar_h = 30
            bar_x = (overlay_width - bar_w) // 2
            bar_y = 20
            
            # Background
            draw.rectangle(
                (bar_x, bar_y, bar_x + bar_w, bar_y + bar_h),
                fill=(10, 14, 20, 200),
                outline="#214d66",
                width=1
            )
            
            # Center marker
            center_x = bar_x + bar_w // 2
            draw.line([(center_x, bar_y), (center_x, bar_y + bar_h)], fill="white", width=2)
            
            # Delta Bar
            # Scale: +/- 2 seconds full width
            scale_sec = 2.0
            
            # Clamp delta
            clamped_delta = max(-scale_sec, min(scale_sec, delta))
            
            # Calculate width in pixels
            # 0 -> center_x
            # -2 -> bar_x
            # +2 -> bar_x + bar_w
            
            px_offset = (clamped_delta / scale_sec) * (bar_w / 2)
            
            if clamped_delta < 0:
                # Green (faster) - Left side
                # From center_x + px_offset to center_x
                # px_offset is negative
                rect = (center_x + px_offset, bar_y + 2, center_x, bar_y + bar_h - 2)
                color = "#00ff00" # Green
            else:
                # Red (slower) - Right side
                rect = (center_x, bar_y + 2, center_x + px_offset, bar_y + bar_h - 2)
                color = "#ff0000" # Red
                
            if abs(px_offset) > 1:
                draw.rectangle(rect, fill=color)
                
            # Text
            delta_str = f"{delta:+.2f}"
            text_font = load_font("medium", 24)
            bbox = draw.textbbox((0, 0), delta_str, font=text_font)
            tw = bbox[2] - bbox[0]
            
            # Position text below bar
            draw.text((center_x - tw // 2, bar_y + bar_h + 5), delta_str, font=text_font, fill=color)
            
        except Exception as e:
            # print(f"Delta error: {e}")
            pass

    # --- 1. Center Gauge (Speed & RPM) ---
    # Bottom Center
    gauge_width = 600
    gauge_height = 200
    gauge_x = (overlay_width - gauge_width) // 2
    gauge_y = overlay_height - gauge_height - 40
    
    # Background
    draw.rounded_rectangle(
        (gauge_x, gauge_y, gauge_x + gauge_width, gauge_y + gauge_height),
        radius=20,
        fill=(10, 14, 20, 200),
        outline="#214d66",
        width=2
    )
    
    # Speed
    speed_val = int(speed) if speed is not None else 0
    speed_text = f"{speed_val}"
    speed_font = load_font("large", 120)
    unit_font = load_font("small", 30)
    
    # Center speed text
    bbox = draw.textbbox((0, 0), speed_text, font=speed_font)
    text_w = bbox[2] - bbox[0]
    text_x = gauge_x + (gauge_width - text_w) // 2
    
    draw.text((text_x, gauge_y + 20), speed_text, font=speed_font, fill="white")
    draw.text((text_x + text_w + 10, gauge_y + 100), "km/h", font=unit_font, fill="#aaaaaa")
    
    # RPM Bar
    bar_x = gauge_x + 40
    bar_y = gauge_y + 140
    bar_w = gauge_width - 80
    bar_h = 30
    
    # Empty bar
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), fill=(50, 50, 50, 255))
    
    # Filled bar
    if rpm is not None and max_rpm > 0:
        pct = min(1.0, max(0.0, rpm / max_rpm))
        fill_w = bar_w * pct
        
        # Color based on RPM (Green -> Red)
        # Simple redline logic: if > 90% max_rpm, red, else blue/green
        bar_color = "#5ad2ff"
        if pct > 0.9:
            bar_color = "#ff3333"
        elif pct > 0.75:
            bar_color = "#ffd479"
            
        draw.rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), fill=bar_color)
        
        # RPM Text
        rpm_font = load_font("medium", 24)
        draw.text((bar_x, bar_y - 30), f"{int(rpm)} RPM", font=rpm_font, fill="white")

    # --- 2. Lap List (Top Right) ---
    list_w = 800
    list_h = 400
    list_x = overlay_width - list_w - 60
    list_y = 60
    
    draw.rounded_rectangle(
        (list_x, list_y, list_x + list_w, list_y + list_h),
        radius=20,
        fill=(10, 14, 20, 200),
        outline="#214d66",
        width=2
    )
    
    header_font = load_font("medium", 24)
    draw.text((list_x + 20, list_y + 20), "LAPS", font=header_font, fill="#e8f6ff")
    
    # Column Headers
    col_font = load_font("medium", 16)
    draw.text((list_x + 250, list_y + 25), "RPM (min/max)", font=col_font, fill="#aaaaaa")
    draw.text((list_x + 500, list_y + 25), "Speed (min/max)", font=col_font, fill="#aaaaaa")
    
    # --- Prepare Lap List Data (Static Table) ---
    # Show all laps from stats immediately.
    
    item_font = load_font("small", 20)
    highlight_font = load_font("medium", 20)
    small_font = load_font("small", 16)
    
    # Find best lap time
    best_lap_time = float("inf")
    for s in lap_stats:
        if s["time"] < best_lap_time:
            best_lap_time = s["time"]

    final_display_list = []
    for s in lap_stats:
        # Include all laps from stats
        # (lap_stats should include Lap 0 if we updated calculate_lap_stats)
        final_display_list.append({
            "id": s["id"],
            "time": format_duration(s["time"], 3),
            "status": "best" if s["time"] == best_lap_time else "done",
            "rpm": f"{int(s['min_rpm'])}/{int(s['max_rpm'])}",
            "speed": f"{int(s['min_speed'])}/{int(s['max_speed'])}",
        })
    
    # If current lap is not yet in lap_stats (i.e., it's still in progress)
    # and it's not Lap 0 (which is usually handled by stats)
    if lap_number is not None and lap_number > 0 and not any(s["id"] == lap_number for s in lap_stats):
        final_display_list.append({
            "id": lap_number,
            "time": format_duration(lap_time, 3) if lap_time is not None else "--:--",
            "status": "current",
            "rpm": "...",
            "speed": "...",
        })
        
    # Sort by lap ID to ensure correct display order
    final_display_list.sort(key=lambda x: x["id"])

    # Render list
    start_y = list_y + 60
    row_h = 35
    max_rows = (list_h - 80) // row_h
    
    # Scroll logic: keep current lap in view
    visible_laps = final_display_list
    if len(final_display_list) > max_rows:
        # Try to center current lap
        if lap_number is not None:
            # Find index of current lap in the list
            # Assuming list is sorted by ID and contiguous
            # But safe way is to find index
            target_idx = next((i for i, l in enumerate(final_display_list) if l["id"] == lap_number), 0)
            
            half_page = max_rows // 2
            start_idx = max(0, target_idx - half_page)
            end_idx = start_idx + max_rows
            if end_idx > len(final_display_list):
                end_idx = len(final_display_list)
                start_idx = max(0, end_idx - max_rows)
            visible_laps = final_display_list[start_idx:end_idx]
        else:
            visible_laps = final_display_list[:max_rows]
        
    for i, lap in enumerate(visible_laps):
        y = start_y + i * row_h
        
        # Highlight current lap
        is_current = (lap_number is not None and lap["id"] == lap_number)
        
        if is_current:
            draw.rectangle(
                (list_x + 10, y - 5, list_x + list_w - 10, y + 25),
                fill=(90, 210, 255, 50)
            )
            
        color = "#aaaaaa"
        font = item_font
        prefix = ""
        
        if is_current:
            color = "#ffffff"
            font = highlight_font
            prefix = "> "
        elif lap["status"] == "best":
            color = "#ffd479"
            
        # Lap ID
        draw.text((list_x + 20, y), f"{prefix}Lap {lap['id']}", font=font, fill=color)
        
        # Time
        draw.text((list_x + 120, y), lap["time"], font=font, fill=color)
        
        # RPM
        draw.text((list_x + 250, y), lap["rpm"], font=small_font, fill=color)
        
        # Speed
        draw.text((list_x + 500, y), lap["speed"], font=small_font, fill=color)

    # --- 3. Current Lap Counter (Top Left) ---
    if lap_number is not None:
        counter_x = 60
        counter_y = 60
        
        # Background for counter
        draw.rounded_rectangle(
            (counter_x, counter_y, counter_x + 200, counter_y + 100),
            radius=20,
            fill=(10, 14, 20, 200),
            outline="#214d66",
            width=2
        )
        
        draw.text((counter_x + 20, counter_y + 10), "CURRENT LAP", font=load_font("medium", 16), fill="#aaaaaa")
        draw.text((counter_x + 20, counter_y + 35), f"{lap_number}", font=load_font("large", 48), fill="white")

    # --- 4. Debug Info (Bottom Right) ---
    # Frame number and HH:MM:SS.mmm
    # Position above the built-in camera timestamp (usually bottom right)
    debug_x = overlay_width - 600
    debug_y = overlay_height - 100
    
    # Format time as HH:MM:SS.mmm
    total_seconds = video_time
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    ms = int((total_seconds - int(total_seconds)) * 1000)
    
    time_str = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    
    # Absolute Time
    abs_time_str = ""
    if absolute_time is not None:
        # absolute_time is a Timestamp or datetime
        try:
            abs_time_str = absolute_time.strftime("%H:%M:%S.%f")[:-3]
        except:
            pass
            
    debug_text = f"Frame: {index}  Video: {time_str}"
    if abs_time_str:
        debug_text += f"\nUTC: {abs_time_str}"
    
    draw.text((debug_x, debug_y), debug_text, font=load_font("medium", 24), fill="yellow", stroke_width=2, stroke_fill="black")

    target = output_dir / f"{index:05d}.png"
    img.save(target)
    return index


def get_font_path(bold: bool = False) -> str | None:
    """Return path to a readable font or None."""
    candidates = [
        REPO_ROOT / "render" / "fonts" / ("Inter-SemiBold.ttf" if bold else "Inter-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/SFNSDisplay.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def render_info_frames(
    data: pd.DataFrame,
    *,
    session,
    video: VideoInfo,
    lap_stats: list[dict],
    output_dir: Path,
    predictive_model: PredictiveLapModel | None = None,
    projector: CenterlineProjector | None = None,
) -> tuple[int, tuple[int, int]]:
    """Render semi-transparent overlay frames for ffmpeg."""
    data = data.reset_index(drop=True).copy()
    # Full screen overlay for info frames (to handle top-right lap list and bottom-center gauges)
    overlay_width = video.width
    overlay_height = video.height
    output_dir.mkdir(parents=True, exist_ok=True)

    font_paths = {
        "large": get_font_path(bold=True),
        "medium": get_font_path(bold=True),
        "small": get_font_path(bold=False),
    }

    # Extract series
    speed_col = pick_column(data, ["GPS Speed", "Speed", "Vitesse"])
    rpm_col = pick_column(data, ["RPM", "Régime"])
    lat_g_col = pick_column(data, ["GPS LatAcc", "LatAcc", "Lateral Acceleration"])
    lon_g_col = pick_column(data, ["GPS LonAcc", "LonAcc", "Longitudinal Acceleration"])
    
    # GPS for predictive delta
    lat_col = pick_column(data, ["GPS Latitude", "Latitude"])
    lon_col = pick_column(data, ["GPS Longitude", "Longitude"])
    
    # Ensure numeric
    def numeric_series(column: str | None):
        if not column or column not in data.columns:
            return None
        return pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    speeds = numeric_series(speed_col)
    rpms = numeric_series(rpm_col)
    lat_gs = numeric_series(lat_g_col)
    lon_gs = numeric_series(lon_g_col)
    
    lats = numeric_series(lat_col)
    lons = numeric_series(lon_col)
    
    # Pre-calculate positions in WebMercator if we have projector
    positions_webmerc = None
    if projector and lats is not None and lons is not None:
        # Interpolate to fill gaps
        lats = lats.replace(0.0, np.nan).interpolate().ffill().bfill()
        lons = lons.replace(0.0, np.nan).interpolate().ffill().bfill()
        
        xs, ys = WGS84_TO_WEBMERC.transform(lons.to_numpy(), lats.to_numpy())
        positions_webmerc = np.column_stack([xs, ys])
    
    # Pre-calculate max values for gauges
    max_speed = speeds.max() if speeds is not None else 100.0
    max_rpm = rpms.max() if rpms is not None else 10000.0
    
    # Distance for odometer (cumulative sum of speed * time_delta)
    # Approximate distance
    distance = None
    if speeds is not None:
        # simple integration: speed (km/h) * (1/fps) / 3600 -> km
        # This is rough, but okay for display
        dist_inc = speeds / video.fps / 3600.0
        distance = dist_inc.cumsum()

    tasks = []
    venue = session.venue
    driver = session.driver
    session_name = session.session
    event_date = session.event_date

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = []
        for index, row in data.iterrows():
            video_time = row.get("VideoTime", 0.0)
            
            # Get current values
            spd = float(speeds.iloc[index]) if speeds is not None else None
            r_pm = float(rpms.iloc[index]) if rpms is not None else None
            lat = float(lat_gs.iloc[index]) if lat_gs is not None else None
            lon = float(lon_gs.iloc[index]) if lon_gs is not None else None
            dist = float(distance.iloc[index]) if distance is not None else None
            
            current_pos = None
            if positions_webmerc is not None:
                current_pos = tuple(positions_webmerc[index])
            
            # Lap info
            lap_num = int(row.get("LapNumber", 0))
            lap_t = float(row.get("LapTime", 0.0))
            
            absolute_time = row.get("AbsoluteTime")
            if pd.isna(absolute_time):
                absolute_time = None
                
            futures.append(executor.submit(
                render_info_frame,
                index=index,
                video_time=video_time, # Use the extracted video_time
                driver=driver, # Use the pre-extracted driver
                venue=venue, # Use the pre-extracted venue
                session_name=session_name, # Use the pre-extracted session_name
                event_date=event_date, # Use the pre-extracted event_date
                overlay_width=overlay_width, # Use the pre-extracted overlay_width
                overlay_height=overlay_height, # Use the pre-extracted overlay_height
                font_paths=font_paths,
                heading=None, # Heading was explicitly removed
                lap_number=lap_num, # Use the extracted lap_num
                lap_time=lap_t, # Use the extracted lap_t
                speed=spd, # Use the extracted spd
                rpm=r_pm, # Use the extracted r_pm
                max_speed=max_speed,
                max_rpm=max_rpm,
                lat_g=lat, # Use the extracted lat
                lon_g=lon, # Use the extracted lon
                distance=dist, # Use the extracted dist
                lap_stats=lap_stats,
                output_dir=output_dir, # Use the passed output_dir
                absolute_time=absolute_time,
                predictive_model=predictive_model,
                projector=projector,
                current_pos=current_pos,
            ))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering Info Frames"):
            future.result()  # Raise exception if any occurred

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
    video_codec: str = "libaom-av1",
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
    parser.add_argument("--start", type=float, default=0.0, help="Clip start time (seconds).")
    parser.add_argument("--duration", type=float, default=None, help="Clip duration in seconds (default: full video).")
    parser.add_argument(
        "--time-shift",
        type=float,
        default=0.0,
        help="Telemetry time offset relative to the trimmed video (seconds).",
    )
    parser.add_argument("--sync-crossing-time", type=float, help="Video time (seconds) when the vehicle crosses the start/finish line.")
    parser.add_argument("--sync-lap", type=int, default=1, help="Which lap crossing corresponds to --sync-crossing-time (default: 1, i.e. end of out-lap).")
    parser.add_argument("--interactive-sync", action="store_true", help="Launch interactive tool to visually sync video and telemetry.")
    parser.add_argument("--utc-offset", type=float, default=4.0, help="UTC offset in hours (default 4.0 for GMT+4)")
    parser.add_argument("--telemetry-frequency", type=float, default=20.0, help="Samplerate hint for AIM sessions.")
    parser.add_argument(
        "--track-dir",
        type=Path,
        default=REPO_ROOT / "data" / "tracks" / "RIMSportKarting",
        help="Track shape directory (expects centerline shapefile).",
    )
    parser.add_argument("--keep-frames", action="store_true", help="Do not delete generated overlay PNG frames.")
    parser.add_argument("--hwaccel-cuda", action="store_true", help="Add '-hwaccel cuda' to the ffmpeg command.")
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libaom-av1",
        help="Video codec passed to ffmpeg (e.g. libaom-av1, av1_nvenc).",
    )
    return parser.parse_args()


def has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available via nvidia-smi."""
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    telemetry_dir = args.telemetry.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    # Auto-detect NVIDIA GPU if not explicitly requested
    use_cuda = args.hwaccel_cuda
    if not use_cuda and has_nvidia_gpu():
        print("[overlay] NVIDIA GPU detected: enabling CUDA hardware acceleration")
        use_cuda = True

    video_codec = args.video_codec
    if use_cuda and video_codec == "libaom-av1":
        print("[overlay] CUDA enabled: switching video codec to av1_nvenc")
        video_codec = "av1_nvenc"

    if not video_path.is_file():
        raise SystemExit(f"Video {video_path} is missing")
    if not telemetry_dir.is_dir():
        raise SystemExit(f"Telemetry folder {telemetry_dir} is missing")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for saved time shift
    sync_file = video_path.parent / f".{video_path.name}.timeshift"
    skip_sync = False
    if sync_file.exists():
        try:
            saved_shift = float(sync_file.read_text().strip())
            print(f"[overlay] Found saved synchronization time shift: {saved_shift}s")
            # We use sys.stdin to avoid issues if input is redirected, though input() usually handles it.
            # But since we might be running in an environment where we want to be careful:
            choice = input(f"[overlay] Use saved shift? [Y/n] ").strip().lower()
            if choice in ("", "y", "yes"):
                args.time_shift = saved_shift
                skip_sync = True
                print(f"[overlay] Using saved time shift: {args.time_shift}s")
            else:
                print("[overlay] Ignoring saved shift.")
        except ValueError:
            print("[overlay] Warning: Could not read saved time shift file.")

    video = probe_video(video_path)
    available = max(0.0, video.duration - args.start)
    if args.duration is None:
        duration = available
    else:
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

    # Check that video and log durations are similar (within 2 minutes)
    import pandas as pd
    log_times = pd.to_numeric(session.table["Time"], errors="coerce").dropna()
    if not log_times.empty:
        log_duration = log_times.max() - log_times.min()
        print(f"[overlay] Video Duration: {video.duration:.2f}s, Log Duration: {log_duration:.2f}s")
        assert abs(video.duration - log_duration) <= 1250, \
            f"Log duration ({log_duration:.2f}s) differs from video duration ({video.duration:.2f}s) by more than 2 minutes."
    
    # Calculate AbsoluteTime
    # session.event_date is YYYY-MM-DD
    # session.event_time is HH:MM or HH:MM:SS
    try:
        from datetime import datetime, timedelta
        import pandas as pd
        # Combine date and time
        dt_str = f"{session.event_date} {session.event_time}"
        # Try parsing with seconds or without
        try:
            start_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            start_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            
        # Apply UTC offset to get UTC time if input is local
        # User said video is GMT+4. Telemetry is likely local too?
        # If we want to display UTC, we subtract offset.
        # If we want to display Local, we keep it.
        # User said "can convert it to utc".
        # Let's store UTC timestamp.
        start_dt_utc = start_dt - timedelta(hours=args.utc_offset)
        
        print(f"[overlay] Session Start (Local): {start_dt}")
        print(f"[overlay] Session Start (UTC):   {start_dt_utc}")
        
        # Add AbsoluteTime column (seconds from epoch or datetime objects?)
        # Dataframes handle datetime objects well.
        # Time column in session.table is seconds from start.
        
        # We need to ensure Time is numeric
        session.table["Time"] = pd.to_numeric(session.table["Time"], errors="coerce")
        
        # Create AbsoluteTime series
        # session.table["AbsoluteTime"] = start_dt_utc + pd.to_timedelta(session.table["Time"], unit="s")
        # Actually, let's just keep it as datetime objects in a column
        
        # We'll do this AFTER calculate_laps so we don't mess up anything, 
        # but calculate_laps doesn't care about extra columns.
        
        # Vectorized addition
        start_ts = pd.Timestamp(start_dt_utc)
        session.table["AbsoluteTime"] = start_ts + pd.to_timedelta(session.table["Time"], unit="s")
        
        # Convert to numeric (nanoseconds) for interpolation
        session.table["AbsoluteTime"] = session.table["AbsoluteTime"].astype("int64")
        
    except Exception as e:
        print(f"[overlay] Warning: Could not calculate absolute time: {e}")
        # session.table["AbsoluteTime"] = pd.NaT # Can't mix types easily if we want int64

    print("[overlay] Calculating laps...")
    session.table, lap_durations, crossings = calculate_laps(session.table, track_geometry)
    
    # Interactive Sync
    if args.interactive_sync and not skip_sync:
        if not crossings:
            print("[overlay] Warning: No lap crossings detected. Cannot run interactive sync.")
        else:
            print("[overlay] Launching Interactive Sync...")
            try:
                from sync_ui import run_interactive_sync
                # We need to pass the full video path
                calculated_shift = run_interactive_sync(
                    video_path, 
                    crossings, 
                    fps=video.fps, 
                    duration=duration
                )
                if calculated_shift is not None:
                    print(f"[overlay] Interactive Sync applied. Shift: {calculated_shift:.4f}s")
                    args.time_shift = calculated_shift
                    try:
                        sync_file.write_text(str(calculated_shift))
                        print(f"[overlay] Saved time shift to {sync_file}")
                    except Exception as e:
                        print(f"[overlay] Warning: Could not save time shift: {e}")
                else:
                    print("[overlay] Interactive Sync cancelled. Using default/arg shift.")
            except ImportError:
                print("[overlay] Error: Could not import sync_ui. Make sure opencv-python is installed.")
            except Exception as e:
                print(f"[overlay] Error running interactive sync: {e}")

    # Calculate time_shift from sync arguments if provided (and not overridden by interactive)
    if args.sync_crossing_time is not None and not args.interactive_sync and not skip_sync:
        if not crossings:
            print("[overlay] Warning: No lap crossings detected in telemetry. Cannot sync.")
        else:
            # Lap 1 crossing is at index 0 (end of out-lap)
            # Lap N crossing is at index N-1
            crossing_idx = args.sync_lap - 1
            if 0 <= crossing_idx < len(crossings):
                telemetry_crossing = crossings[crossing_idx]
                # Formula: TelemetryTime = VideoTime + Shift
                # Shift = TelemetryTime - VideoTime
                calculated_shift = telemetry_crossing - args.sync_crossing_time
                print(f"[overlay] Syncing Lap {args.sync_lap} crossing.")
                print(f"[overlay] Telemetry Crossing: {telemetry_crossing:.3f}s")
                print(f"[overlay] Video Crossing:     {args.sync_crossing_time:.3f}s")
                print(f"[overlay] Calculated Shift:   {calculated_shift:.3f}s")
                
                # Override the command line time_shift (or add to it? usually override)
                args.time_shift = calculated_shift
                try:
                    sync_file.write_text(str(calculated_shift))
                    print(f"[overlay] Saved time shift to {sync_file}")
                except Exception as e:
                    print(f"[overlay] Warning: Could not save time shift: {e}")
            else:
                print(f"[overlay] Warning: Lap {args.sync_lap} crossing not found (total {len(crossings)} crossings).")

    # --- TRIM VIDEO TO TELEMETRY DURATION ---
    # Now that we have the final time_shift, we can calculate the video range
    # that corresponds to the telemetry log.
    # Telemetry Time = Video Time + Time Shift
    # Video Time = Telemetry Time - Time Shift
    
    # Get telemetry start/end
    # We use the original session table before resampling
    if "Time" in session.table.columns:
        # Ensure numeric
        t_values = pd.to_numeric(session.table["Time"], errors="coerce").dropna()
        if not t_values.empty:
            t_min = t_values.min()
            t_max = t_values.max()
            
            # Calculate corresponding video times
            video_start_needed = t_min - args.time_shift
            video_end_needed = t_max - args.time_shift
            
            print(f"[overlay] Telemetry Range: {t_min:.2f}s to {t_max:.2f}s")
            print(f"[overlay] Mapped to Video: {video_start_needed:.2f}s to {video_end_needed:.2f}s")
            
            # Clamp to available video
            # We want to trim the output to ONLY the part covered by telemetry
            # But we also respect the user's --start/--duration if they were more restrictive?
            # The user request says: "export the trimmed interval - start and end determined by start and end of telemetry log"
            # So we should override or intersect. Let's intersect with the physical video limits.
            
            new_start = max(0.0, video_start_needed)
            new_end = min(video.duration, video_end_needed)
            
            if new_end > new_start:
                # Update args.start and duration
                # Note: args.start is used in resample_telemetry as 'clip_start'
                # and in run_ffmpeg as 'start'.
                
                # If the user manually specified --start, we might want to respect it if it's LATER than telemetry start?
                # But the request implies automatic trimming. Let's override, but maybe warn if it differs significantly.
                
                print(f"[overlay] Trimming output to match telemetry coverage: Start {new_start:.2f}s, End {new_end:.2f}s")
                args.start = new_start
                duration = new_end - new_start
                
                # Update available duration for safety
                available = max(0.0, video.duration - args.start)
                duration = min(duration, available)
            else:
                print("[overlay] Warning: Telemetry range is outside video duration (or invalid). Not trimming.")


    # Calculate stats
    lap_stats = calculate_lap_stats(session.table, lap_durations)
    print(f"[overlay] Calculated stats for {len(lap_stats)} laps")

    print("[overlay] Resampling telemetry...")
    samples = resample_telemetry(
        session,
        fps=video.fps,
        duration=duration,
        time_shift=args.time_shift,
        clip_start=args.start,
    )
    
    if "AbsoluteTime" in samples.columns:
        samples["AbsoluteTime"] = pd.to_datetime(samples["AbsoluteTime"], unit="ns")

    # 4. Render Overlay
    print("[overlay] Building track map...")
    track_overlay = build_track_overlay(track_geometry, samples)
    if track_overlay is None:
        print("[overlay] Warning: Track map generation failed.")
        # Create a dummy overlay to proceed with other gauges

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        if args.keep_frames:
            # If keeping frames, use a persistent directory
            tmp_path = args.output.parent / "overlay_frames"
            tmp_path.mkdir(exist_ok=True)
        
        # 1. Render Track Map Frames
        # Note: Track map is now in bottom left, but we still render it separately
        # We just need to adjust the overlay position in ffmpeg
        track_frames, track_size = render_track_frames(
            samples,
            session=session,
            video=video,
            track_overlay=track_overlay,
            output_dir=tmp_path / "track",
        )
        
        # 2. Render Info/Gauge Frames
        print("[overlay] Rendering gauge frames...")
        
        # Prepare Predictive Delta Model
        predictive_model = None
        projector = None
        
        if track_geometry and track_geometry.layout.polylines:
            try:
                # Setup Centerline Projector
                centerline_raw = track_geometry.layout.polylines[0]
                lons, lats = zip(*centerline_raw)
                
                # Check if already projected
                if max(abs(x) for x in lons) > 180 or max(abs(y) for y in lats) > 90:
                    centerline_webmerc = np.array(centerline_raw)
                else:
                    xs, ys = WGS84_TO_WEBMERC.transform(np.array(lons), np.array(lats))
                    centerline_webmerc = np.column_stack((xs, ys))
                
                # Filter NaNs/Infs
                if not np.isfinite(centerline_webmerc).all():
                    centerline_webmerc = centerline_webmerc[np.isfinite(centerline_webmerc).all(axis=1)]
                
                if len(centerline_webmerc) >= 2:
                    projector = CenterlineProjector(centerline_webmerc)
                    
                    # Find Best Lap
                    valid_stats = [s for s in lap_stats if s["time"] > 20.0]
                    if valid_stats:
                        best_lap = min(valid_stats, key=lambda x: x["time"])
                        best_lap_id = best_lap["id"]
                        print(f"[overlay] Best Lap for Predictive Delta: {best_lap_id} ({best_lap['time']:.3f}s)")
                        
                        # Extract Best Lap Data from full session table
                        best_lap_df = session.table[session.table["LapNumber"] == best_lap_id].copy()
                        if not best_lap_df.empty:
                            # Ensure numeric
                            for col in ["GPS Latitude", "GPS Longitude", "LapTime"]:
                                 best_lap_df[col] = pd.to_numeric(best_lap_df[col], errors="coerce")
                            
                            best_lap_df = best_lap_df.dropna(subset=["GPS Latitude", "GPS Longitude", "LapTime"])
                            
                            if not best_lap_df.empty:
                                best_lons = best_lap_df["GPS Longitude"].values
                                best_lats = best_lap_df["GPS Latitude"].values
                                best_xs, best_ys = WGS84_TO_WEBMERC.transform(best_lons, best_lats)
                                best_points = np.column_stack((best_xs, best_ys))
                                best_times = best_lap_df["LapTime"].values
                                
                                # Build Map
                                dist_time_map = []
                                for pt, t in zip(best_points, best_times):
                                    d = projector.project(pt)
                                    dist_time_map.append((d, t))
                                    
                                predictive_model = PredictiveLapModel(dist_time_map)
                                print("[overlay] Predictive Lap Model built.")
            except Exception as e:
                print(f"[overlay] Failed to build predictive model: {e}")

        info_frames, info_size = render_info_frames(
            samples,
            session=session,
            video=video,
            lap_stats=lap_stats, # Pass lap stats
            output_dir=tmp_path / "info",
            predictive_model=predictive_model,
            projector=projector,
        )

        # 3. Compose with ffmpeg
        print("[overlay] Composing video...")
        overlay_specs = []
        if track_frames > 0:
            # Track map at bottom left
            # Map size is approx video.width * 0.28
            # We want it at (60, height - map_size - 60)
            # But render_track_frames returns frames of size (map_size + 120, map_size + 140)
            # And the content is drawn at (60, 90) inside that frame.
            # Let's just position the whole frame at bottom left.
            # The frame height is track_size[1].
            # y = video.height - track_size[1]
            map_y = video.height - track_size[1]
            overlay_specs.append({
                "pattern": str(tmp_path / "track" / "%05d.png"),
                "x": "0", # Left aligned (frame has padding)
                "y": str(map_y),
            })
        
        overlay_specs.append({
            "pattern": str(tmp_path / "info" / "%05d.png"),
            "x": "0",
            "y": "0",
        })

        run_ffmpeg(
            video_path=args.video,
            overlay_specs=overlay_specs,
            fps=video.fps,
            start=args.start,
            duration=video.duration if args.duration is None else args.duration,
            output_path=args.output,
            hwaccel_cuda=use_cuda,
            video_codec=video_codec,
        )

    if not args.keep_frames:
        pass # Temp dir is auto-cleaned


if __name__ == "__main__":
    main()
