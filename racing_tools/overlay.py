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
import cv2
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
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

CONVERTER_SRC = REPO_ROOT / "converter" / "converter"
if str(CONVERTER_SRC) not in sys.path:
    sys.path.append(str(CONVERTER_SRC))

import racing_tools.stab as stab

from session.session import Session  # type: ignore
from session.session import VideoSession
from session.video_info import VideoInfo, probe_video
from track.models import Track, TrackGeometry, TrackLayout, WGS84_TO_WEBMERC, normalize_angle






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





def load_session(folder: Path, frequency: float, normalize: bool = True):
    """Detect telemetry type and return a normalized session object."""
    return Session.load(folder, frequency=frequency, normalize=normalize)


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


def detect_crossings(
    df: pd.DataFrame,
    geometry: TrackGeometry,
) -> list[float]:
    """
    Detect start-finish line crossings in telemetry data.
    
    Returns:
        List of crossing times in seconds.
    """
    if geometry.start_finish_wgs84 is None or len(geometry.start_finish_wgs84) < 2:
        return []

    lat_col = pick_column(df, ["GPS Latitude", "Latitude"])
    lon_col = pick_column(df, ["GPS Longitude", "Longitude"])
    if not lat_col or not lon_col:
        return []

    lats = pd.to_numeric(df[lat_col], errors="coerce").values
    lons = pd.to_numeric(df[lon_col], errors="coerce").values
    times = pd.to_numeric(df["Time"], errors="coerce").values if "Time" in df.columns else df.index.values

    sf_points = list(dict.fromkeys(geometry.start_finish_wgs84))
    if len(sf_points) < 2:
        return []
    
    sf_p1, sf_p2 = sf_points[0], sf_points[-1]

    crossings = []
    for i in range(len(df) - 1):
        p1, p2 = (lons[i], lats[i]), (lons[i+1], lats[i+1])
        if p1 == p2:
            continue
        intersects, t = segments_intersect(p1, p2, sf_p1, sf_p2)
        if intersects:
            crossings.append(times[i] + t * (times[i+1] - times[i]))

    return crossings


def calculate_lap_durations(crossings: list[float], start_time: float = 0.0) -> dict[int, float]:
    """
    Calculate lap durations from crossing times.
    
    Returns:
        Dict mapping lap_id to duration in seconds.
    """
    if not crossings:
        return {}
    
    lap_durations = {0: crossings[0] - start_time}
    for i in range(1, len(crossings)):
        lap_durations[i] = crossings[i] - crossings[i-1]
    return lap_durations


def add_lap_numbers(df: pd.DataFrame, crossings: list[float]) -> pd.DataFrame:
    """
    Add LapNumber column to DataFrame based on crossing times.
    
    Lap 0 = before first crossing, Lap 1 = after first crossing, etc.
    """
    df = df.copy()
    times = pd.to_numeric(df["Time"], errors="coerce").values if "Time" in df.columns else df.index.values
    
    lap_numbers = np.zeros(len(df), dtype=int)
    crossing_idx = 0
    for i, t in enumerate(times):
        while crossing_idx < len(crossings) and t >= crossings[crossing_idx]:
            crossing_idx += 1
        lap_numbers[i] = crossing_idx

    df["LapNumber"] = lap_numbers
    return df


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


def select_best_lap(lap_stats: list[dict], min_lap_time: float = 20.0) -> dict | None:
    """
    Select the best lap using statistical filtering to reject outliers.
    """
    valid_stats = [s for s in lap_stats if s["time"] > min_lap_time]
    if not valid_stats:
        return None
        
    if len(valid_stats) < 3:
        # Not enough data for statistics, just take the minimum
        return min(valid_stats, key=lambda x: x["time"])

    # Statistical filtering
    times = np.array([s["time"] for s in valid_stats])
    median = np.median(times)
    
    # Calculate MAD (Median Absolute Deviation)
    mad = np.median(np.abs(times - median))
    
    # If MAD is very small (laps are identical), use standard deviation or just small epsilon
    if mad < 0.001:
        sigma = np.std(times)
        threshold = 3 * sigma if sigma > 0 else 0.1
    else:
        # Robust threshold: e.g., anything faster than Median - 3 * MAD might be an outlier
        # But for racing, a "magic lap" might be real. 
        # User requested: "Concentrated around the best time... maybe take median... group around it"
        # Let's say we trust laps within [Median - 2*MAD, Median + 2*MAD] as the "core group".
        # But we want the FASTEST valid lap.
        # So we want to reject laps that are suspiciously fast.
        
        # Criterion: A lap is valid if it is not *too far* below the median of the "good" laps.
        # Let's use a loose lower bound: Median - 5 * MAD is extremely unlikely unless it's a glitch/shortcut.
        # Standard deviation rule often uses 3 sigma. For MAD, 3*MAD is roughly 2*Sigma. 
        # So 5*MAD is quite generous.
        
        threshold = 6.0 * mad
        
    lower_bound = median - threshold
    
    # Debug info
    # print(f"[overlay] Lap Stats: Median={median:.3f}, MAD={mad:.3f}, LowerBound={lower_bound:.3f}")
    
    # Filter candidates
    candidates = [
        s for s in valid_stats 
        if s["time"] >= lower_bound
    ]
    
    if not candidates:
        # If all were rejected (unlikely usually), fall back to min of original valid
        return min(valid_stats, key=lambda x: x["time"])
        
    # Return the fastest of the statistically valid laps
    best_candidate = min(candidates, key=lambda x: x["time"])
    
    # Double check: if the "outlier" was actually close to this one (e.g. within 0.5s), maybe it was real?
    # But for now, trust the filter.
    
    # Print if we rejected something faster
    raw_best = min(valid_stats, key=lambda x: x["time"])
    if raw_best["id"] != best_candidate["id"]:
        print(f"[overlay] Statistical Filter: Rejected Lap {raw_best['id']} ({raw_best['time']:.3f}s) as outlier. "
              f"Selected Lap {best_candidate['id']} ({best_candidate['time']:.3f}s). "
              f"(Median={median:.2f}, Threshold={threshold:.2f})")
              
    return best_candidate


@dataclass
class TrackOverlay:
    normalized_lines: list[list[tuple[float, float]]]
    positions: np.ndarray | None
    start_finish_normalized: list[tuple[float, float]] | None
    start_finish_wgs84: list[tuple[float, float]] | None
    segments: list[dict] | None


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





def get_gradient_color(value: float, v_min: float, v_max: float) -> str:
    if v_max <= v_min:
        t = 0.5
    else:
        t = (value - v_min) / (v_max - v_min)
    t = max(0.0, min(1.0, t))
    
    # Gradient: Blue (#4a90e2) -> Green (#50e3c2) -> Red (#e24a4a)
    start_rgb = (74, 144, 226)   # Blue
    mid_rgb = (80, 227, 194)     # Green
    end_rgb = (226, 74, 74)      # Red
    
    if t < 0.5:
        local_t = t * 2.0
        r = int(start_rgb[0] + (mid_rgb[0] - start_rgb[0]) * local_t)
        g = int(start_rgb[1] + (mid_rgb[1] - start_rgb[1]) * local_t)
        b = int(start_rgb[2] + (mid_rgb[2] - start_rgb[2]) * local_t)
    else:
        local_t = (t - 0.5) * 2.0
        r = int(mid_rgb[0] + (end_rgb[0] - mid_rgb[0]) * local_t)
        g = int(mid_rgb[1] + (end_rgb[1] - mid_rgb[1]) * local_t)
        b = int(mid_rgb[2] + (end_rgb[2] - mid_rgb[2]) * local_t)
        
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_track_static(
    draw, # ImageDraw object
    map_box: tuple[int, int, int, int],
    track_overlay_data: dict,
) -> tuple[int, int, int, int]:
    """Draw static track map elements (background, lines, start/finish).
    
    Returns:
        tuple: (inner_x0, inner_y0, width, height) defining the drawing area
    """
    from PIL import ImageFont
    
    # Draw track map background
    x0, y0, x1, y1 = map_box
    draw.rounded_rectangle(map_box, radius=18, fill=(0, 0, 0, 140), outline="#214d66")
    
    pad = 18
    inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
    width = max(inner[2] - inner[0], 1)
    height = max(inner[3] - inner[1], 1)
    
    segments = track_overlay_data.get("segments")
    normalized_lines = track_overlay_data.get("normalized_lines", [])
    
    # Draw Track Lines (Standard / Fallback only)
    # The DYNAMIC colorful lines are now drawn in draw_dynamic_track_map
    if segments:
        for seg in segments:
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
            
            color = "#333333" # Dark grey for base track
            draw.line(scaled, fill=color, width=8)
    else:
        # Fallback lines
        for line in normalized_lines:
            if len(line) < 2: continue
            scaled = [
                (
                    inner[0] + max(0.0, min(1.0, pt[0])) * width,
                    inner[1] + max(0.0, min(1.0, pt[1])) * height,
                )
                for pt in line
            ]
            draw.line(scaled, fill="#333333", width=8)

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
        
    # Draw Legend (Static, Global)
    if "legend_min" in track_overlay_data:
         legend_min = track_overlay_data["legend_min"]
         legend_max = track_overlay_data["legend_max"]
         
         # Position: Bottom Left of the box area
         lh = 120
         lw = 10
         lx = x0 + 20
         ly = y1 - lh - 20
         
         # Background for legend
         # draw.rectangle((lx-5, ly-5, lx+lw+50, ly+lh+5), fill=(0,0,0,100))

         steps = 30
         for i in range(steps):
             step_y = ly + (lh * i / steps)
             step_h = (lh / steps) + 1
             ratio = 1.0 - (i / steps) 
             
             draw_val = legend_min + ratio * (legend_max - legend_min)
             c = get_gradient_color(draw_val, legend_min, legend_max)
             draw.rectangle((lx, step_y, lx+lw, step_y+step_h), fill=c)
             
         fnt = ensure_font(12)
         draw.text((lx + lw + 5, ly - 5), f"{int(legend_max/1000)}k", fill="white", font=fnt)
         draw.text((lx + lw + 5, ly + lh - 5), f"{int(legend_min/1000)}k", fill="white", font=fnt)
         
         # Intermediate labels (5k, 7k)
         for val in [5000, 7000]:
             if legend_min < val < legend_max:
                 ratio = (val - legend_min) / (legend_max - legend_min)
                 # Top is ratio=1.0, Bottom is ratio=0.0
                 # y = ly + lh * (1 - ratio)
                 y_val = ly + lh * (1.0 - ratio)
                 draw.text((lx + lw + 5, y_val - 6), f"{int(val/1000)}k", fill="white", font=fnt)


    return (inner[0], inner[1], width, height)


def draw_dynamic_track_map_lines(
    draw, 
    drawing_area: tuple[int, int, int, int],
    colored_points: list,
    v_min: float,
    v_max: float,
) -> None:
    """Draws the colored track lines for a specific lap."""
    if not colored_points or len(colored_points) < 2: return
    
    inner_x, inner_y, width, height = drawing_area

    # Optimization: use wider lines
    line_width = 8
    
    for i in range(len(colored_points) - 1):
        pt1 = colored_points[i]
        pt2 = colored_points[i+1]
         # pt is (x, y, val)
         
        p1_x = inner_x + max(0.0, min(1.0, pt1[0])) * width
        p1_y = inner_y + max(0.0, min(1.0, pt1[1])) * height
        p2_x = inner_x + max(0.0, min(1.0, pt2[0])) * width
        p2_y = inner_y + max(0.0, min(1.0, pt2[1])) * height
        
        # Interpolate value for segment color (use avg or p1)
        val = (pt1[2] + pt2[2]) / 2.0
        color = get_gradient_color(val, v_min, v_max)
        
        draw.line([(p1_x, p1_y), (p2_x, p2_y)], fill=color, width=line_width)



def draw_track_stats(
    draw, # ImageDraw object
    drawing_area: tuple[int, int, int, int],
    track_overlay_data: dict,
    font_paths: dict[str, str],
) -> None:
    """Draw dynamic track statistics (e.g. min/max speeds on segments)."""
    from PIL import ImageFont

    def ensure_font(size, bold=False):
        key = "medium" if bold else "small"
        p = font_paths.get(key)
        if p: return ImageFont.truetype(p, size)
        return ImageFont.load_default()

    inner_x, inner_y, width, height = drawing_area
    segments = track_overlay_data.get("segments")
    
    if segments:
        current_lap = track_overlay_data.get("current_lap")
        stats_by_lap = track_overlay_data.get("segment_stats", {})
        
        for seg_idx, seg in enumerate(segments):
            if seg["type"] != "straight":
                continue

            points = seg["points"]
            scaled = [
                (
                    inner_x + max(0.0, min(1.0, pt[0])) * width,
                    inner_y + max(0.0, min(1.0, pt[1])) * height,
                )
                for pt in points
            ]

            seg_stats = stats_by_lap.get(int(current_lap), {}).get(seg_idx) if current_lap is not None else None
            if seg_stats:
                min_spd, max_spd = seg_stats
                # Place text
                mid_idx = len(scaled) // 2
                mid_pt = scaled[mid_idx]
                txt = f"{int(min_spd)}/{int(max_spd)}"
                fnt = ensure_font(16)
                # Text shadow/outline for readability?
                draw.text((mid_pt[0] + 5, mid_pt[1] - 12), txt, font=fnt, fill="#ffffff")


def draw_track_position(
    draw, # ImageDraw object
    drawing_area: tuple[int, int, int, int],
    track_overlay_data: dict,
    index: int,
) -> None:
    """Draw the current position dot on the track map."""
    import numpy as np
    
    inner_x, inner_y, width, height = drawing_area
    positions = track_overlay_data.get("positions")
    
    if positions is not None and index < len(positions):
        pos = positions[index]
        # pos is [x, y] in normalized coords (0..1)
        if pos is not None and not np.isnan(pos).any():
             px = inner_x + float(np.clip(pos[0], 0.0, 1.0)) * width
             py = inner_y + float(np.clip(pos[1], 0.0, 1.0)) * height
             draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill="#ff7272", outline="white", width=2)


def draw_track_map(
    draw, # ImageDraw object
    map_box: tuple[int, int, int, int],
    track_overlay_data: dict,
    index: int,
    font_paths: dict[str, str],
) -> None:
    """Draw track map onto an existing PIL ImageDraw context."""
    # 1. Draw static elements (track geometry)
    drawing_area = draw_track_static(draw, map_box, track_overlay_data)
    
    # 2. Draw statistics (speed on straights)
    draw_track_stats(draw, drawing_area, track_overlay_data, font_paths)
    
    # 3. Draw current position
    draw_track_position(draw, drawing_area, track_overlay_data, index)




def draw_predictive_delta_static(
    draw,
    overlay_width: int,
) -> tuple[int, int, int, int]:
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
    
    return (bar_x, bar_y, bar_w, bar_h)


def draw_predictive_delta_dynamic(
    draw,
    bar_rect: tuple[int, int, int, int] | None, # Accept None if static didn't run or dynamic called standalone? No, should be valid.
    load_font,
    predictive_model,
    projector,
    current_pos: tuple[float, float] | None,
    lap_time: float | None,
) -> None:
    if predictive_model and projector and current_pos and lap_time is not None and bar_rect:
        try:
            dist = projector.project(np.array(current_pos))
            predicted_time = predictive_model.get_time(dist)
            delta = lap_time - predicted_time
            
            bar_x, bar_y, bar_w, bar_h = bar_rect
            center_x = bar_x + bar_w // 2
            
            # Delta Bar
            scale_sec = 2.0
            clamped_delta = max(-scale_sec, min(scale_sec, delta))
            px_offset = (clamped_delta / scale_sec) * (bar_w / 2)
            
            if clamped_delta < 0:
                # Green (faster) - Left side
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
            pass


def draw_center_gauge_static(
    draw,
    overlay_width: int,
    overlay_height: int,
    load_font,
) -> tuple[int, int, int, int]:
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
    
    # RPM Bar Background
    bar_x = gauge_x + 40
    bar_y = gauge_y + 140
    bar_w = gauge_width - 80
    bar_h = 30
    
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), fill=(50, 50, 50, 255))
    
    return (gauge_x, gauge_y, gauge_width, gauge_height)


def draw_center_gauge_dynamic(
    draw,
    gauge_rect: tuple[int, int, int, int],
    load_font,
    speed: float | None,
    rpm: float | None,
    max_rpm: float,
) -> None:
    gauge_x, gauge_y, gauge_width, gauge_height = gauge_rect
    
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
    
    # Filled bar
    if rpm is not None and max_rpm > 0:
        pct = min(1.0, max(0.0, rpm / max_rpm))
        fill_w = bar_w * pct
        
        # Color based on RPM (Green -> Red)
        bar_color = "#5ad2ff"
        if pct > 0.9:
            bar_color = "#ff3333"
        elif pct > 0.75:
            bar_color = "#ffd479"
            
        draw.rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), fill=bar_color)
        
        # RPM Text
        rpm_font = load_font("medium", 24)
        draw.text((bar_x, bar_y - 30), f"{int(rpm)} RPM", font=rpm_font, fill="white")


def draw_static_laptime_table(
    draw,
    list_x: int,
    list_y: int,
    list_w: int,
    list_h: int,
    load_font,
    lap_stats: list[dict],
) -> None:
    # draw.rounded_rectangle(
    #     (list_x, list_y, list_x + list_w, list_y + list_h),
    #     radius=20,
    #     fill=(10, 14, 20, 200),
    #     outline="#214d66",
    #     width=2
    # )
    
    header_font = load_font("medium", 24)
    draw.text((list_x + 20, list_y + 20), "LAPS", font=header_font, fill="#e8f6ff")
    
    # Column Headers
    col_font = load_font("medium", 16)
    draw.text((list_x + 250, list_y + 25), "RPM (min/max)", font=col_font, fill="#aaaaaa")
    draw.text((list_x + 500, list_y + 25), "Speed (min/max)", font=col_font, fill="#aaaaaa")

    # Draw static rows (completed laps)
    item_font = load_font("small", 20)
    small_font = load_font("small", 16)
    color = "#aaaaaa"
    
    start_y = list_y + 60
    row_h = 35
    
    # Sort to ensure order matches
    sorted_stats = sorted(lap_stats, key=lambda x: x["id"])
    
    for i, s in enumerate(sorted_stats):
        y = start_y + i * row_h
        
        # Prepare text data
        time_str = format_duration(s["time"], 3)
        rpm_str = f"{int(s['min_rpm'])}/{int(s['max_rpm'])}"
        spd_str = f"{int(s['min_speed'])}/{int(s['max_speed'])}"
        
        draw.text((list_x + 20, y), f"Lap {s['id']}", font=item_font, fill=color)
        draw.text((list_x + 120, y), time_str, font=item_font, fill=color)
        draw.text((list_x + 250, y), rpm_str, font=small_font, fill=color)
        draw.text((list_x + 500, y), spd_str, font=small_font, fill=color)

        # Stop if next row would overflow
        if y + row_h + row_h > list_y + list_h:
            break


def draw_best_lap_pointer(
    draw,
    list_x: int,
    y: int,
    lap_data: dict, # standardized dict
    font,
    small_font,
) -> None:
    color = "#ffd479"
    
    # Redraw over the static text
    draw.text((list_x + 20, y), f"Lap {lap_data['id']}", font=font, fill=color)
    draw.text((list_x + 120, y), lap_data['time_str'], font=font, fill=color)
    draw.text((list_x + 250, y), lap_data['rpm_str'], font=small_font, fill=color)
    draw.text((list_x + 500, y), lap_data['spd_str'], font=small_font, fill=color)


def draw_current_lap_pointer(
    draw,
    list_x: int,
    list_w: int,
    y: int,
    lap_data: dict,
    font,
    small_font,
) -> None:
    draw.rectangle(
        (list_x + 10, y - 5, list_x + list_w - 10, y + 25),
        fill=(90, 210, 255, 50)
    )
    
    color = "#ffffff"
    prefix = "> "
    
    draw.text((list_x + 20, y), f"{prefix}Lap {lap_data['id']}", font=font, fill=color)
    draw.text((list_x + 120, y), lap_data['time_str'], font=font, fill=color)
    draw.text((list_x + 250, y), lap_data['rpm_str'], font=small_font, fill=color)
    draw.text((list_x + 500, y), lap_data['spd_str'], font=small_font, fill=color)


def draw_lap_list(
    draw,
    overlay_width: int,
    overlay_height: int,
    load_font,
    lap_stats: list[dict],
    lap_number: int | None,
    lap_time: float | None,
) -> None:
    list_w = 800
    list_y = 60
    # Dynamic height: use available vertical space minus margins (top 60 + bottom ~150)
    list_h = overlay_height - list_y - 150
    list_x = overlay_width - list_w - 60
    
    # 1. Draw Static Table (Background + Headers + Completed Laps)
    draw_static_laptime_table(draw, list_x, list_y, list_w, list_h, load_font, lap_stats)
    
    # --- Prepare for Dynamic Overlays ---
    item_font = load_font("small", 20)
    highlight_font = load_font("medium", 20)
    small_font = load_font("small", 16)
    
    # Determine Best Lap ID
    best_lap_id = -1
    best_lap_stats = select_best_lap(lap_stats, min_lap_time=20.0)
    if best_lap_stats:
        best_lap_id = best_lap_stats["id"]
    elif lap_stats:
        # Fallback
        valid_table_stats = [s for s in lap_stats if s["time"] > 20.0]
        if valid_table_stats:
             best_lap_id = min(valid_table_stats, key=lambda s: s["time"])["id"]

    start_y = list_y + 60
    row_h = 35

    # Re-construct list to find indices exactly as static did
    # We need to know where each lap is to draw pointers
    sorted_stats = sorted(lap_stats, key=lambda x: x["id"])
    
    # 2. Draw Overlays
    
    # Best Lap Pointer (if it's in the static list)
    # Find index of best lap
    for i, s in enumerate(sorted_stats):
        if s["id"] == best_lap_id:
            # Check if this is NOT the current lap (current lap takes precedence for styling usually, or we assume current != best from history yet)
            # If current lap is re-driving a past lap? No, lap_number increases.
            
            y = start_y + i * row_h
            
            # Data for pointer
            l_data = {
                "id": s["id"],
                "time_str": format_duration(s["time"], 3),
                "rpm_str": f"{int(s['min_rpm'])}/{int(s['max_rpm'])}",
                "spd_str": f"{int(s['min_speed'])}/{int(s['max_speed'])}",
            }
            
            # Only draw best pointer if it's NOT the current lap (which shouldn't happen for past stats usually, unless we are reviewing)
            if lap_number != s["id"]:
                draw_best_lap_pointer(draw, list_x, y, l_data, item_font, small_font)
            break
            
    # Current Lap Pointer
    # If the current lap is "live" (not in stats yet, or is the last one being updated?)
    # Usually `lap_number` > max(stats.id).
    
    if lap_number is not None and lap_number > 0:
        # Is it in the static list?
        idx_in_static = next((i for i, s in enumerate(sorted_stats) if s["id"] == lap_number), -1)
        
        y = 0
        l_data = {}
        
        if idx_in_static != -1:
            # It's in the static list (maybe we are re-rendering or it was just added)
            s = sorted_stats[idx_in_static]
            y = start_y + idx_in_static * row_h
            l_data = {
                "id": s["id"],
                "time_str": format_duration(s["time"], 3),
                "rpm_str": f"{int(s['min_rpm'])}/{int(s['max_rpm'])}",
                "spd_str": f"{int(s['min_speed'])}/{int(s['max_speed'])}",
            }
        else:
             # It is a new lap, append to the end
             new_idx = len(sorted_stats)
             y = start_y + new_idx * row_h
             
             l_data = {
                "id": lap_number,
                "time_str": format_duration(lap_time, 3) if lap_time is not None else "--:--",
                "rpm_str": "...",
                "spd_str": "...",
             }
        
        draw_current_lap_pointer(draw, list_x, list_w, y, l_data, highlight_font, small_font)


def draw_current_lap_counter_static(
    draw,
    load_font,
) -> tuple[int, int]:
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
    return (counter_x, counter_y)


def draw_current_lap_counter_dynamic(
    draw,
    pos: tuple[int, int],
    load_font,
    lap_number: int | None,
) -> None:
    if lap_number is not None:
        counter_x, counter_y = pos
        draw.text((counter_x + 20, counter_y + 35), f"{lap_number}", font=load_font("large", 48), fill="white")


def draw_debug_info(
    draw,
    overlay_width: int,
    overlay_height: int,
    load_font,
    index: int,
    video_time: float,
    absolute_time: pd.Timestamp | None,
    projector,
    current_pos: tuple[float, float] | None,
) -> None:
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

    if projector and current_pos:
        try:
            dist_along = projector.project(np.array(current_pos))
            remaining = projector.total_length - dist_along
            debug_text += f"\nDist to Finish: {remaining:.1f} m"
        except Exception:
            pass
    
    draw.text((debug_x, debug_y), debug_text, font=load_font("medium", 24), fill="yellow", stroke_width=2, stroke_fill="black")

def generate_static_overlay(
    overlay_width: int,
    overlay_height: int,
    font_paths: dict,
    lap_stats: list[dict],
    track_overlay_data: dict | None,
) -> "Image.Image":
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new("RGBA", (overlay_width, overlay_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    print(f"[debug] generate_static_overlay called. lap_stats len: {len(lap_stats) if lap_stats else 'None'}")

    
    # Helper to load fonts locally
    def load_font(key: str, size: int):
        path = font_paths.get(key)
        if path:
            return ImageFont.truetype(path, size)
        return ImageFont.load_default()
        
    # 1. Predictive Delta Background
    draw_predictive_delta_static(draw, overlay_width)
    
    # 2. Gauge Background
    draw_center_gauge_static(draw, overlay_width, overlay_height, load_font)
    
    # 3. Static Lap Table (Background + Headers + Completed Laps)
    # We need to calculate list dimensions as in draw_lap_list
    list_w = 800
    list_y = 60
    list_h = overlay_height - list_y - 300
    list_x = overlay_width - list_w - 60
    draw_static_laptime_table(draw, list_x, list_y, list_w, list_h, load_font, lap_stats)
    
    # 4. Current Lap Counter Background
    draw_current_lap_counter_static(draw, load_font)
    
    # 5. Track Map Background
    if track_overlay_data:
        map_size = min(int(overlay_width * 0.28), 420)
        map_h = map_size + 140
        map_y = overlay_height - map_h
        target_y = overlay_height - map_size - 60
        target_x = 60
        mb = (target_x, target_y, target_x + map_size, target_y + map_size)
        
        draw_track_static(draw, mb, track_overlay_data)
        
    return img


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
    output_dir: Path | None = None,
    absolute_time: pd.Timestamp | None = None,
    predictive_model: PredictiveLapModel | None = None,
    projector: CenterlineProjector | None = None,
    current_pos: tuple[float, float] | None = None,
    track_overlay_data: dict | None = None,
    static_overlay: "Image.Image" | None = None,
) -> bytes | int:
    from PIL import Image, ImageDraw, ImageFont
    
    # Use static overlay or create new
    if static_overlay:
        img = static_overlay.copy()
    else:
        # Fallback if no static overlay provided
        print("TROUBLE")
        img = generate_static_overlay(
            overlay_width, overlay_height, font_paths, lap_stats, track_overlay_data
        )
        
    draw = ImageDraw.Draw(img)

    # Helper to load fonts
    def load_font(key: str, size: int):
        path = font_paths.get(key)
        if path:
            return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    # --- 0. Predictive Delta Bar (Dynamic) ---
    # We need the rect coordinates. They are static, hardcoded in the static function.
    # Ideally should be shared constants or returned.
    # For now, we know them:
    bar_w = 600
    bar_h = 30
    bar_x = (overlay_width - bar_w) // 2
    bar_y = 20
    bar_rect = (bar_x, bar_y, bar_w, bar_h)
    draw_predictive_delta_dynamic(draw, bar_rect, load_font, predictive_model, projector, current_pos, lap_time)

    # --- 1. Center Gauge (Dynamic) ---
    gauge_width = 600
    gauge_height = 200
    gauge_x = (overlay_width - gauge_width) // 2
    gauge_y = overlay_height - gauge_height - 40
    gauge_rect = (gauge_x, gauge_y, gauge_width, gauge_height)
    draw_center_gauge_dynamic(draw, gauge_rect, load_font, speed, rpm, max_rpm)

    # --- 1.5 Dynamic Track Map (Colored) ---
    if track_overlay_data and lap_number is not None:
        # Check cache
        cache = track_overlay_data.get("track_map_cache")
        if cache is not None:
             track_img = cache.get(lap_number)
             
             if track_img is None:
                 # Generate and cache
                 laps_points = track_overlay_data.get("colored_track_points_by_lap", {})
                 points = laps_points.get(lap_number)
                 
                 # Debug: Check if we have points
                 # print(f"[debug] Frame {index}: Lap {lap_number}, Points found: {len(points) if points else 0}")
                 
                 # Prepare transparent image for track lines
                 map_size = min(int(overlay_width * 0.28), 420)
                 map_h = map_size + 140 
                 target_y = overlay_height - map_size - 60
                 target_x = 60
                 mb = (target_x, target_y, target_x + map_size, target_y + map_size)
                 
                 x0, y0, x1, y1 = mb
                 pad = 18
                 inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
                 width = max(inner[2] - inner[0], 1)
                 height = max(inner[3] - inner[1], 1)
                 drawing_area = (inner[0], inner[1], width, height)
                 
                 t_img = Image.new("RGBA", (overlay_width, overlay_height), (0,0,0,0))
                 t_draw = ImageDraw.Draw(t_img)
                 
                 legend_min = track_overlay_data.get("legend_min", 0)
                 legend_max = track_overlay_data.get("legend_max", 10000)
                 
                 if points:
                     # Draw dynamic colored lines
                     draw_dynamic_track_map_lines(t_draw, drawing_area, points, legend_min, legend_max)
                 
                 cache[lap_number] = t_img
                 track_img = t_img
             
             # Composite
             if track_img:
                  img.alpha_composite(track_img)

    # --- 2. Lap List (Dynamic Overlays only) --- 
    # If we are using static overlay, the static part is already on `img`.
    # So we should ONLY call the dynamic parts.
    # Let's inline the dynamic parts here or extract a `draw_lap_list_dynamic`.
    # To keep it clean, let's just implement the dynamic logic directly here or use a helper.
    # Actually, `draw_lap_list` is:
    #   1. Draw Static
    #   2. Draw Overlays
    # Let's extract 2 into `draw_lap_list_dynamic` in the same file later?
    # Or just copy the logic for now to avoid another refactor step?
    # Re-using logic is better.
    # I will do a quick refactor of `draw_lap_list` to `draw_lap_list_dynamic` in a subsequent step if needed,
    # but for now, I will manually orchestrate the dynamic parts like I did for others.
    
    # Dynamic Lap List Logic
    list_w = 800
    list_y = 60
    list_x = overlay_width - list_w - 60
    # Recalculate height same as static for clipping
    list_h = overlay_height - list_y - 300
    
    start_y = list_y + 60
    row_h = 35
    
    item_font = load_font("small", 20)
    highlight_font = load_font("medium", 20)
    small_font = load_font("small", 16)
    
    # [Logic from draw_lap_list]
    best_lap_id = -1
    best_lap_stats = select_best_lap(lap_stats, min_lap_time=20.0)
    if best_lap_stats:
        best_lap_id = best_lap_stats["id"]
    elif lap_stats:
        valid_table_stats = [s for s in lap_stats if s["time"] > 20.0]
        if valid_table_stats:
             best_lap_id = min(valid_table_stats, key=lambda s: s["time"])["id"]

    sorted_stats = sorted(lap_stats, key=lambda x: x["id"])
    
    # Best Lap Pointer
    for i, s in enumerate(sorted_stats):
        if s["id"] == best_lap_id:
            y = start_y + i * row_h
            # Check bounds
            if y + row_h + row_h > list_y + list_h:
                break
                
            l_data = {
                "id": s["id"],
                "time_str": format_duration(s["time"], 3),
                "rpm_str": f"{int(s['min_rpm'])}/{int(s['max_rpm'])}",
                "spd_str": f"{int(s['min_speed'])}/{int(s['max_speed'])}",
            }
            if lap_number != s["id"]:
                draw_best_lap_pointer(draw, list_x, y, l_data, item_font, small_font)
            break
            
    # Current Lap Pointer
    if lap_number is not None and lap_number > 0:
        idx_in_static = next((i for i, s in enumerate(sorted_stats) if s["id"] == lap_number), -1)
        y = 0
        l_data = {}
        visible = True
        
        if idx_in_static != -1:
            s = sorted_stats[idx_in_static]
            y = start_y + idx_in_static * row_h
            l_data = {
                "id": s["id"],
                "time_str": format_duration(s["time"], 3),
                "rpm_str": f"{int(s['min_rpm'])}/{int(s['max_rpm'])}",
                "spd_str": f"{int(s['min_speed'])}/{int(s['max_speed'])}",
            }
        else:
             new_idx = len(sorted_stats)
             y = start_y + new_idx * row_h
             l_data = {
                "id": lap_number,
                "time_str": format_duration(lap_time, 3) if lap_time is not None else "--:--",
                "rpm_str": "...",
                "spd_str": "...",
             }
        
        # Check bounds
        if y + row_h + row_h > list_y + list_h:
            visible = False

        # draw_current_lap_pointer signature is: (draw, list_x, list_w, y, lap_data, font, small_font)
        if visible:
            draw_current_lap_pointer(draw, list_x, list_w, y, l_data, highlight_font, small_font)

    # --- 3. Current Lap Counter (Dynamic) ---
    counter_x = 60
    counter_y = 60
    draw_current_lap_counter_dynamic(draw, (counter_x, counter_y), load_font, lap_number)

    # --- 4. Debug Info (Bottom Right) ---
    draw_debug_info(draw, overlay_width, overlay_height, load_font, index, video_time, absolute_time, projector, current_pos)

    # --- 5. Track Map (Dynamic) ---
    if track_overlay_data:
        map_size = min(int(overlay_width * 0.28), 420)
        map_h = map_size + 140
        map_y = overlay_height - map_h
        target_y = overlay_height - map_size - 60
        target_x = 60
        mb = (target_x, target_y, target_x + map_size, target_y + map_size)
        
        # Static part already drawn. Use draw_track_stats (dynamic) and draw_track_position.
        # But draw_track_stats needs drawing_area. 
        # drawing_area comes from draw_track_static return value.
        # We need to calculate it again or return it from somewhere?
        # It is: (inner[0], inner[1], width, height)
        # inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
        # We can recalc it here cheaply.
        pad = 18
        x0, y0, x1, y1 = mb
        inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
        width = max(inner[2] - inner[0], 1)
        height = max(inner[3] - inner[1], 1)
        drawing_area = (inner[0], inner[1], width, height)
        
        draw_track_stats(draw, drawing_area, track_overlay_data, font_paths)
        draw_track_position(draw, drawing_area, track_overlay_data, index)
    

    if output_dir:
        target = output_dir / f"{index:05d}.png"
        img.save(target)
        return index
    else:
        # Return raw RGBA bytes
        return img.tobytes()


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
    stdin_generator: Iterator[bytes] | None = None,
    stab_filter: str | None = None,
) -> None:
    """Invoke ffmpeg to trim, overlay, and export.
    
    If stdin_generator is provided, it must yield bytes (frames) to be written to stdin.
    """
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
        if "start_number" in spec:
            cmd.extend(["-start_number", str(spec["start_number"])])
            
        # Handle rawvideo options if present
        if spec.get("format") == "rawvideo":
            cmd.extend([
                "-f", "rawvideo",
                "-pix_fmt", spec.get("pixel_format", "bgr24"),
                "-s", spec.get("size", "1920x1080"),
            ])
            
        cmd.extend([
            "-thread_queue_size", "1024",
            "-framerate",
            f"{fps:.5f}",
            "-i",
            spec["pattern"],
        ])

    # Construct Filter Chain
    lines: list[str] = []
    
    # 0:v is the video stream
    current_video_stream = "[0:v]"
    
    # Apply stabilization if requested (must be first)
    if stab_filter:
        # Force pixel format to yuv420p BEFORE vidstab to prevent "Assertion fi->planes==1 failed"
        # This occurs because libvidstab is picky about input formats.
        lines.append(f"{current_video_stream}format=yuv420p, {stab_filter}[vstable]")
        current_video_stream = "[vstable]"
    
    # Apply Overlays
    if overlay_specs:
        for idx, spec in enumerate(overlay_specs):
            overlay_stream = f"[{idx + 1}:v]"
            out_label = "[vout]" if idx == len(overlay_specs) - 1 else f"[tmp{idx}]"
            lines.append(f"{current_video_stream}{overlay_stream}overlay={spec['x']}:{spec['y']}:format=auto{out_label}")
            current_video_stream = out_label
    else:
        # No overlay specs? Just pass through (or stab only)
        # If we had stab, current_video_stream is [vstable]. Assign it to something we map?
        # Actually -map argument usually takes [label] if filter_complex is used.
        pass

    if lines:
        cmd.extend(["-filter_complex", ";".join(lines)])
        video_map = current_video_stream
    else:
        video_map = "0:v"

    # Codec Settings
    codec = video_codec.lower()
    quality_flags: list[str] = []
    
    if "nvenc" in codec:
        quality_flags = [
            "-preset", "p7",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "12",
            "-multipass", "2",
            "-maxrate", "50M",
            "-bufsize", "100M",
            "-rc-lookahead", "64",
            "-spatial-aq", "1",
            "-temporal-aq", "1",
            "-aq-strength", "15",
            "-g", "480",
            "-bf", "3",
            "-pix_fmt", "yuv420p"
        ]
        # Override preset argument below by NOT adding it if it's already in quality_flags?
        # run_ffmpeg adds "-preset medium" generally. We should change that logic.
        
    else:
        quality_flags = ["-crf", "14", "-preset", "medium"]

    cmd.extend([
        "-map",
        video_map,
        "-map",
        "0:a?",
        "-c:v",
        video_codec,
    ])
    
    # If nvenc, we already put preset in quality_flags, so don't double add if logic allows
    # But currently the code structure likely adds "-preset medium" unconditionally.
    # We need to replace the entire cmd.extend block for encoding settings.
    
    cmd.extend(quality_flags)
    
    cmd.extend([
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ])

    
    if stdin_generator:
        print(f"[overlay_ffmpeg] Starting ffmpeg with stdin pipe...")
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        try:
            assert process.stdin is not None
            for chunk in stdin_generator:
                 process.stdin.write(chunk)
            process.stdin.flush()
        except BrokenPipeError:
            print("[overlay_ffmpeg] FFMPEG input pipe broken (process likely died).")
        except KeyboardInterrupt:
            print("\n[overlay_ffmpeg] Interrupted by user! Stopping input generation...")
            print("[overlay_ffmpeg] Closing pipe to allow ffmpeg to finalize the video...")
            # The finally block will close stdin.
            # We then exit the script to stop further processing (like best lap export).
            sys.exit(0)
        except Exception as e:
             print(f"[overlay_ffmpeg] Error writing to pipe: {e}")
        finally:
            if process.stdin:
                process.stdin.close()
            process.wait()
            if process.returncode != 0:
                print(f"[overlay_ffmpeg] ffmpeg exited with code {process.returncode}")
                # raise subprocess.CalledProcessError(process.returncode, cmd)
    else:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a telemetry overlay demo clip.")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video file.")
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=None,
        help="Path to telemetry folder. If not provided, manual lap marking mode is used.",
    )
    parser.add_argument(
        "--laps-file",
        type=Path,
        default=None,
        help="Path to JSON file containing manual lap boundaries (for manual mode).",
    )
    parser.add_argument("--output", type=Path, default=None)
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
    parser.add_argument("--export-best-lap", action="store_true", default=True, help="Additionally export the best lap (+/- 10s) as a separate video (default).")
    
    # Stabilization args
    parser.add_argument("--stabilize", action="store_true", default=False, help="Apply video stabilization using ffmpeg vidstab.")
    parser.add_argument("--stab-smoothing", type=int, default=10, help="Stabilization smoothing window (default: 10).")
    parser.add_argument("--stab-zoom", type=int, default=0, help="Stabilization zoom percentage (default: 0).")
    parser.add_argument("--stab-optzoom", type=int, default=0, choices=[0,1,2], help="Stabilization optimal zoom behavior (default: 0).")
    
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
    
    # Sync file path
    sync_file = video_path.parent / f".sync-{video_path.name}.txt"
    
    
    # Deferred output path calculation
    # Only set if args.output was provided
    custom_output_path = None
    if args.output:
        custom_output_path = args.output.expanduser().resolve()


    track = None
    track_geometry = None
    crossings = []

    # Auto-detect NVIDIA GPU if not explicitly requested
    use_cuda = args.hwaccel_cuda
    if not use_cuda and has_nvidia_gpu():
        print("[overlay] NVIDIA GPU detected: enabling CUDA hardware acceleration")
        use_cuda = True

    video_codec = args.video_codec
    if use_cuda and video_codec == "libaom-av1":
        print("[overlay] CUDA enabled: switching video codec to av1_nvenc")
        video_codec = "av1_nvenc"
    
    # Define skip_sync based on args or logic (assuming interactive_sync overrides or similar)
    # The original error was 'and not skip_sync'. It seems I missed defining it.
    # Assuming intent: if explicitly set or if manual.
    skip_sync = False 

    if args.telemetry:
        telemetry_path = args.telemetry.expanduser().resolve()
        if not telemetry_path.exists():
            raise SystemExit(f"Telemetry path {telemetry_path} is missing")

        # Use VideoSession
        print(f"[overlay] Loading telemetry from {telemetry_path.name}")
        vsession = VideoSession(telemetry_path, video_path)
        
        # Load video metadata
        video = vsession.load_video_metadata()
        
        # Sync
        # Interactive sync logic
        if args.interactive_sync:
             # Force UI
             vsession.sync(interactive=True, force_ui=True)
        else:
             # Try load, else check manual args or ask
             if args.time_shift != 0.0:
                 # Override
                 vsession.sync_offset = args.time_shift
                 print(f"[overlay] Using command-line time shift: {args.time_shift:.4f}s")
             else:
                 # Check for saved
                 vsession.sync(interactive=not args.interactive_sync) 
                 pass

        if args.time_shift != 0.0:
             vsession.sync_offset = args.time_shift
        else:
             vsession.sync(interactive=False)
             if vsession.sync_offset != 0.0:
                 print(f"[overlay] Found saved sync offset: {vsession.sync_offset:.4f}s")
                 if not args.interactive_sync:
                     print(f"[overlay] Using saved time shift: {vsession.sync_offset:.4f}s")

        # Extract session back for rest of script
        session = vsession.session
        args.time_shift = vsession.sync_offset
        
        available = max(0.0, video.duration - args.start)
        if args.duration is None:
            duration = available
        else:
            duration = max(1.0, min(args.duration, available))
        import pandas as pd
        log_times = pd.to_numeric(session.table["Time"], errors="coerce").dropna()
        if not log_times.empty:
            log_duration = log_times.max() - log_times.min()
            print(f"[overlay] Video Duration: {video.duration:.2f}s, Log Duration: {log_duration:.2f}s")
            assert abs(video.duration - log_duration) <= 1250, \
                f"Log duration ({log_duration:.2f}s) differs from video duration ({video.duration:.2f}s) by more than 2 minutes."
        
        # Calculate AbsoluteTime
        try:
            from datetime import datetime, timedelta
            dt_str = f"{session.event_date} {session.event_time}"
            try:
                start_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                start_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
                
            start_dt_utc = start_dt - timedelta(hours=args.utc_offset)
            print(f"[overlay] Session Start (Local): {start_dt}")
            print(f"[overlay] Session Start (UTC):   {start_dt_utc}")
            
            session.table["Time"] = pd.to_numeric(session.table["Time"], errors="coerce")
            start_ts = pd.Timestamp(start_dt_utc)
            session.table["AbsoluteTime"] = start_ts + pd.to_timedelta(session.table["Time"], unit="s")
            session.table["AbsoluteTime"] = session.table["AbsoluteTime"].astype("int64")
            
        except Exception as e:
            print(f"[overlay] Warning: Could not calculate absolute time: {e}")

        print("[overlay] Calculating laps...")
        
        # Load track geometry if available
        if args.track_dir:
            track_dir = args.track_dir.expanduser().resolve()
            if track_dir.is_dir():
                print(f"[overlay] Loading track geometry from {track_dir.name}")
                try:
                    track = Track.load(track_dir)
                    track_geometry = track.geometry
                except Exception as e:
                    print(f"[overlay] Error loading track: {e}")
            else:
                print(f"[overlay] Warning: Track directory {track_dir} not found.")
        
        session.table, lap_durations, crossings = calculate_laps(session.table, track_geometry)
        
        # Interactive Sync
        if args.interactive_sync and not skip_sync:
            if not crossings:
                print("[overlay] Warning: No lap crossings detected. Cannot run interactive sync.")
            else:
                print("[overlay] Launching Interactive Sync...")
                try:
                    from sync_ui import run_interactive_sync
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

        # Prompt for interactive sync if no shift set and not explicitly requested
        if (
            not args.interactive_sync
            and args.time_shift == 0.0
            and args.sync_crossing_time is None
            and not skip_sync
            and crossings
        ):
            print("[overlay] Time delta not set.")
            try:
                choice = input("[overlay] Do you want to run interactive_sync? [y/N] ").strip().lower()
                if choice in ("y", "yes"):
                     # Re-run the interactive sync block logic
                     # Duplicate logic call or just set flag and loop? 
                     # Easier to just call the function here directly or restructure.
                     # Let's just run it inline here to avoid complex refactor
                    print("[overlay] Launching Interactive Sync...")
                    try:
                        from sync_ui import run_interactive_sync
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
                    except ImportError:
                        print("[overlay] Error: Could not import sync_ui.")
                    except Exception as e:
                        print(f"[overlay] Error running interactive sync: {e}")

            except EOFError:
                pass # Non-interactive


        # Calculate time_shift from sync arguments if provided
        if args.sync_crossing_time is not None and not args.interactive_sync and not skip_sync:
            if not crossings:
                print("[overlay] Warning: No lap crossings detected in telemetry. Cannot sync.")
            else:
                crossing_idx = args.sync_lap - 1
                if 0 <= crossing_idx < len(crossings):
                    telemetry_crossing = crossings[crossing_idx]
                    calculated_shift = telemetry_crossing - args.sync_crossing_time
                    print(f"[overlay] Syncing Lap {args.sync_lap} crossing.")
                    print(f"[overlay] Telemetry Crossing: {telemetry_crossing:.3f}s")
                    print(f"[overlay] Video Crossing:     {args.sync_crossing_time:.3f}s")
                    print(f"[overlay] Calculated Shift:   {calculated_shift:.3f}s")
                    
                    args.time_shift = calculated_shift
                    try:
                        sync_file.write_text(str(calculated_shift))
                        print(f"[overlay] Saved time shift to {sync_file}")
                    except Exception as e:
                        print(f"[overlay] Warning: Could not save time shift: {e}")
                else:
                    print(f"[overlay] Warning: Lap {args.sync_lap} crossing not found (total {len(crossings)} crossings).")

        # --- TRIM VIDEO TO TELEMETRY DURATION ---
        if "Time" in session.table.columns:
            t_values = pd.to_numeric(session.table["Time"], errors="coerce").dropna()
            if not t_values.empty:
                t_min = t_values.min()
                t_max = t_values.max()
                
                video_start_needed = t_min - args.time_shift
                video_end_needed = t_max - args.time_shift
                
                print(f"[overlay] Telemetry Range: {t_min:.2f}s to {t_max:.2f}s")
                print(f"[overlay] Mapped to Video: {video_start_needed:.2f}s to {video_end_needed:.2f}s")
                
                new_start = max(0.0, video_start_needed)
                new_end = min(video.duration, video_end_needed)
                
                if new_end > new_start:
                    print(f"[overlay] Trimming output to match telemetry coverage: Start {new_start:.2f}s, End {new_end:.2f}s")
                    args.start = new_start
                    duration = new_end - new_start
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

    else:
        # Manual Mode (No Telemetry)
        print("[overlay] No telemetry provided. Checking for Manual Lap Data...")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video = probe_video(video_path)
        
        laps_file = args.laps_file
        if laps_file is None:
             # Try default naming: .laps.json or .laps-VIDEO.json
             candidates = [
                 video_path.with_suffix(".laps.json"),
                 video_path.parent / f".laps-{video_path.name}.json"
             ]
             for c in candidates:
                 if c.exists():
                     laps_file = c
                     break
        
        boundaries = []
        if laps_file and laps_file.exists():
            try:
                content = laps_file.read_text().strip()
                boundaries = json.loads(content)
                if isinstance(boundaries, list):
                    print(f"[overlay] Loaded {len(boundaries)} manual laps from {laps_file}")
            except Exception as e:
                 raise SystemExit(f"Error reading laps file {laps_file}: {e}")
        else:
             print(f"[overlay] Error: No telemetry and no laps file found.")
             print("To use manual mode, please run 'racing_tools/mark_laps.py' first or provide --laps-file.")
             raise SystemExit(1)

        if not boundaries:
             raise SystemExit("Laps file is empty or invalid.")
            
        print(f"[overlay] Marked {len(boundaries)} lap boundaries.")
        
        # Create synthetic session
        from types import SimpleNamespace
        session = SimpleNamespace()
        session.venue = "Unknown Venue"
        session.driver = "Unknown Driver"
        session.session = "Manual Session"
        session.event_date = "Unknown Date"
        session.event_time = "00:00"
        
        # Create synthetic samples DataFrame (Manual Mode)
        # We need a proper index for iteration
        import pandas as pd
        import numpy as np
        
        # Create minimal samples df
        # Duration based
        total_frames = int(video.duration * video.fps)
        times = np.linspace(0, video.duration, total_frames)
        samples = pd.DataFrame({"VideoTime": times})
        samples["LapNumber"] = 0
        samples["LapTime"] = 0.0
        
        # Fill Lap Numbers
        current_lap = 0
        current_lap_start = 0.0
        
        boundaries_sorted = sorted(boundaries)
        
        # This is slow, use vectorized if possible
        # Or just loop once
        lap_arr = np.zeros(total_frames, dtype=int)
        lap_time_arr = np.zeros(total_frames, dtype=float)
        
        bs = [0.0] + boundaries_sorted + [video.duration + 1.0]
        
        for i in range(len(bs)-1):
            t0 = bs[i]
            t1 = bs[i+1]
            mask = (times >= t0) & (times < t1)
            lap_arr[mask] = i # Lap 0 is out lap, or Lap 1?
            # User wants Lap 1 to be first valid lap usually.
            # boundaries marked crossings.
            # First segment is Lap 1? Or Out Lap?
            # Usually strict lap marking implies valid laps.
            # Let's say segment 0 is Lap 1.
            lap_arr[mask] = i + 1
            lap_time_arr[mask] = times[mask] - t0
            
        # For Manual Mode, treat boundaries as crossings
        # In Manual Mode, Telemetry Time == Video Time (args.time_shift = 0 typically)
        # boundaries[0] is start of Lap 1 / end of Lap 0.
        crossings = boundaries

        samples["LapNumber"] = lap_arr
        samples["LapTime"] = lap_time_arr

        # Minimal stats for manual
        lap_stats = []
        for i in range(len(boundaries_sorted)):
            # Lap i defined by boundaries
            # Duration = bound[i] - bound[i-1] (if i>0 else 0)
            t_start = boundaries_sorted[i-1] if i > 0 else 0.0
            t_end = boundaries_sorted[i]
            dur = t_end - t_start
            lap_stats.append({
                "id": i+1,
                "time": dur,
                "min_rpm": 0, "max_rpm": 0,
                "min_speed": 0, "max_speed": 0,
            })
            
    # --- END OF TELEMETRY/MANUAL SETUP ---
    import numpy as np
    if custom_output_path:
        output_path = custom_output_path
    else:
        # Determine suffix based on mode
        suffix = "-overlay_telemetry" if args.telemetry else "-overlay_photofinish"
        output_path = video_path.with_name(f"{video_path.stem}{suffix}{video_path.suffix}")


    # 4. Render Overlay
    print("[overlay] Building track map...")
    track_overlay = None
    if track_geometry:
        track_overlay = build_track_overlay(track_geometry, samples)
    if track_overlay is None:
        print("[overlay] Warning: Track map generation failed.")
        # Create a dummy overlay to proceed with other gauges

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"[overlay] Using temporary directory: {tmp_path}")
        if args.keep_frames:
            # If keeping frames, use a persistent directory
            tmp_path = args.output.parent / "overlay_frames"
            tmp_path.mkdir(exist_ok=True)
        
        # 1. Render Track Map Frames
        # Note: Track map is now in bottom left, but we still render it separately
        # We just need to adjust the overlay position in ffmpeg
        if track_overlay:
            # We skip separate rendering.
            # But we need to prepare track_overlay_data for the generator.
            
            # Reimplement preparation logic from render_track_frames (simplified)
            # 1. Basic data
            track_overlay_data = {
                "normalized_lines": track_overlay.normalized_lines,
                "positions": track_overlay.positions.tolist() if track_overlay.positions is not None else None,
                "start_finish_normalized": track_overlay.start_finish_normalized,
                "segments": track_overlay.segments,
            }
            
            # 2. Segment Stats
            # Reusing the logic from render_track_frames
            # We need to know speed min/max per segment for each lap.
            
            segment_stats = {}
            if track_overlay.segments:
                # Helper to find nearest segment index
                segments_pts = [np.array(seg["points"]) for seg in track_overlay.segments]
                
                # Optimized approach:
                # Iterate rows? Or vectorize?
                # For simplicity, let's iterate rows like before (it is fast enough usually)
                # But we can't access 'data' directly, we have 'samples' dataframe.
                
                # We need column names
                def pick_col(df, candidates):
                    for c in candidates:
                        if c in df.columns: return c
                    return None
                
                speed_c = pick_col(samples, ["GPS Speed", "Speed", "Vitesse"])
                lap_c = "LapNumber"
                
                if speed_c and lap_c in samples.columns:
                     # Access underlying numpy arrays for speed
                     s_pos = track_overlay.positions # shape (N, 2)
                     
                     if s_pos is not None:
                         # We can pre-assign segments to indices?
                         # Or just loop. 10k frames is fine.
                         
                         # Let's do a KDTree or simple distance check?
                         # Simple distance check for each point is O(N_frames * M_segments). 
                         # M is small (~20), N ~10k. 200k ops. Fine.
                         
                         for idx, row in samples.iterrows():
                             lap = row.get(lap_c)
                             speed = row.get(speed_c)
                             if pd.isna(lap) or pd.isna(speed): continue
                             
                             pos = s_pos[idx] # Assuming idx matches track_overlay.positions index
                             if np.isnan(pos).any(): continue
                             
                             # Find nearest segment
                             best_idx = None
                             best_dist = float('inf')
                             for si, pts in enumerate(segments_pts):
                                 if pts.size == 0: continue
                                 # pts is array of points.
                                 # pos is single point.
                                 dists = np.linalg.norm(pts - pos, axis=1)
                                 min_d = dists.min()
                                 if min_d < best_dist:
                                     best_dist = min_d
                                     best_idx = si
                                     
                             if best_idx is not None:
                                 lap_dict = segment_stats.setdefault(int(lap), {})
                                 seg_dict = lap_dict.setdefault(best_idx, [float('inf'), -float('inf')])
                                 seg_dict[0] = min(seg_dict[0], float(speed))
                                 seg_dict[1] = max(seg_dict[1], float(speed))

            track_overlay_data["segment_stats"] = segment_stats
            print("[overlay] Track overlay data prepared.")
            
        else:
            track_overlay_data = None

        
        # 2. Render Info/Gauge Frames
        print("[overlay] Rendering gauge frames...")
        
        # Prepare Predictive Delta Model
        predictive_model = None
        projector = None
        
        if track and track.projector:
            projector = track.projector
            try:
                if True:
                    
                    # Find Best Lap
                    best_lap = select_best_lap(lap_stats, min_lap_time=20.0)
                    if best_lap:
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

        # --- Prepare Colored Track Map Data (Per Lap) ---
        if track_overlay_data and track_overlay is not None and track_overlay.positions is not None:
             try:
                 track_overlay_data["colored_track_points_by_lap"] = {}
                 
                 # Prepare for all laps provided in samples
                 lap_ids = samples["LapNumber"].unique()
                 all_rpms = []
                 
                 def pick_col_local(df, candidates):
                       for c in candidates:
                           if c in df.columns: return c
                       return None
                 rpm_c = pick_col_local(samples, ["RPM", "Régime", "EngineSpeed"])
                 
                 if rpm_c:
                     valid_laps_count = 0
                     for lap_id in lap_ids:
                         indices = samples.index[samples["LapNumber"] == lap_id].tolist()
                         if not indices: continue
                         
                         subset_pos = track_overlay.positions[indices]
                         subset_rpm = pd.to_numeric(samples.loc[indices, rpm_c], errors='coerce').fillna(0).values
                         
                         valid_mask = ~np.isnan(subset_pos).any(axis=1)
                         valid_pos = subset_pos[valid_mask]
                         valid_rpm = subset_rpm[valid_mask]
                         
                         if len(valid_pos) > 0:
                             colored_pts = np.column_stack((valid_pos, valid_rpm)).tolist()
                             track_overlay_data["colored_track_points_by_lap"][int(lap_id)] = colored_pts
                             all_rpms.extend(valid_rpm)
                             valid_laps_count += 1
                     
                     if all_rpms:
                         track_overlay_data["legend_min"] = float(np.min(all_rpms))
                         track_overlay_data["legend_max"] = float(np.max(all_rpms))
                         track_overlay_data["track_map_cache"] = {} # Initialize cache
                         print(f"[overlay] Prepared colored track maps for {valid_laps_count} laps. Range: {track_overlay_data['legend_min']} - {track_overlay_data['legend_max']}")

             except Exception as e:
                 print(f"[overlay] Warning: Failed to prepare colored track map: {e}")

        # 2. Render Info/Gauge Frames (On-the-fly)
        print("[overlay] Preparing on-the-fly frame generation...")
        
        # Initialize overlay_specs
        overlay_specs = []

        
        # Prepare data series once (copied solely from render_info_frames logic)
        # We need to reimplement the preparation logic from render_info_frames here locally
        # or split it out. Since we are replacing render_info_frames, let's just do it here.
        
        # ... actually, render_info_frames had a lot of prep code (picking columns etc).
        # To avoid duplicating massive code, I should probably keep a helper "prepare_overlay_data" 
        # or just inline the relevant parts since render_info_frames is no longer used.
        
        # Let's verify what we need.
        # We need:
        # font_paths
        # speeds, rpms, lat_gs, lon_gs (numeric series)
        # distance
        # lap_stats
        # predictive_model, projector
        # positions_webmerc
        
        font_paths = {
            "large": get_font_path(bold=True),
            "medium": get_font_path(bold=True),
            "small": get_font_path(bold=False),
        }
        
        def pick_col(df, candidates):
            for c in candidates:
                if c in df.columns:
                     return c
            return None
            
        def numeric_s(col_name):
            if not col_name: return None
            return pd.to_numeric(samples[col_name], errors="coerce").fillna(0.0)

        # Extract series
        speed_col = pick_col(samples, ["GPS Speed", "Speed", "Vitesse"])
        rpm_col = pick_col(samples, ["RPM", "Régime"])
        lat_g_col = pick_col(samples, ["GPS LatAcc", "LatAcc", "Lateral Acceleration"])
        lon_g_col = pick_col(samples, ["GPS LonAcc", "LonAcc", "Longitudinal Acceleration"])

        speeds = numeric_s(speed_col)
        rpms = numeric_s(rpm_col)
        lat_gs = numeric_s(lat_g_col)
        lon_gs = numeric_s(lon_g_col)
        
        # Pre-calculate max values
        max_speed = speeds.max() if speeds is not None else 100.0
        max_rpm = rpms.max() if rpms is not None else 10000.0
        
        # Distance (already in samples? render_info_frames calculated it again?)
        # render_info_frames calculated distance if not present.
        # But samples came from resample_telemetry.
        # Let's just recalculate to be safe.
        distance = None
        if speeds is not None:
             dist_inc = speeds / video.fps / 3600.0
             distance = dist_inc.cumsum()
             
        # Projector positions
        positions_webmerc = None
        if projector:
             lats = numeric_s("GPS Latitude")
             lons = numeric_s("GPS Longitude")
             if lats is not None and lons is not None:
                  lats = lats.replace(0.0, np.nan).interpolate().ffill().bfill()
                  lons = lons.replace(0.0, np.nan).interpolate().ffill().bfill()
                  xs, ys = WGS84_TO_WEBMERC.transform(lons.to_numpy(), lats.to_numpy())
                  positions_webmerc = np.column_stack([xs, ys])

        # Define the generator
        def overlay_generator():
            # Generate Static Overlay ONCE
            print("[overlay] Generating static overlay...")
            static_img = generate_static_overlay(
                video.width,
                video.height,
                font_paths,
                lap_stats,
                track_overlay_data
            )
            
            total_frames = len(samples)
            print(f"[overlay] Generator started for {total_frames} frames.")
            
            # Use main process sequential rendering
            from tqdm import tqdm
            
            for index, row in samples.iterrows():
                 video_time = row.get("VideoTime", 0.0)
                 
                 # Extract values
                 spd = float(speeds.iloc[index]) if speeds is not None else None
                 r_pm = float(rpms.iloc[index]) if rpms is not None else None
                 lat = float(lat_gs.iloc[index]) if lat_gs is not None else None
                 lon = float(lon_gs.iloc[index]) if lon_gs is not None else None
                 dist = float(distance.iloc[index]) if distance is not None else None
                 
                 current_pos = None
                 if positions_webmerc is not None:
                     current_pos = tuple(positions_webmerc[index])
                 
                 lap_num = int(row.get("LapNumber", 0))
                 lap_t = float(row.get("LapTime", 0.0))
                 
                 absolute_time = row.get("AbsoluteTime")
                 if pd.isna(absolute_time):
                     absolute_time = None
                     
                 # Render Frame
                 frame_bytes = render_info_frame(
                     index=index,
                     video_time=video_time,
                     driver=session.driver,
                     venue=session.venue,
                     session_name=session.session,
                     event_date=session.event_date,
                     overlay_width=video.width,
                     overlay_height=video.height,
                     font_paths=font_paths,
                     heading=None,
                     lap_number=lap_num,
                     lap_time=lap_t,
                     speed=spd,
                     rpm=r_pm,
                     max_speed=max_speed,
                     max_rpm=max_rpm,
                     lat_g=lat,
                     lon_g=lon,
                     distance=dist,
                     lap_stats=lap_stats,
                     output_dir=None, # In-memory
                     absolute_time=absolute_time,
                     predictive_model=predictive_model,
                     projector=projector,

                     current_pos=current_pos,
                     
                     # Pass track data for composition
                     track_overlay_data={**track_overlay_data, "current_lap": lap_num} if track_overlay_data else None,
                     static_overlay=static_img,
                 )
                 
                 yield frame_bytes

        overlay_specs.append({
            "pattern": "pipe:0", # Read from stdin
            "x": "0",
            "y": "0",
            "format": "rawvideo",
            "pixel_format": "rgba", # PIL uses RGBA by default
            "size": f"{video.width}x{video.height}",
        })

        
        # Prepare stdin generator specs

    # Stabilization Preparation
    stab_filter = None
    if args.stabilize:
        print("[overlay] Stabilization enabled. Generating transforms...")
        transforms_file = stab.generate_transforms(args.video)
        if transforms_file:
            print(f"[overlay] Transforms ready: {transforms_file.name}")
            stab_filter = stab.get_transform_filter(
                transforms_file,
                smoothing=args.stab_smoothing,
                zoom=args.stab_zoom,
                optzoom=args.stab_optzoom
            )
        else:
            print("[overlay] Warning: Stabilization failed during detection. Skipping stabilization.")

    # --- RENDER VIDEO ---
    print(f"[overlay] Rendering to {output_path}...")
    run_ffmpeg(
        video_path=args.video,
        overlay_specs=overlay_specs,
        fps=video.fps,
        start=args.start,
        duration=video.duration if args.duration is None else args.duration,
        output_path=output_path,
        hwaccel_cuda=use_cuda,
        video_codec=video_codec,
        stdin_generator=overlay_generator(),
        stab_filter=stab_filter,
    )
    
    print("-" * 40)
    print(f"Done! Output saved to:\n{output_path}")
    
    # Optional: Cleanup transforms? 
    # User requested persistent caching, so we keep it.
    if args.export_best_lap:
        # Find best lap
        best_lap = select_best_lap(lap_stats, min_lap_time=20.0)
        if best_lap:
            best_lap_id = best_lap["id"]
            print(f"[overlay] Exporting Best Lap: {best_lap_id} ({best_lap['time']:.3f}s)")
            
            # Get start/end time of best lap
            # We need to find the crossing times for this lap
            # Lap N starts at crossings[N-1] (or start if N=0/1?)
            # crossings[0] is end of Lap 0 (Out Lap) / Start of Lap 1
            # crossings[k] is end of Lap k+1
            
            # Lap 1: Start = crossings[0] (approx? No, Lap 1 starts when we cross line first time? No, Lap 0 is out lap)
            # Let's look at calculate_laps:
            # crossings[0] is the FIRST crossing.
            # data before crossings[0] is Lap 0.
            # data between crossings[0] and crossings[1] is Lap 1.
            
            # So Lap K (where K >= 1) corresponds to interval [crossings[K-1], crossings[K]]
            # Wait, indices:
            # Lap 1: crossings[0] to crossings[1]
            # Lap id is 1-based in stats?
            # calculate_lap_stats uses "LapNumber" from dataframe.
            # calculate_laps assigns:
            # current_lap = 0
            # if t >= crossings[0]: current_lap = 1
            # if t >= crossings[1]: current_lap = 2
            
            # So Lap K is between crossings[K-1] and crossings[K]
            # Exception: Last lap might not have an end crossing if incomplete?
            # But valid_stats usually implies completed laps (if we rely on lap_durations).
            
            # Let's get times
            # We need to map Lap ID to start/end time.
            # crossings is a list of floats (video time? No, it's whatever time base was in telemetry)
            # In calculate_laps, we used "Time" column or index.
            # In resample_telemetry, we aligned telemetry to video time.
            # But calculate_laps was called on the ORIGINAL session.table (before resampling).
            # So crossings are in ORIGINAL telemetry time.
            # We need to convert them to Video Time using args.time_shift.
            # VideoTime = TelemetryTime - args.time_shift
            
            # Re-calculate crossings in Video Time?
            # Or just use the original crossings and apply shift.
            
            # Lap ID = best_lap_id
            # Start Index = best_lap_id - 1
            # End Index = best_lap_id
            
            # Check bounds
            if 0 <= best_lap_id - 1 < len(crossings) and best_lap_id < len(crossings):
                t_start_telemetry = crossings[best_lap_id - 1]
                t_end_telemetry = crossings[best_lap_id]
                
                t_start_video = t_start_telemetry - args.time_shift
                t_end_video = t_end_telemetry - args.time_shift
                
                # Add buffer
                clip_start = max(0.0, t_start_video - 10.0)
                clip_end = min(video.duration, t_end_video + 10.0)
                clip_duration = clip_end - clip_start
                
                if clip_duration > 0:
                    print(f"[overlay] Best Lap Video Range: {clip_start:.2f}s to {clip_end:.2f}s")
                    
                    best_lap_output = output_path.with_name(f"{output_path.stem}-best-lap{output_path.suffix}")
                    
                    # Calculate start_number for overlay frames
                    # Overlay frames were generated starting at args.start
                    # Frame 0 corresponds to args.start
                    # We want frame corresponding to clip_start
                    # delta = clip_start - args.start
                    # start_frame = int(delta * video.fps)
                    
                    delta = clip_start - args.start
                    start_frame = int(delta * video.fps)
                    
                    # We need to adjust overlay specs
                    # Since we are starting a NEW pipe stream for this clip, we treat it as starting from frame 0
                    # But the backing video is cut from clip_start.
                    
                    best_lap_specs = []
                    for spec in overlay_specs:
                        new_spec = spec.copy()
                        # Remove start_number if present, as we are providing a fresh stream
                        if "start_number" in new_spec:
                            del new_spec["start_number"]
                        best_lap_specs.append(new_spec)

                    # Create a generator for just this lap
                    # We need to slice the samples DF
                    
                    # clip_start is relative to the ORIGINAL video timeline (video_time)? 
                    # No, clip_start was calculated as t_start_video.
                    # video.duration is absolute duration of original file.
                    # args.start is where the MAIN overlay started.
                    # samples contains "VideoTime" relative to args.start?
                    # Let's check:
                    # resample_telemetry:
                    # relative_video_times = np.arange(total_frames) / fps
                    # samples["VideoTime"] = relative_video_times
                    # "Time" (Telemetry) = clip_start + relative_video_times + time_shift
                    #
                    # In the loop: video_time = row["VideoTime"]
                    # render_info_frame uses video_time for display.
                    
                    # So if we want to render the Best Lap which is at absolute video time `clip_start`...
                    # We must map `clip_start` to `VideoTime` in samples.
                    # samples["VideoTime"] runs from 0 to duration (of the main clip).
                    # args.start is the offset of the main clip from the video file start.
                    
                    # clip_start (Best Lap Start) is absolute video time (0-based from file start).
                    # So we need to find samples where:
                    # absolute_video_time >= clip_start
                    # absolute_video_time = args.start + samples["VideoTime"]
                    
                    # Let's verify clip_start calculation
                    # t_start_video = t_start_telemetry - args.time_shift
                    # This IS absolute video time.
                    
                    # So we need samples where:
                    # args.start + VideoTime >= clip_start
                    # VideoTime >= clip_start - args.start
                    
                    rel_start = clip_start - args.start
                    rel_end = clip_end - args.start
                    
                    # It is possible clip_start < args.start if best lap was before the main clip start.
                    # But usually we render the whole session.
                    
                    mask = (samples["VideoTime"] >= rel_start) & (samples["VideoTime"] < rel_end)
                    best_lap_samples = samples[mask]
                    
                    print(f"[overlay] Exporting {len(best_lap_samples)} frames for best lap.")
                    
                    def best_lap_generator():
                         # We duplicate logic from main generator but iterating best_lap_samples
                         
                         static_img_bl = generate_static_overlay(
                            video.width, video.height, font_paths, lap_stats, track_overlay_data
                         )
                         
                         for index, row in best_lap_samples.iterrows():
                             video_time = row.get("VideoTime", 0.0)
                             
                             # Extract values (same as main loop)
                             spd = float(speeds.iloc[index]) if speeds is not None else None
                             r_pm = float(rpms.iloc[index]) if rpms is not None else None
                             lat = float(lat_gs.iloc[index]) if lat_gs is not None else None
                             lon = float(lon_gs.iloc[index]) if lon_gs is not None else None
                             dist = float(distance.iloc[index]) if distance is not None else None
                             
                             current_pos = None
                             if positions_webmerc is not None:
                                 current_pos = tuple(positions_webmerc[index])
                             
                             lap_num = int(row.get("LapNumber", 0))
                             lap_t = float(row.get("LapTime", 0.0))
                             
                             absolute_time = row.get("AbsoluteTime")
                             if pd.isna(absolute_time): absolute_time = None
                                 
                             yield render_info_frame(
                                 index=index,
                                 video_time=video_time,
                                 driver=session.driver,
                                 venue=session.venue,
                                 session_name=session.session,
                                 event_date=session.event_date,
                                 overlay_width=video.width,
                                 overlay_height=video.height,
                                 font_paths=font_paths,
                                 heading=None,
                                 lap_number=lap_num,
                                 lap_time=lap_t,
                                 speed=spd,
                                 rpm=r_pm,
                                 max_speed=max_speed,
                                 max_rpm=max_rpm,
                                 lat_g=lat,
                                 lon_g=lon,
                                 distance=dist,
                                 lap_stats=lap_stats,
                                 output_dir=None,
                                 absolute_time=absolute_time,
                                 predictive_model=predictive_model,
                                 projector=projector,
                                 current_pos=current_pos,
                                 track_overlay_data={**track_overlay_data, "current_lap": lap_num} if track_overlay_data else None,
                                 static_overlay=static_img_bl,
                             )
                            
                    if not best_lap_samples.empty:
                        run_ffmpeg(
                            video_path=args.video,
                            overlay_specs=best_lap_specs,
                            fps=video.fps,
                            start=clip_start,
                            duration=clip_duration,
                            output_path=best_lap_output,
                            hwaccel_cuda=use_cuda,
                            video_codec=video_codec,
                            stdin_generator=best_lap_generator(),
                            stab_filter=stab_filter,
                        )
                    else:
                        print("[overlay] Warning: Best lap time range is outside the processed video range.")
                else:
                    print("[overlay] Best lap clip duration invalid.")
            else:
                print(f"[overlay] Could not determine start/end times for Lap {best_lap_id}")
        else:
            print("[overlay] No valid best lap found.")

    if not args.keep_frames:
        pass # Temp dir is auto-cleaned


    print(f"\n[overlay] Successfully generated: {output_path}")

if __name__ == "__main__":
    main()
