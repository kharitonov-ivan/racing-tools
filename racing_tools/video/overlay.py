"""Overlay rendering utilities for video generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from PIL import ImageDraw
    from racing_tools.track.track import Track


def format_duration(value: float | int | None, decimals: int = 1) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "--:--"
    total = max(0.0, float(value))
    minutes = int(total // 60)
    seconds = total - minutes * 60
    width = 3 + decimals
    seconds_fmt = f"{seconds:0{width}.{decimals}f}"
    return f"{minutes:02d}:{seconds_fmt}"


def get_gradient_color(value: float, v_min: float, v_max: float) -> str:
    if v_max <= v_min:
        t = 0.5
    else:
        t = (value - v_min) / (v_max - v_min)
    t = max(0.0, min(1.0, t))

    start_rgb = (74, 144, 226)
    mid_rgb = (80, 227, 194)
    end_rgb = (226, 74, 74)

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


def pick_column(frame: "pd.DataFrame", candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def normalize_track_polylines(layout: "Track") -> list[list[tuple[float, float]]]:
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


def normalize_polyline(points: list[tuple[float, float]] | None, bounds: tuple[float, float, float, float]) -> list[tuple[float, float]] | None:
    if not points:
        return None
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    return [((x - min_x) / span_x, 1.0 - (y - min_y) / span_y) for x, y in points]


def normalize_track_positions(points: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    if points.size == 0:
        return points
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    xs = (points[:, 0] - min_x) / span_x
    ys = 1.0 - (points[:, 1] - min_y) / span_y
    stacked = np.column_stack([np.clip(xs, 0.0, 1.0), np.clip(ys, 0.0, 1.0)])
    return stacked


@dataclass
class TrackOverlay:
    normalized_lines: list[list[tuple[float, float]]]
    positions: np.ndarray | None
    start_finish_normalized: list[tuple[float, float]] | None
    start_finish_wgs84: list[tuple[float, float]] | None
    segments: list[dict] | None


def build_track_overlay(geometry: "Track", samples: "pd.DataFrame") -> TrackOverlay | None:
    from racing_tools.session.session import WGS84_TO_WEBMERC

    layout = geometry  # Track.layout returns self, so use directly

    lat_col = pick_column(samples, ["GPS Latitude", "Latitude"])
    lon_col = pick_column(samples, ["GPS Longitude", "Longitude"])
    if not lat_col or not lon_col:
        print("[overlay] Telemetry lacks GPS Latitude/Longitude; track map disabled")
        return None

    lat = samples[lat_col].interpolate().ffill().bfill()
    lon = samples[lon_col].interpolate().ffill().bfill()
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
            norm_points = normalize_polyline(seg["points"], layout.bounds)
            normalized_segments.append({"type": seg["type"], "points": norm_points})

    return TrackOverlay(
        normalized_lines=normalized_lines,
        positions=normalized_positions,
        start_finish_normalized=start_finish_norm,
        start_finish_wgs84=geometry.start_finish_wgs84,
        segments=normalized_segments,
    )


def draw_track_static(
    draw: "ImageDraw.ImageDraw",
    map_box: tuple[int, int, int, int],
    track_overlay_data: dict,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = map_box
    draw.rounded_rectangle(map_box, radius=18, fill=(0, 0, 0, 140), outline="#214d66")

    pad = 18
    inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
    width = max(inner[2] - inner[0], 1)
    height = max(inner[3] - inner[1], 1)

    segments = track_overlay_data.get("segments")
    normalized_lines = track_overlay_data.get("normalized_lines", [])

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

            draw.line(scaled, fill="#333333", width=8)
    else:
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

    return (inner[0], inner[1], width, height)
