#!/usr/bin/env python3
"""Generate a PNG report with track map, sectors, lap table, and statistics."""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

from racing_tools.session.session import Session, WGS84_TO_WEBMERC
from racing_tools.track.track import Track


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get font with fallback."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def format_time(seconds: Optional[float]) -> str:
    """Format seconds as M:SS.mmm."""
    if seconds is None or not math.isfinite(seconds):
        return "-"
    m = int(seconds / 60)
    s = seconds - m * 60
    return f"{m}:{s:06.3f}"


def format_sector_time(seconds: Optional[float]) -> str:
    """Format sector time as SS.mm (no minutes, 2 decimal places)."""
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return "-"
    return f"{seconds:.2f}"


def format_speed(speed: Optional[float]) -> str:
    """Format speed in km/h."""
    if speed is None or not math.isfinite(speed):
        return "-"
    return f"{int(speed)}"


def calculate_sector_stats(
    session_table: pd.DataFrame,
    lap_id: int,
    sectors: list[dict],
) -> dict[int, tuple[float, float, float, float]]:
    """Calculate min/max speed and sector time for each sector in a lap.

    Returns:
        Dict mapping sector index to (min_speed, max_speed, sector_time, sector_distance)
    """
    stats = {}

    speed_col = next((c for c in ("GPS Speed", "Speed", "Wheel Speed") if c in session_table.columns), None)
    if "Distance" not in session_table.columns or speed_col is None:
        return stats

    lap_data = session_table[session_table["LapNumber"] == lap_id]
    if lap_data.empty:
        return stats

    lap_start_dist = lap_data["Distance"].min()
    cumulative_dist = 0.0

    for idx, sector in enumerate(sectors):
        pts = sector.get("points", [])
        if len(pts) < 2:
            continue

        pts_arr = np.array(pts)

        if np.max(np.abs(pts_arr[:, 0])) <= 180 and np.max(np.abs(pts_arr[:, 1])) <= 90:
            xs, ys = WGS84_TO_WEBMERC.transform(pts_arr[:, 0], pts_arr[:, 1])
            pts_m = np.column_stack((xs, ys))
        else:
            pts_m = pts_arr

        diffs = np.linalg.norm(pts_m[1:] - pts_m[:-1], axis=1)
        seg_len = np.sum(diffs)

        start_d = cumulative_dist
        end_d = cumulative_dist + seg_len
        cumulative_dist += seg_len

        mask = (lap_data["Distance"] >= (lap_start_dist + start_d)) & (lap_data["Distance"] < (lap_start_dist + end_d))
        seg_samples = lap_data[mask]

        if not seg_samples.empty:
            spd = seg_samples[speed_col]
            if "Time" in seg_samples.columns:
                sector_time = seg_samples["Time"].max() - seg_samples["Time"].min()
            else:
                sector_time = 0.0
            stats[idx] = (float(spd.min()), float(spd.max()), sector_time, seg_len)

    return stats


def draw_track_map(
    draw: ImageDraw.ImageDraw,
    track: Track,
    sectors: list[dict],
    sector_colors: list[str],
    origin: tuple[int, int],
    size: tuple[int, int],
    padding: int = 20,
) -> None:
    """Draw track outline and sector markers."""
    bounds = track.bounds
    min_x, max_x, min_y, max_y = bounds
    x_range = max_x - min_x or 1
    y_range = max_y - min_y or 1

    ox, oy = origin
    w, h = size

    def scale_point(pt: tuple[float, float]) -> tuple[int, int]:
        nx = (pt[0] - min_x) / x_range
        ny = 1.0 - (pt[1] - min_y) / y_range
        sx = ox + padding + nx * (w - 2 * padding)
        sy = oy + padding + ny * (h - 2 * padding)
        return (int(sx), int(sy))

    # Draw centerline
    centerline = track.centerline
    if centerline is not None:
        scaled = [scale_point(p) for p in centerline]
        if len(scaled) > 1:
            draw.line(scaled, fill="#666666", width=3)

    for idx, sector in enumerate(sectors):
        pts = sector.get("points", [])
        if pts:
            start_pt = pts[0]
            sx, sy = scale_point(start_pt)
            r = 8
            color = sector_colors[idx % len(sector_colors)]
            draw.ellipse([(sx - r, sy - r), (sx + r, sy + r)], fill=color, outline="#000000")
            fnt = get_font(12)
            draw.text((sx, sy), str(idx + 1), font=fnt, fill="#000000", anchor="mm")


def draw_lap_table(
    draw: ImageDraw.ImageDraw,
    lap_stats: list[dict],
    sector_splits: dict[int, dict[str, float]],
    origin: tuple[int, int],
    split_names: list[str],
    best_lap_id: int,
) -> int:
    """Draw lap table with sector split times.

    Returns:
        Height used
    """
    fnt_header = get_font(14)
    fnt_row = get_font(12)

    ox, oy = origin
    row_h = 22
    header_h = 30

    n_sectors = len(split_names)
    col_widths = [50, 80] + [80] * n_sectors
    col_headers = ["Lap", "Time"] + split_names

    valid_laps = [s for s in lap_stats if s.get("time") and s["time"] > 20.0 and not s.get("label")]
    top3 = sorted([(s["id"], s["time"]) for s in valid_laps], key=lambda x: x[1])[:3]
    top3_ids = {lid: i + 1 for i, (lid, _) in enumerate(top3)}

    # Find best time per sector split
    best_split_times: dict[str, float] = {}
    for s in valid_laps:
        splits = sector_splits.get(s["id"], {})
        for name, t in splits.items():
            if t > 0 and (name not in best_split_times or t < best_split_times[name]):
                best_split_times[name] = t

    y = oy
    x = ox
    for header, w in zip(col_headers, col_widths):
        draw.rectangle([x, y, x + w - 1, y + header_h - 1], fill="#333333", outline="#555555")
        draw.text((x + w // 2, y + header_h // 2), header, font=fnt_header, fill="#FFFFFF", anchor="mm")
        x += w

    y += header_h

    sorted_stats = sorted(lap_stats, key=lambda s: s["id"])

    for s in sorted_stats:
        lap_id = s["id"]
        label = s.get("label")
        is_pit = label in ("POUT", "PIN")

        x = ox

        if is_pit:
            bg_color = "#444444"
            text_color = "#888888"
        elif lap_id in top3_ids:
            rank = top3_ids[lap_id]
            bg_color = ["#4a0000", "#00304a", "#004a00"][rank - 1]
            text_color = "#FFFFFF"
        elif lap_id == best_lap_id:
            bg_color = "#4a4a00"
            text_color = "#FFFFFF"
        else:
            bg_color = "#222222"
            text_color = "#CCCCCC"

        for col_idx, w in enumerate(col_widths):
            draw.rectangle([x, y, x + w - 1, y + row_h - 1], fill=bg_color, outline="#333333")

            if col_idx == 0:
                text = label if label else str(lap_id)
                draw.text((x + 5, y + row_h // 2), text, font=fnt_row, fill=text_color, anchor="lm")
            elif col_idx == 1:
                text = format_time(s.get("time"))
                draw.text((x + w - 5, y + row_h // 2), text, font=fnt_row, fill=text_color, anchor="rm")
            else:
                split_name = split_names[col_idx - 2]
                splits = sector_splits.get(lap_id, {})
                if split_name in splits and not is_pit:
                    sec_time = splits[split_name]
                    time_text = format_sector_time(sec_time)
                    is_best = sec_time > 0 and abs(sec_time - best_split_times.get(split_name, -1)) < 0.001
                    time_color = "#00FF00" if is_best else text_color
                    draw.text((x + w // 2, y + row_h // 2), time_text, font=fnt_row, fill=time_color, anchor="mm")

            x += w

        y += row_h

    return y - oy


def draw_statistics(
    draw: ImageDraw.ImageDraw,
    lap_stats: list[dict],
    sector_splits: dict[int, dict[str, float]],
    origin: tuple[int, int],
    split_names: list[str],
    session_table: Optional[pd.DataFrame] = None,
) -> int:
    """Draw statistics table (mean, median) for valid laps.

    Returns:
        Height used
    """
    fnt_header = get_font(14)
    fnt_row = get_font(12)
    fnt_small = get_font(11)

    ox, oy = origin
    row_h = 24
    header_h = 30

    n_sectors = len(split_names)
    col_widths = [60, 70] + [80] * n_sectors
    col_headers = ["Stat", "Lap Time"] + split_names

    valid_laps = [s for s in lap_stats if s.get("time") and s["time"] > 20.0 and not s.get("label")]

    y = oy + 20

    draw.text((ox, y), "Statistics (Valid Laps)", font=fnt_header, fill="#FFFFFF")
    y += 25

    x = ox
    for header, w in zip(col_headers, col_widths):
        draw.rectangle([x, y, x + w - 1, y + header_h - 1], fill="#333333", outline="#555555")
        draw.text((x + w // 2, y + header_h // 2), header, font=fnt_header, fill="#FFFFFF", anchor="mm")
        x += w
    y += header_h

    if not valid_laps:
        draw.text((ox + 10, y + row_h // 2), "No valid laps", font=fnt_row, fill="#888888")
        return y - oy + row_h

    lap_times = np.array([s["time"] for s in valid_laps])
    sector_times: dict[str, list[float]] = {name: [] for name in split_names}

    for s in valid_laps:
        lap_id = s["id"]
        splits = sector_splits.get(lap_id, {})
        for name in split_names:
            if name in splits:
                sector_times[name].append(splits[name])

    stats_rows = [
        ("Mean", np.mean),
        ("Median", np.median),
        ("Std", np.std),
        ("Min", np.min),
        ("Max", np.max),
    ]

    for stat_name, stat_fn in stats_rows:
        x = ox

        for col_idx, w in enumerate(col_widths):
            draw.rectangle([x, y, x + w - 1, y + row_h - 1], fill="#1a1a1a", outline="#333333")

            if col_idx == 0:
                draw.text((x + 5, y + row_h // 2), stat_name, font=fnt_row, fill="#AAAAAA", anchor="lm")
            elif col_idx == 1:
                try:
                    val = stat_fn(lap_times)
                    text = format_time(val)
                except Exception:
                    text = "-"
                draw.text((x + w - 5, y + row_h // 2), text, font=fnt_row, fill="#CCCCCC", anchor="rm")
            else:
                name = split_names[col_idx - 2]
                times = sector_times.get(name, [])
                if times:
                    try:
                        val = stat_fn(times)
                        text = format_sector_time(val)
                    except Exception:
                        text = "-"
                else:
                    text = "-"
                draw.text((x + w // 2, y + row_h // 2), text, font=fnt_row, fill="#CCCCCC", anchor="mm")

            x += w
        y += row_h

    y += 10
    n_laps = len(valid_laps)
    best_time = np.min(lap_times)
    mean_time = np.mean(lap_times)
    median_time = np.median(lap_times)
    variance_time = np.var(lap_times)

    summary = f"Valid Laps: {n_laps}  |  Best: {format_time(best_time)}  |  Mean: {format_time(mean_time)}  |  Median: {format_time(median_time)}  |  Var: {variance_time:.3f}s²"
    draw.text((ox, y), summary, font=fnt_row, fill="#88CCFF")

    y += 30

    accel_stats = calculate_acceleration_stats(session_table)
    if accel_stats:
        draw.text((ox, y), "Acceleration (G)", font=fnt_header, fill="#FFFFFF")
        y += 25

        accel_headers = ["Type", "Min", "Max"]
        accel_widths = [100, 80, 80]
        x = ox
        for header, w in zip(accel_headers, accel_widths):
            draw.rectangle([x, y, x + w - 1, y + header_h - 1], fill="#333333", outline="#555555")
            draw.text((x + w // 2, y + header_h // 2), header, font=fnt_header, fill="#FFFFFF", anchor="mm")
            x += w
        y += header_h

        for accel_type, (min_val, max_val) in accel_stats.items():
            x = ox
            for col_idx, w in enumerate(accel_widths):
                draw.rectangle([x, y, x + w - 1, y + row_h - 1], fill="#1a1a1a", outline="#333333")
                if col_idx == 0:
                    draw.text((x + 5, y + row_h // 2), accel_type, font=fnt_small, fill="#AAAAAA", anchor="lm")
                elif col_idx == 1:
                    draw.text((x + w - 5, y + row_h // 2), f"{min_val:.2f}", font=fnt_small, fill="#CCCCCC", anchor="rm")
                else:
                    draw.text((x + w - 5, y + row_h // 2), f"{max_val:.2f}", font=fnt_small, fill="#CCCCCC", anchor="rm")
                x += w
            y += row_h

    return y - oy + 20


def calculate_acceleration_stats(session_table: Optional[pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """Calculate min/max acceleration values from session data.

    Returns:
        Dict mapping acceleration type to (min, max) tuple in G units
    """
    if session_table is None or session_table.empty:
        return {}

    stats = {}

    lat_cols = [
        "GPS Accel Lat",
        "GPS LatAcc",
        "LatAcc",
        "Lateral Acceleration",
        "IMU Accel Lat",
        "LateralAcc",
        "AccelerometerY",
        "G Lat",
        "G_Lat",
        "Accel Lat Smoothed",
        "G Lat Smth",
        "IMU Accel Lat Filtered",
        "G Lat imu f",
    ]
    lon_cols = [
        "GPS Accel Long",
        "GPS LonAcc",
        "LonAcc",
        "Longitudinal Acceleration",
        "IMU Accel Long",
        "LongitudinalAcc",
        "AccelerometerX",
        "G Long",
        "G_Long",
        "Accel Long Smoothed",
        "G Long Smth",
        "IMU Accel Long Filtered",
        "G Long imu f",
    ]

    lat_col = next((c for c in lat_cols if c in session_table.columns), None)
    lon_col = next((c for c in lon_cols if c in session_table.columns), None)

    if lat_col:
        lat_vals = pd.to_numeric(session_table[lat_col], errors="coerce").dropna()
        if not lat_vals.empty:
            stats["Lateral"] = (float(lat_vals.min()), float(lat_vals.max()))

    if lon_col:
        lon_vals = pd.to_numeric(session_table[lon_col], errors="coerce").dropna()
        if not lon_vals.empty:
            stats["Longitudinal"] = (float(lon_vals.min()), float(lon_vals.max()))

    return stats


def draw_segment_table(
    draw: ImageDraw.ImageDraw,
    lap_stats: list[dict],
    segment_stats_all: dict[int, dict[int, tuple[float, float, float, float]]],
    segments: list[dict],
    origin: tuple[int, int],
    best_lap_id: int,
) -> int:
    """Draw segment (auto-generated straights/turns) table with times and speeds.

    Returns:
        Height used
    """
    fnt_header = get_font(12)
    fnt_row = get_font(10)

    n_segments = len(segments)
    if n_segments == 0:
        return 0

    ox, oy = origin
    row_h = 18
    header_h = 30

    # Column per segment, narrow to fit many
    seg_col_w = 55
    col_widths = [40, 70] + [seg_col_w] * n_segments

    # Build header labels: segment type abbreviation
    seg_headers = []
    for idx, seg in enumerate(segments):
        seg_type = seg.get("type", "?")
        label = f"{'T' if seg_type == 'turn' else 'S'}{idx + 1}"
        seg_headers.append(label)
    col_headers = ["Lap", "Time"] + seg_headers

    valid_laps = [s for s in lap_stats if s.get("time") and s["time"] > 20.0 and not s.get("label")]
    top3 = sorted([(s["id"], s["time"]) for s in valid_laps], key=lambda x: x[1])[:3]
    top3_ids = {lid: i + 1 for i, (lid, _) in enumerate(top3)}

    # Best segment times
    best_seg_times: dict[int, float] = {}
    for s in valid_laps:
        stats = segment_stats_all.get(s["id"], {})
        for seg_idx, (_, _, seg_time, _) in stats.items():
            if seg_time > 0 and (seg_idx not in best_seg_times or seg_time < best_seg_times[seg_idx]):
                best_seg_times[seg_idx] = seg_time

    y = oy
    draw.text((ox, y), "Track Segments", font=get_font(14), fill="#FFFFFF")
    y += 20

    x = ox
    for header, w in zip(col_headers, col_widths):
        draw.rectangle([x, y, x + w - 1, y + header_h - 1], fill="#333333", outline="#555555")
        draw.text((x + w // 2, y + header_h // 2), header, font=fnt_header, fill="#FFFFFF", anchor="mm")
        x += w
    y += header_h

    sorted_stats = sorted(lap_stats, key=lambda s: s["id"])

    for s in sorted_stats:
        lap_id = s["id"]
        label = s.get("label")
        is_pit = label in ("POUT", "PIN")
        x = ox

        if is_pit:
            bg_color = "#444444"
            text_color = "#888888"
        elif lap_id in top3_ids:
            rank = top3_ids[lap_id]
            bg_color = ["#4a0000", "#00304a", "#004a00"][rank - 1]
            text_color = "#FFFFFF"
        elif lap_id == best_lap_id:
            bg_color = "#4a4a00"
            text_color = "#FFFFFF"
        else:
            bg_color = "#222222"
            text_color = "#CCCCCC"

        for col_idx, w in enumerate(col_widths):
            draw.rectangle([x, y, x + w - 1, y + row_h - 1], fill=bg_color, outline="#333333")

            if col_idx == 0:
                text = label if label else str(lap_id)
                draw.text((x + 3, y + row_h // 2), text, font=fnt_row, fill=text_color, anchor="lm")
            elif col_idx == 1:
                text = format_time(s.get("time"))
                draw.text((x + w - 3, y + row_h // 2), text, font=fnt_row, fill=text_color, anchor="rm")
            else:
                seg_idx = col_idx - 2
                seg_stats = segment_stats_all.get(lap_id, {})
                if seg_idx in seg_stats and not is_pit:
                    min_spd, max_spd, seg_time, _ = seg_stats[seg_idx]
                    time_text = format_sector_time(seg_time)
                    is_best = seg_time > 0 and abs(seg_time - best_seg_times.get(seg_idx, -1)) < 0.001
                    color = "#00FF00" if is_best else text_color
                    draw.text((x + w // 2, y + row_h // 2), time_text, font=fnt_row, fill=color, anchor="mm")

            x += w
        y += row_h

    return y - oy


def generate_report(
    telemetry_path: Path,
    track_dir: Path,
    output_path: Path,
    map_size: tuple[int, int] = (500, 500),
) -> None:
    """Generate PNG report with track map, lap table, and statistics."""
    print(f"Loading telemetry from {telemetry_path}...")
    session = Session.load(telemetry_path)

    print(f"Loading track from {track_dir}...")
    track = Track.load(track_dir)

    session.track = track
    print("Detecting crossings...")
    session.detect_crossings()
    session.add_lap_numbers()

    if "Distance" not in session.table.columns:
        print("Warning: No Distance column available")

    lap_stats = session.get_lap_stats()
    best_lap = session.best_lap
    best_lap_id = best_lap["id"] if best_lap else -1

    print(f"Found {len(lap_stats)} laps, best lap: {best_lap_id}")

    # Detect sector crossings and compute split times
    session.detect_sector_crossings()
    sector_splits = session.get_sector_splits()

    # Get split names from first valid lap
    split_names: list[str] = []
    for lap_splits in sector_splits.values():
        split_names = list(lap_splits.keys())
        break
    n_sectors = len(split_names)
    print(f"Track has {n_sectors} sector splits: {split_names}")

    # Track segments for map visualization and segment stats
    segments = track.segments or []
    n_segments = len(segments)
    segment_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]

    # Calculate segment stats
    segment_stats_all: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for s in lap_stats:
        lap_id = s["id"]
        if "Distance" in session.table.columns and segments:
            segment_stats_all[lap_id] = calculate_sector_stats(session.table, lap_id, segments)

    sector_table_width = 50 + 80 + 80 * n_sectors
    segment_table_width = 40 + 70 + 55 * n_segments
    table_width = max(sector_table_width, segment_table_width)
    img_w = map_size[0] + 50 + table_width + 50
    n_rows = len(lap_stats)
    img_h = max(map_size[1] + 100, 80 + n_rows * 22 + 30 + 200 + 30 + 20 + 30 + n_rows * 18 + 100)

    img = Image.new("RGB", (img_w, img_h), "#111111")
    draw = ImageDraw.Draw(img)

    fnt_title = get_font(20)
    draw.text((25, 15), "Lap Analysis Report", font=fnt_title, fill="#FFFFFF")

    venue = session.venue or track_dir.name
    date_str = session.event_date or ""
    driver = session.driver or ""
    info_text = f"Track: {venue}  |  Date: {date_str}  |  Driver: {driver}"
    draw.text((25, 45), info_text, font=get_font(14), fill="#888888")

    draw_track_map(draw, track, segments, segment_colors, origin=(25, 80), size=map_size)

    table_origin = (map_size[0] + 75, 80)
    table_height = draw_lap_table(draw, lap_stats, sector_splits, table_origin, split_names, best_lap_id)

    stats_origin = (map_size[0] + 75, 80 + table_height + 30)
    stats_height = draw_statistics(draw, lap_stats, sector_splits, stats_origin, split_names, session.table)

    if segments:
        seg_table_origin = (map_size[0] + 75, 80 + table_height + 30 + stats_height + 20)
        draw_segment_table(draw, lap_stats, segment_stats_all, segments, seg_table_origin, best_lap_id)

    legend_y = 80 + map_size[1] + 20
    fnt_legend = get_font(12)
    draw.text((25, legend_y), "Sector Colors:", font=fnt_legend, fill="#AAAAAA")
    legend_x = 130
    for idx in range(min(len(segments), len(segment_colors))):
        color = segment_colors[idx]
        draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 15], fill=color, outline="#000000")
        draw.text((legend_x + 25, legend_y + 7), f"Seg{idx + 1}", font=fnt_legend, fill="#CCCCCC", anchor="lm")
        legend_x += 60

    print(f"Saving report to {output_path}...")
    img.save(output_path)
    print(f"Done! Report saved to {output_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate PNG report with track map and lap statistics")
    p.add_argument("--telemetry", required=True, help="Path to telemetry file (.xrk, .csv, etc.)")
    p.add_argument("--track", dest="track_dir", required=True, help="Path to track directory")
    p.add_argument("--out", default="report.png", help="Output PNG path")
    args = p.parse_args()

    generate_report(
        telemetry_path=Path(args.telemetry),
        track_dir=Path(args.track_dir),
        output_path=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    exit(main())
