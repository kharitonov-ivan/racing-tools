"""ASS subtitle generation utilities for video overlays."""

import atexit
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from racing_tools.session.session import VideoSession

from racing_tools.session.predictive import PredictiveLapModel
from racing_tools.track.constants import DEFAULT_MAX_RPM, MAX_DELTA_FOR_DISPLAY, MIN_VALID_LAP_TIME
from racing_tools.video.overlay import format_duration


class AssBuilder:
    """Accumulates ASS styles and events, writes a single unified .ass file."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.styles: list[str] = []
        self.events: list[str] = []
        self._style_names: set[str] = set()

    def add_style(self, style_line: str) -> None:
        """Add a Style: line (without 'Style: ' prefix duplication check)."""
        # Extract name from "Style: Name,..." format
        name = style_line.split(",")[0].replace("Style: ", "").strip()
        if name not in self._style_names:
            self._style_names.add(name)
            self.styles.append(style_line)

    def add_event(self, event_line: str) -> None:
        """Add a Dialogue: line."""
        self.events.append(event_line)

    def write(self, output_path: str | Path | None = None) -> str:
        """Write unified ASS file and return path.

        Args:
            output_path: If provided, save ASS to this path (persistent).
                         Otherwise, use a temp file that auto-deletes on exit.
        """
        if output_path is not None:
            path = str(output_path)
        else:
            fd, path = tempfile.mkstemp(suffix="_unified.ass")
            os.close(fd)

            def cleanup() -> None:
                if os.path.exists(path):
                    os.unlink(path)

            atexit.register(cleanup)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"[Script Info]\nScriptType: v4.00+\nPlayResX: {self.width}\nPlayResY: {self.height}\n\n")
            f.write("[V4+ Styles]\n")
            f.write(
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            )
            for s in self.styles:
                f.write(s + "\n")
            f.write("\n[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            for e in self.events:
                f.write(e + "\n")

        print(f"[ASS] Unified file: {len(self.styles)} styles, {len(self.events)} events -> {path}")
        return path

    def write_with_offset(self, output_path: Path, time_offset: float) -> None:
        """Write ASS file to specified path with time offset applied to all events."""

        def parse_ass_time(ts: str) -> float:
            parts = ts.split(":")
            h = int(parts[0])
            m = int(parts[1])
            s_cs = parts[2].split(".")
            s = int(s_cs[0])
            cs = int(s_cs[1]) if len(s_cs) > 1 else 0
            return h * 3600 + m * 60 + s + cs / 100.0

        def offset_event(event: str, offset: float) -> str | None:
            match = re.match(r"(Dialogue:\s*\d+,)(\d+:\d+:\d+\.\d+),(\d+:\d+:\d+\.\d+),(.*)", event)
            if not match:
                return event
            prefix, start, end, rest = match.groups()
            new_end_t = parse_ass_time(end) + offset
            if new_end_t <= 0:
                return None
            new_start_t = max(0.0, parse_ass_time(start) + offset)
            return f"{prefix}{fmt_ass_time(new_start_t)},{fmt_ass_time(new_end_t)},{rest}"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"[Script Info]\nScriptType: v4.00+\nPlayResX: {self.width}\nPlayResY: {self.height}\n\n")
            f.write("[V4+ Styles]\n")
            f.write(
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            )
            for s in self.styles:
                f.write(s + "\n")
            f.write("\n[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            for e in self.events:
                shifted = offset_event(e, time_offset)
                if shifted is not None:
                    f.write(shifted + "\n")

        print(f"[ASS] Saved with offset {time_offset:.2f}s: {output_path}")


def fmt_ass_time(t: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc with millisecond precision."""
    h = int(t / 3600)
    m = int((t % 3600) / 60)
    s = int(t % 60)
    # Use millisecond precision for better timing at high fps
    ms = int((t * 1000) % 1000)
    return f"{h}:{m:02d}:{s:02d}.{ms // 10:02d}"


def emit_lap_stats_ass(ass: AssBuilder, session: "VideoSession") -> None:
    """Emit lap list styles and events into AssBuilder."""
    lap_stats = session.get_lap_stats()
    crossings = session.crossings
    best_lap = session.best_lap
    total_duration = session.info.duration
    width = session.info.width

    if not lap_stats:
        return

    scale_h = session.info.height / 1080.0
    s20 = max(12, int(20 * scale_h))
    s24 = max(14, int(24 * scale_h))

    # TODO: Use a narrow/condensed font (e.g., "Arial Narrow", "Roboto Condensed", "Impact Condensed")
    # to save horizontal space and fit more content. Test readability on video output.
    # Styles
    ass.add_style(f"Style: Header,Arial,{s20},&H00AAAAAA,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1")
    ass.add_style(f"Style: Row,Arial,{s24},&H00FFFFFF,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1")
    ass.add_style(f"Style: RowGold,Arial,{s24},&H0000D7FF,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1")
    ass.add_style(f"Style: RowPit,Arial,{s24},&H00887766,&H000000FF,&H00000000,&H60000000,0,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1")
    # Top-3 lap styles (ASS uses BGR colors)
    ass.add_style(f"Style: RowRed,Arial,{s24},&H000000FF,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1")  # Best (red)
    ass.add_style(f"Style: RowBlue,Arial,{s24},&H00FF8000,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1")  # 2nd (blue)
    ass.add_style(f"Style: RowGreen,Arial,{s24},&H0000FFFF,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,1,1,1,7,10,10,10,1")  # 3rd (green)

    # Get sector splits
    sector_splits = session.get_sector_splits()
    split_names: list[str] = []
    for lap_splits in sector_splits.values():
        split_names = list(lap_splits.keys())
        break

    sample = lap_stats[0]
    columns: list[tuple[str, str, int]] = [("Lap", "id", 60), ("Video LT", "time", 120)]
    if any(s.get("gps_time") is not None for s in lap_stats):
        columns.append(("GPS LT", "gps_time", 120))
    for name in split_names:
        columns.append((name, f"_split_{name}", 100))
    for key, value in sample.items():
        if key in ("id", "time", "gps_time", "label"):
            continue
        if value is not None:
            columns.append((key.replace("_", " ").title(), key, 100))

    margin_right = int(50 * scale_h)
    col_gap = int(20 * scale_h)

    # Scale column widths
    scaled_columns = [(name, key, int(w * scale_h)) for name, key, w in columns]

    total_width = sum(c[2] for c in scaled_columns) + col_gap * (len(scaled_columns) - 1)
    base_x = width - total_width - margin_right
    start_y = int(50 * scale_h)
    # TODO: Reduce row height (currently 40 * scale_h) to fit more laps on screen
    row_h = int(40 * scale_h)

    col_positions: list[int] = []
    x = 0
    for _, _, w in scaled_columns:
        col_positions.append(x)
        x += w + col_gap

    best_lap_id = best_lap["id"] if best_lap else -1
    sorted_stats = sorted(lap_stats, key=lambda s: s["id"])

    # Find top-3 laps for coloring
    valid_lap_times_table = [(s["id"], s.get("time") or 0) for s in lap_stats if (s.get("time") or 0) > MIN_VALID_LAP_TIME]
    top3_sorted_table = sorted(valid_lap_times_table, key=lambda x: x[1])[:3]
    top3_ids_table = [lap_id for lap_id, _ in top3_sorted_table]
    top3_rank_table = {lap_id: i + 1 for i, lap_id in enumerate(top3_ids_table)}

    # Header row (static, full duration)
    for i, (header, _, _) in enumerate(scaled_columns):
        x_pos = base_x + col_positions[i]
        ass.add_event(f"Dialogue: 0,0:00:00.00,99:59:59.99,Header,,0,0,0,,{{\\pos({x_pos},{start_y})}}{header}")

    data_start_y = start_y + int(40 * scale_h)

    # Data rows (static)
    for row_idx, s in enumerate(sorted_stats):
        y_pos = data_start_y + row_idx * row_h
        label = s.get("label")
        is_pit = label in ("POUT", "PIN")
        lap_id = s["id"]

        # Determine style: pit -> top3 color -> gold (best) -> default
        if is_pit:
            style = "RowPit"
        elif lap_id in top3_rank_table:
            rank = top3_rank_table[lap_id]
            style = ["RowRed", "RowBlue", "RowGreen"][rank - 1]
        elif lap_id == best_lap_id:
            style = "RowGold"
        else:
            style = "Row"
        common = f"Dialogue: 0,0:00:00.00,99:59:59.99,{style},,0,0,0,,"

        for col_idx, (_, key, _) in enumerate(scaled_columns):
            x_pos = base_x + col_positions[col_idx]
            if key.startswith("_split_"):
                split_name = key[7:]  # strip "_split_" prefix
                splits = sector_splits.get(lap_id, {})
                value = splits.get(split_name)
            else:
                value = s.get(key)
            if key == "id":
                text = label if label else (f"{value}*" if value == best_lap_id else str(value))
            elif key in ("time", "gps_time"):
                text = format_duration(value, decimals=3) if value else "-"
            elif key.startswith("_split_"):
                if is_pit or value is None:
                    text = ""
                else:
                    text = f"{value:.2f}"
            elif is_pit:
                text = ""
            elif value is None:
                text = "-"
            elif isinstance(value, float):
                text = f"{value:.1f}"
            else:
                text = str(int(value))
            if text:
                ass.add_event(f"{common}{{\\pos({x_pos},{y_pos})}}{text}")

    # Dynamic pointer >
    times = [0.0] + list(crossings) + [total_duration]
    for i in range(len(times) - 1):
        range_start, range_end = times[i], times[i + 1]
        if range_end <= range_start:
            continue
        idx = next((j for j, s in enumerate(sorted_stats) if s["id"] == i), -1)
        if idx != -1:
            y_pos = data_start_y + idx * row_h
            ptr_x = base_x - int(30 * scale_h)
            ass.add_event(f"Dialogue: 1,{fmt_ass_time(range_start)},{fmt_ass_time(range_end)},Row,,0,0,0,,{{\\pos({ptr_x},{y_pos})}}>")


def emit_gauge_ass(ass: AssBuilder, session: "VideoSession") -> None:
    """Emit gauge (speed/RPM/delta/timer) events into AssBuilder."""
    fps = session.info.fps
    lap_stats = session.get_lap_stats()
    best_lap = session.best_lap
    session_table = session.table

    # Find speed column
    speed_col = next((n for n in ["Speed", "GPS Speed", "Wheel Speed"] if n in session_table.columns), None)
    if speed_col is None:
        print(f"[gauge] No speed column found in {list(session_table.columns)}, skipping")
        return

    # Find RPM column
    rpm_col = next((n for n in ["RPM", "Engine RPM"] if n in session_table.columns), None)

    # Find GPS columns
    lat_col = next((n for n in ["GPS Latitude", "Latitude", "Lat."] if n in session_table.columns), None)
    lon_col = next((n for n in ["GPS Longitude", "Longitude", "Lon."] if n in session_table.columns), None)

    # Styles
    scale_h = session.info.height / 1080.0
    scale_h = session.info.height / 1080.0
    s24 = max(14, int(24 * scale_h))
    s36 = max(18, int(36 * scale_h))
    s40 = max(20, int(40 * scale_h))
    s48 = max(24, int(48 * scale_h))
    s60 = max(30, int(60 * scale_h))

    ass.add_style(f"Style: Gauge,Arial,{s48},&H00FFFFFF,&H000000FF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,3,0,0,2,10,10,50,1")
    ass.add_style(f"Style: LapTimer,Arial,{s36},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,1,10,10,100,1")

    # Alfano-style delta styles (ASS uses BGR colors) - larger fonts
    ass.add_style(f"Style: DeltaLine,Arial,{s48},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,8,10,10,10,1")
    ass.add_style(f"Style: DeltaTop1,Arial,{s60},&H000000FF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,8,10,10,10,1")  # Red (best)
    ass.add_style(f"Style: DeltaTop2,Arial,{s60},&H00FF8000,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,8,10,10,10,1")  # Blue (2nd)
    ass.add_style(f"Style: DeltaTop3,Arial,{s60},&H0000FFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,8,10,10,10,1")  # Green (3rd)
    ass.add_style(
        f"Style: DeltaGray,Arial,{s40},&H00888888,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,8,10,10,10,1"
    )  # Gray (other laps)
    ass.add_style(
        f"Style: DeltaCurrent,Arial,{s60},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,8,10,10,10,1"
    )  # White (current)
    ass.add_style(f"Style: DeltaLabel,Arial,{s36},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,8,10,10,10,1")
    ass.add_style(
        f"Style: DeltaLapNum,Arial,{s24},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,0,8,10,10,10,1"
    )  # Smaller font for lap numbers

    # Build predictive models for ALL laps (Alfano-style)
    all_lap_models: dict[int, PredictiveLapModel] = {}
    lap_start_distances: dict[int, float] = {}

    width = session.info.width
    height = session.info.height

    print(f"[gauge] Building predictive models for all laps: lap_stats={len(lap_stats) if lap_stats else 0}")

    if lap_stats and session_table is not None and "Distance" in session_table.columns and "LapNumber" in session_table.columns:
        lap_start_distances = session_table.groupby("LapNumber")["Distance"].min().to_dict()

        for stat in lap_stats:
            lap_id = stat["id"]
            if lap_id == 0 or (stat.get("time") or 0) < MIN_VALID_LAP_TIME:
                continue
            mask = session_table["LapNumber"] == lap_id
            lap_df = session_table[mask].copy()
            if lap_df.empty:
                continue
            if "LapTime" not in lap_df.columns:
                continue
            start_dist = lap_df["Distance"].min()
            lap_dists = (lap_df["Distance"] - start_dist).values
            lap_times = lap_df["LapTime"].values
            valid = ~np.isnan(lap_dists) & ~np.isnan(lap_times)
            if valid.sum() > 10:
                all_lap_models[lap_id] = PredictiveLapModel(list(zip(lap_dists[valid], lap_times[valid])))

        print(f"[gauge] Built {len(all_lap_models)} predictive models for laps: {list(all_lap_models.keys())}")

    # Find top-3 laps
    valid_lap_times = [(s["id"], s["time"]) for s in lap_stats if (s.get("time") or 0) > MIN_VALID_LAP_TIME]
    top3_sorted = sorted(valid_lap_times, key=lambda x: x[1])[:3]
    top3_ids = [lap_id for lap_id, _ in top3_sorted]
    top3_rank = {lap_id: i + 1 for i, lap_id in enumerate(top3_ids)}
    print(f"[gauge] Top-3 laps: {[(lid, f'{t:.3f}s') for lid, t in top3_sorted]}")

    # Alfano delta parameters
    bar_width = min(int(3000 * scale_h), int(width * 0.9))
    scale_sec = 3.0
    center_x = width // 2
    line_y = height * 2 // 3  # Lower position (bottom third of screen)
    cars_y = line_y + int(50 * scale_h)
    lap_num_y = cars_y - int(50 * scale_h)  # Lap number above the bar (smaller font)
    label_y = cars_y + int(40 * scale_h)

    def delta_to_x(delta: float) -> float:
        clamped = max(-scale_sec, min(scale_sec, delta))
        return center_x + (clamped / scale_sec) * (bar_width / 2)

    speeds = session_table[speed_col].fillna(0).values
    rpms = session_table[rpm_col].fillna(0).values if rpm_col else np.zeros(len(session_table))

    GAUGE_FPS, alpha = 10, 0.3
    filtered_speeds, filtered_rpms = speeds.copy(), rpms.copy()
    for i in range(1, len(speeds)):
        filtered_speeds[i] = alpha * speeds[i] + (1 - alpha) * filtered_speeds[i - 1]
        filtered_rpms[i] = alpha * rpms[i] + (1 - alpha) * filtered_rpms[i - 1]

    # Debug: check if values vary
    print(f"[gauge] speed_col={speed_col}, rpm_col={rpm_col}")
    print(f"[gauge] speeds shape={speeds.shape}, rpms shape={rpms.shape}")
    print(f"[gauge] speeds first 5: {speeds[:5]}")
    print(f"[gauge] speeds last 5: {speeds[-5:]}")
    print(f"[gauge] speeds min/max: {speeds.min():.1f}/{speeds.max():.1f}")

    if lat_col and lon_col:
        lats = session_table[lat_col].values
        lons = session_table[lon_col].values
    else:
        lats = None
        lons = None

    max_rpm = DEFAULT_MAX_RPM
    if rpm_col:
        m = session_table[rpm_col].max()
        if not np.isnan(m):
            max_rpm = int(np.ceil(m / 1000) * 1000)

    total_frames = len(session_table)
    display_crossings: list[float] = getattr(session, "crossings", []) or []

    def fmt_lap_time(t: float) -> str:
        m = int(t / 60)
        s = t - m * 60
        return f"{m}:{s:06.3f}"

    interval = max(1, round(fps / GAUGE_FPS))
    for i in range(0, total_frames, interval):
        t_start = i / fps
        t_end = min((i + interval) / fps, total_frames / fps)
        s_str = fmt_ass_time(t_start)
        e_str = fmt_ass_time(t_end)

        speed = filtered_speeds[i]
        rpm = filtered_rpms[i]

        # RPM color
        if rpm < 7500:
            rpm_color = "&H000000FF"
        elif rpm < 8500:
            rpm_color = "&H0000FFFF"
        else:
            rpm_color = "&H0000FF00"

        bar_len = 20
        filled = max(0, min(bar_len, int((rpm / max_rpm) * bar_len)))
        bar_text = "[" + "|" * filled + " " * (bar_len - filled) + "]"

        text = f"{{\\c&H00FFFFFF&}}{int(round(speed))} km/h   {{\\c{rpm_color}&}}{bar_text}   {int(round(rpm))} RPM"
        ass.add_event(f"Dialogue: 0,{s_str},{e_str},Gauge,,0,0,0,,{text}")

        # Lap timer
        lap_elapsed = t_start
        if display_crossings:
            last_crossing = 0.0
            for cx in display_crossings:
                if cx <= t_start:
                    last_crossing = cx
                else:
                    break
            lap_elapsed = t_start - last_crossing
        timer_text = f"Lap {fmt_lap_time(lap_elapsed)}  |  Frame {i}"
        if lats is not None:
            timer_text += f"  |  {lats[i]:.6f}, {lons[i]:.6f}"
        ass.add_event(f"Dialogue: 1,{s_str},{e_str},LapTimer,,0,0,0,,{timer_text}")

        # Alfano-style Delta (all laps on horizontal line)
        if all_lap_models and "Distance" in session_table.columns and "LapNumber" in session_table.columns and "LapTime" in session_table.columns:
            dists_arr = session_table["Distance"].values
            laps_arr = session_table["LapNumber"].values
            l_times_arr = session_table["LapTime"].values

            try:
                curr_lap = laps_arr[i]
                if curr_lap in lap_start_distances:
                    curr_lap_dist = dists_arr[i] - lap_start_distances[curr_lap]
                    if curr_lap_dist >= 0:
                        # Central line
                        ass.add_event(f"Dialogue: 2,{s_str},{e_str},DeltaLine,,0,0,0,,{{\\pos({center_x},{line_y})}}──────|──────")

                        # Current lap marker (always center)
                        ass.add_event(f"Dialogue: 3,{s_str},{e_str},DeltaCurrent,,0,0,0,,{{\\pos({center_x},{cars_y})}}▲")

                        # All other laps
                        for lap_id, model in all_lap_models.items():
                            if lap_id == curr_lap:
                                continue

                            try:
                                pred_time = model.get_time(curr_lap_dist)
                                delta = l_times_arr[i] - pred_time

                                if abs(delta) > MAX_DELTA_FOR_DISPLAY:
                                    continue

                                x = delta_to_x(delta)

                                if lap_id in top3_rank:
                                    rank = top3_rank[lap_id]
                                    style = f"DeltaTop{rank}"
                                    ass.add_event(f"Dialogue: 5,{s_str},{e_str},{style},,0,0,0,,{{\\pos({x},{cars_y})}}|")
                                    ass.add_event(f"Dialogue: 6,{s_str},{e_str},DeltaLapNum,,0,0,0,,{{\\pos({x},{lap_num_y})}}L{lap_id}")
                                    ass.add_event(f"Dialogue: 7,{s_str},{e_str},{style},,0,0,0,,{{\\pos({x},{label_y})}}{delta:+.2f}")
                                else:
                                    ass.add_event(f"Dialogue: 8,{s_str},{e_str},DeltaGray,,0,0,0,,{{\\pos({x},{cars_y})}}|")
                                    ass.add_event(f"Dialogue: 9,{s_str},{e_str},DeltaLapNum,,0,0,0,,{{\\pos({x},{lap_num_y})}}L{lap_id}")
                            except Exception:
                                pass
            except Exception:
                pass
