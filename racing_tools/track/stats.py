"""Track segment and sector statistics: speed min/max calculation and rendering."""

import numpy as np
from PIL import ImageFont

from racing_tools.session.session import WGS84_TO_WEBMERC


def calculate_segment_stats(session_table, lap_id: int, raw_segments: list) -> dict:
    """Calculate min/max speed for each segment for a specific lap."""
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
            if np.max(np.abs(pts[:, 0])) <= 180 and np.max(np.abs(pts[:, 1])) <= 90:
                xs, ys = WGS84_TO_WEBMERC.transform(pts[:, 0], pts[:, 1])
                projected_pts = np.column_stack((xs, ys))
            else:
                projected_pts = pts

            diffs = projected_pts[1:] - projected_pts[:-1]
            dists = np.linalg.norm(diffs, axis=1)
            seg_len = np.sum(dists)

        start_d = current_dist
        end_d = current_dist + seg_len
        current_dist += seg_len

        mask = (lap_data["Distance"] >= (lap_start_dist + start_d)) & (lap_data["Distance"] < (lap_start_dist + end_d))

        seg_samples = lap_data[mask]

        if not seg_samples.empty:
            spd = seg_samples[speed_col]
            s_min, s_max = spd.min(), spd.max()
            stats[idx] = (s_min, s_max)

    print(f"[stats] Calculated stats for {len(stats)}/{len(raw_segments)} segments.")
    return stats


def draw_full_track_stats(draw, drawing_area: tuple, track_overlay_data: dict) -> None:
    """Custom drawer to include stats for BOTH straights and turns."""
    inner_x, inner_y, width, height = drawing_area
    segments = track_overlay_data.get("segments")

    if not segments:
        return

    stats_by_lap = track_overlay_data.get("segment_stats", {})

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

        stats = stats_by_lap.get(seg_idx)
        if not stats:
            continue

        min_spd, max_spd = stats
        mid_idx = len(scaled) // 2
        mid_pt = scaled[mid_idx]
        txt = f"{int(min_spd)}/{int(max_spd)}"

        try:
            fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        except Exception:
            fnt = ImageFont.load_default()

        x, y = mid_pt[0], mid_pt[1]

        for off in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((x + off[0], y + off[1]), txt, font=fnt, fill="black", anchor="mm")

        draw.text((x, y), txt, font=fnt, fill="#ffffff", anchor="mm")


def calculate_sector_stats_for_lap(session_table, lap_id: int, sectors: list) -> dict:
    """
    Calculate min/max speed for each sector for a specific lap.

    Computes segment distances from points if start_dist/end_dist not present.
    """
    stats = {}

    speed_col = next((c for c in ("GPS Speed", "Speed", "Wheel Speed") if c in session_table.columns), None)
    if "Distance" not in session_table.columns or speed_col is None:
        print("[sector-stats] Missing Distance or Speed columns")
        return stats

    lap_data = session_table[session_table["LapNumber"] == lap_id]
    if lap_data.empty:
        print(f"[sector-stats] No data for lap {lap_id}")
        return stats

    lap_start_dist = lap_data["Distance"].min()

    cumulative_dist = 0.0

    for idx, sector in enumerate(sectors):
        if "start_dist" in sector and "end_dist" in sector:
            start_d = sector["start_dist"]
            end_d = sector["end_dist"]
        else:
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
            cumulative_dist = end_d

        mask = (lap_data["Distance"] >= (lap_start_dist + start_d)) & (lap_data["Distance"] < (lap_start_dist + end_d))

        seg_samples = lap_data[mask]

        if not seg_samples.empty:
            spd = seg_samples[speed_col]
            stats[idx] = (spd.min(), spd.max())
            if idx < 3:
                print(
                    f"[sector-stats] Sector {idx}: dist {start_d:.0f}-{end_d:.0f}m, "
                    f"matched {len(seg_samples)} samples, speed {spd.min():.0f}-{spd.max():.0f}"
                )
        else:
            if idx < 3:
                print(f"[sector-stats] Sector {idx}: dist {start_d:.0f}-{end_d:.0f}m, NO SAMPLES")

    print(f"[sector-stats] Lap {lap_id}: calculated stats for {len(stats)}/{len(sectors)} sectors")
    return stats


def draw_sectors_with_stats(draw, drawing_area: tuple, sectors: list, sector_stats: dict) -> None:
    """Draw sector speed stats on track map."""
    inner_x, inner_y, width, height = drawing_area

    for idx, sector in enumerate(sectors):
        points = sector.get("points", [])
        if not points:
            continue

        scaled = [
            (
                inner_x + max(0.0, min(1.0, pt[0])) * width,
                inner_y + max(0.0, min(1.0, pt[1])) * height,
            )
            for pt in points
        ]

        stats = sector_stats.get(idx)
        if not (stats and scaled):
            continue

        min_spd, max_spd = stats
        mid_idx = len(scaled) // 2
        mid_pt = scaled[mid_idx]
        txt = f"{int(min_spd)}/{int(max_spd)}"

        try:
            fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        except Exception:
            fnt = ImageFont.load_default()

        x, y = mid_pt[0], mid_pt[1]

        for off in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((x + off[0], y + off[1]), txt, font=fnt, fill="black", anchor="mm")

        draw.text((x, y), txt, font=fnt, fill="#ffffff", anchor="mm")
