#!/usr/bin/env python
"""
Generate bestline (optimal racing line) from a single telemetry session.

Loads one telemetry session, finds the fastest lap, extracts GPS
coordinates, smooths the start/finish junction, and saves as
bestline.geojson. Computes intersection points with all sector lines.

Usage:
    python -m racing_tools.track.generate_bestline_from_telemetry \
        --track racing_tools/track/data/RIMSportKarting \
        --session /path/to/session.zip
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from racing_tools.session.session import Session
from racing_tools.track.track import Track
from pyproj import Transformer


def _extract_best_lap_gps(session: Session) -> dict | None:
    lat_col = session._pick_column(["GPS Latitude", "Latitude"])
    lon_col = session._pick_column(["GPS Longitude", "Longitude"])
    if not lat_col or not lon_col:
        print("No GPS columns found, skipping")
        return None

    best = session.best_lap
    if best is None:
        print("No valid laps found, skipping")
        return None

    lap_id = best["id"]
    lap_time = best["time"]
    print(f"Best lap: #{lap_id} ({lap_time:.3f}s)")

    lap_mask = session.table["LapNumber"] == lap_id
    lap_data = session.table.loc[lap_mask]
    if lap_data.empty:
        print(f"No data for lap {lap_id}")
        return None

    lons = pd.to_numeric(lap_data[lon_col], errors="coerce").values
    lats = pd.to_numeric(lap_data[lat_col], errors="coerce").values

    alt_col = session._pick_column(["Altitude", "GPS Altitude", "Alt"])
    alts = pd.to_numeric(lap_data[alt_col], errors="coerce").values if alt_col else np.zeros_like(lons)

    valid = ~(np.isnan(lons) | np.isnan(lats))
    lons, lats, alts = lons[valid], lats[valid], alts[valid]
    alts = np.nan_to_num(alts, nan=0.0)
    if len(lons) < 10:
        print(f"Too few valid GPS points ({len(lons)})")
        return None

    print(f"Extracted {len(lons)} GPS points from lap {lap_id}" + (f" (alt {alts.min():.0f}-{alts.max():.0f}m)" if alt_col else ""))
    return {"lap_id": lap_id, "lap_time": lap_time, "lons": lons, "lats": lats, "alts": alts}


def _smooth_sf_junction(bestline_utm: np.ndarray, smooth_radius_m: float = 30.0) -> np.ndarray:
    """Smooth the bestline at the start/finish line junction.

    The lap GPS data starts and ends at SF, creating two slightly different
    paths. This averages the "approaching" and "leaving" trajectories,
    then resamples the SF zone uniformly to eliminate spacing artifacts.
    """
    from scipy.interpolate import CubicSpline

    pts = bestline_utm.copy()
    n = len(pts)

    # Compute spacing
    total_length = sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, n))
    avg_spacing = total_length / n
    n_blend = max(4, int(smooth_radius_m / avg_spacing))
    n_blend = min(n_blend, n // 4)

    # Average mirror points near SF
    for k in range(n_blend):
        w = 1.0 - k / n_blend
        w = w * w

        idx_start = k
        idx_end = -(k + 1)

        avg = (pts[idx_start] + pts[idx_end]) / 2
        pts[idx_start] = pts[idx_start] * (1 - w) + avg * w
        pts[idx_end] = pts[idx_end] * (1 - w) + avg * w

    # Fix artifacts: remove reversals and resample SF zone uniformly
    # 1. Fix 180° reversals from resample seam
    for _ in range(3):
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            if np.linalg.norm(v1) < 0.01 or np.linalg.norm(v2) < 0.01:
                continue
            dh = abs((np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])) + 180) % 360 - 180)
            if dh > 90:
                pts[i] = (pts[i - 1] + pts[i + 1]) / 2

    # 2. Resample SF zone to fix uneven spacing from mirror averaging
    n_zone = n_blend + 2
    zone_idx = list(range(n - n_zone, n)) + list(range(n_zone))
    zone_pts = pts[zone_idx]
    arc = np.zeros(len(zone_pts))
    for i in range(1, len(zone_pts)):
        arc[i] = arc[i - 1] + np.linalg.norm(zone_pts[i] - zone_pts[i - 1])
    if arc[-1] > 0:
        s_new = np.linspace(0, arc[-1], len(zone_pts))
        for i, idx in enumerate(zone_idx):
            pts[idx, 0] = np.interp(s_new[i], arc, zone_pts[:, 0])
            pts[idx, 1] = np.interp(s_new[i], arc, zone_pts[:, 1])

    # Remove closing duplicate if present (save_bestline adds it back)
    if np.allclose(pts[0], pts[-1], atol=0.01):
        pts = pts[:-1]

    return pts


def _compute_sector_intersections(
    bestline_utm: list[tuple],
    sectors_utm: dict[str, list[tuple]],
    transformer_to_wgs84: Transformer,
) -> list[tuple[str, float, float, float]]:
    """Find intersection of bestline with each sector line.

    Returns list of (name, distance_m, lat, lon) tuples.
    """
    bestline_line = LineString(bestline_utm)
    results = []

    for name, sector_pts in sectors_utm.items():
        sector_line = LineString(sector_pts)
        intersection = sector_line.intersection(bestline_line)

        if intersection.is_empty:
            # Closest point fallback
            sector_mid = sector_line.interpolate(0.5, normalized=True)
            dist = bestline_line.project(sector_mid)
            proj_pt = bestline_line.interpolate(dist)
        elif intersection.geom_type == "Point":
            dist = bestline_line.project(intersection)
            proj_pt = intersection
        else:
            pt = Point(intersection.coords[0]) if hasattr(intersection, 'coords') else intersection.geoms[0]
            dist = bestline_line.project(pt)
            proj_pt = pt

        lon, lat = transformer_to_wgs84.transform(proj_pt.x, proj_pt.y)
        results.append((name, dist, lat, lon))

    results.sort(key=lambda x: x[1])
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate bestline from a single telemetry session",
    )
    parser.add_argument(
        "--track",
        required=True,
        type=Path,
        help="Path to track directory (e.g. racing_tools/track/data/RIMSportKarting)",
    )
    parser.add_argument(
        "--session",
        required=True,
        type=Path,
        help="Path to telemetry session folder or file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of resampled points (default: 512)",
    )
    parser.add_argument(
        "--smooth-radius",
        type=float,
        default=60.0,
        help="Smoothing radius around SF junction in meters (default: 60)",
    )
    args = parser.parse_args(argv)

    track_dir = args.track.resolve()
    if not track_dir.is_dir():
        print(f"Error: track directory not found: {track_dir}")
        sys.exit(1)

    session_path = args.session.resolve()
    if not session_path.exists():
        print(f"Error: session path not found: {session_path}")
        sys.exit(1)

    print(f"[Track] Loading from {track_dir}")
    track = Track.load(track_dir)
    geometry_dir = track_dir / "geometry"

    print(f"[Track] Sectors: {list(track.sectors_utm.keys())}")

    print(f"\n[Session] Loading {session_path.name}...")
    try:
        session = Session.load(session_path)
    except Exception as e:
        print(f"Failed to load session: {e}")
        sys.exit(1)

    print(f"[Session] {len(session.table)} rows, {session.driver or 'unknown driver'}")

    session.track = track
    try:
        session.compute_heading()
        crossings = session.detect_crossings()
    except Exception as e:
        print(f"Crossing detection failed: {e}")
        sys.exit(1)

    if not crossings:
        print("No crossings detected, skipping")
        sys.exit(1)

    print(f"[Session] {len(crossings)} crossings detected")
    session.add_lap_numbers()

    result = _extract_best_lap_gps(session)
    if not result:
        print("Error: no valid best lap found")
        sys.exit(1)

    print(f"\n[Best] Lap #{result['lap_id']} ({result['lap_time']:.3f}s, {len(result['lons'])} GPS points)")

    track.set_bestline_from_gps(result["lons"], result["lats"], alts=result.get("alts"), n_samples=args.samples)

    # Smooth SF junction (spline bridge)
    sf_utm = track.sectors_utm.get("SF")
    if sf_utm and track.bestline_utm:
        bestline_arr = np.array(track.bestline_utm)
        smoothed = _smooth_sf_junction(bestline_arr, smooth_radius_m=args.smooth_radius)
        track.bestline_utm = list(map(tuple, smoothed))
        print(f"[Bestline] Smoothed SF junction (radius={args.smooth_radius}m)")


    track.save_bestline(geometry_dir)
    bestline_length = LineString(track.bestline_utm).length
    print(f"[Bestline] Saved to {geometry_dir / 'bestline.geojson'}")
    print(f"[Bestline] Length: {bestline_length:.1f}m")

    # Compute sector intersections
    transformer_to_wgs84 = Transformer.from_crs(track.utm_zone, "EPSG:4326", always_xy=True)

    if track.sectors_utm:
        intersections = _compute_sector_intersections(
            track.bestline_utm, track.sectors_utm, transformer_to_wgs84,
        )
        prev_dist = 0.0
        for name, dist, lat, lon in intersections:
            sector_len = dist - prev_dist
            print(f"[{name}] bestline {dist:.1f}m (segment: {sector_len:.1f}m) lat={lat:.6f}, lon={lon:.6f}")
            prev_dist = dist
        final_len = bestline_length - prev_dist
        print(f"[Final] {prev_dist:.1f}m -> {bestline_length:.1f}m (segment: {final_len:.1f}m)")

    track.save_config(track_dir)
    print(f"[Config] Saved to {track_dir / 'track_config.json'}")

    export_dir = geometry_dir / "export"
    track.export_gpx(export_dir)
    track.export_kml(export_dir / "track.kml")
    track.export_ztracks(export_dir / f"{track_dir.name}.ztracks", venue_name=track.name)


if __name__ == "__main__":
    main()
