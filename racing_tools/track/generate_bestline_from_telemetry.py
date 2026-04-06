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


def _build_bestline(
    gps_utm: np.ndarray,
    sf_utm: list[tuple],
    n_samples: int = 512,
    alts: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float] | None]:
    """Build bestline from raw GPS data, starting exactly at SF.

    1. Find where raw GPS crosses the SF line
    2. Trim data to start at that crossing
    3. Resample adaptively (more points in curves, fewer on straights)
    4. End trails off toward SF without forced closure

    Returns (bestline_utm, bestline_alt).
    """
    from scipy.signal import savgol_filter

    sf_line = LineString(sf_utm)

    # Find the GPS point closest to SF near the START of the lap
    # (not the end — both are close to SF but we want the beginning)
    dists_to_sf = [sf_line.distance(Point(gps_utm[i])) for i in range(min(50, len(gps_utm)))]
    start_idx = int(np.argmin(dists_to_sf))

    # Project that GPS point onto SF line to get the exact SF start point
    sf_pt = sf_line.interpolate(sf_line.project(Point(gps_utm[start_idx])))
    sf_xy = np.array([sf_pt.x, sf_pt.y])

    # Arc distance to the start point
    sf_dist = 0.0
    for i in range(1, start_idx + 1):
        sf_dist += np.linalg.norm(gps_utm[i] - gps_utm[i - 1])

    # Compute cumulative arc length of raw GPS
    n_raw = len(gps_utm)
    arc_raw = np.zeros(n_raw)
    for i in range(1, n_raw):
        arc_raw[i] = arc_raw[i - 1] + np.linalg.norm(gps_utm[i] - gps_utm[i - 1])
    total_raw = arc_raw[-1]

    # Find split index closest to SF crossing
    split = np.argmin(np.abs(arc_raw - sf_dist))

    # Rotate raw GPS: SF crossing point first, then the rest of the lap
    rotated = np.vstack([
        [sf_xy],
        gps_utm[split + 1:],
        gps_utm[1:split + 1],
    ])

    # Rotate altitude too
    if alts is not None and len(alts) == n_raw:
        sf_alt = np.interp(sf_dist, arc_raw, alts)
        rotated_alts = np.concatenate([
            [sf_alt],
            alts[split + 1:],
            alts[1:split + 1],
        ])
    else:
        rotated_alts = None

    # Compute curvature for adaptive resampling
    n_rot = len(rotated)
    arc_rot = np.zeros(n_rot)
    for i in range(1, n_rot):
        arc_rot[i] = arc_rot[i - 1] + np.linalg.norm(rotated[i] - rotated[i - 1])
    total_rot = arc_rot[-1]

    # Curvature: heading change per meter
    curvature = np.zeros(n_rot)
    for i in range(1, n_rot - 1):
        v1 = rotated[i] - rotated[i - 1]
        v2 = rotated[i + 1] - rotated[i]
        if np.linalg.norm(v1) < 0.01 or np.linalg.norm(v2) < 0.01:
            continue
        h1 = np.arctan2(v1[1], v1[0])
        h2 = np.arctan2(v2[1], v2[0])
        dh = abs((h2 - h1 + np.pi) % (2 * np.pi) - np.pi)
        seg_len = np.linalg.norm(v1)
        curvature[i] = dh / seg_len

    # Smooth curvature
    if n_rot > 15:
        window = min(15, n_rot // 4 * 2 + 1)
        if window >= 5 and window % 2 == 1:
            curvature = savgol_filter(curvature, window, polyorder=2, mode="nearest")
            curvature = np.maximum(curvature, 0)

    # Build adaptive density: more points where curvature is high
    # density = 1 + curvature_weight * normalized_curvature
    curv_max = np.percentile(curvature, 95) if curvature.max() > 0 else 1.0
    curv_norm = np.clip(curvature / max(curv_max, 1e-6), 0, 1)
    density = 1.0 + 3.0 * curv_norm  # 1x on straights, 4x in tight corners

    # Integrate density to get cumulative distribution
    density_interp = np.interp(np.linspace(0, total_rot, 2000),
                               arc_rot, density)
    cum_density = np.cumsum(density_interp)
    cum_density = cum_density / cum_density[-1]

    # Sample n_samples points according to density
    s_uniform = np.linspace(0, 1, n_samples)
    s_arc = np.interp(s_uniform, cum_density, np.linspace(0, total_rot, 2000))

    # Ensure first point is exactly at SF (arc=0)
    s_arc[0] = 0.0

    # Interpolate coordinates at sampled arc positions
    result = np.column_stack([
        np.interp(s_arc, arc_rot, rotated[:, 0]),
        np.interp(s_arc, arc_rot, rotated[:, 1]),
    ])

    # Smoothly pull the tail toward SF (pts[0])
    # Blend toward the extrapolated start direction (not straight to SF)
    tail_radius_m = 100.0
    tail_arc = np.zeros(len(result))
    for i in range(1, len(result)):
        tail_arc[i] = tail_arc[i - 1] + np.linalg.norm(result[i] - result[i - 1])
    total_result = tail_arc[-1]

    # Target: extrapolate backward from start of bestline
    dir_start = result[0] - result[1]
    dir_start = dir_start / np.linalg.norm(dir_start)

    for i in range(len(result) - 1, 0, -1):
        dist_from_end = total_result - tail_arc[i]
        if dist_from_end > tail_radius_m:
            break
        # How far from end (0=end, 1=anchor)
        t = dist_from_end / tail_radius_m
        # Smooth blend: cubic ease-in (slow start, fast end)
        w = 1.0 - t * t * t
        # Target point: SF + extrapolated offset based on distance from end
        target = result[0] + dir_start * dist_from_end
        result[i] = result[i] * (1 - w) + target * w

    # Interpolate altitude
    result_alt = None
    if rotated_alts is not None:
        alt_interp = np.interp(s_arc, arc_rot, rotated_alts)
        window = min(51, n_samples // 4 * 2 + 1)
        if window >= 5:
            alt_interp = savgol_filter(alt_interp, window, polyorder=2, mode="nearest")
        result_alt = alt_interp.tolist()

    return result, result_alt


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
        # SF is always at distance 0 (bestline starts from SF)
        if name == "SF":
            pt0 = bestline_utm[0]
            lon, lat = transformer_to_wgs84.transform(pt0[0], pt0[1])
            results.append((name, 0.0, lat, lon))
            continue

        sector_line = LineString(sector_pts)
        intersection = sector_line.intersection(bestline_line)

        if intersection.is_empty:
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

    # Build bestline: find SF crossing in raw GPS, adaptive resample
    sf_utm = track.sectors_utm.get("SF")
    transformer = track.get_transformer()
    lons, lats = result["lons"], result["lats"]
    xs, ys = transformer.transform(lons, lats)
    gps_utm = np.column_stack([xs, ys])

    if sf_utm:
        bestline_pts, bestline_alt = _build_bestline(
            gps_utm, sf_utm, n_samples=args.samples, alts=result.get("alts"),
        )
        track.bestline_utm = list(map(tuple, bestline_pts))
        track.bestline_alt = bestline_alt
        print(f"[Bestline] Built from SF crossing ({len(bestline_pts)} pts, adaptive resampling)")
    else:
        track.set_bestline_from_gps(lons, lats, alts=result.get("alts"), n_samples=args.samples)
        print(f"[Bestline] Resampled ({args.samples} pts, no SF available)")


    track.save_bestline(geometry_dir)
    bestline_length = LineString(track.bestline_utm).length
    print(f"[Bestline] Saved to {geometry_dir / 'bestline.geojson'}")
    print(f"[Bestline] Length: {bestline_length:.1f}m")

    # Compute sector intersections once and store on track object
    transformer_to_wgs84 = Transformer.from_crs(track.utm_zone, "EPSG:4326", always_xy=True)

    if track.sectors_utm:
        intersections = _compute_sector_intersections(
            track.bestline_utm, track.sectors_utm, transformer_to_wgs84,
        )
        # Store on track for all exporters to use
        track.sector_intersections = {
            name: (lat, lon, dist) for name, dist, lat, lon in intersections
        }

        prev_dist = 0.0
        for name, dist, lat, lon in intersections:
            sector_len = dist - prev_dist
            print(f"[{name}] bestline {dist:.1f}m (segment: {sector_len:.1f}m) lat={lat:.6f}, lon={lon:.6f}")
            prev_dist = dist
        final_len = bestline_length - prev_dist
        print(f"[Final] {prev_dist:.1f}m -> {bestline_length:.1f}m (segment: {final_len:.1f}m)")

        # Save generated track info
        import json
        gen_info = {
            "bestline_length_m": round(bestline_length, 1),
            "bestline_points": len(track.bestline_utm),
            "source_session": session_path.name,
            "source_lap": result["lap_id"],
            "source_lap_time": round(result["lap_time"], 3),
            "sectors": {
                name: {"lat": round(lat, 7), "lon": round(lon, 7), "bestline_distance_m": round(dist, 1)}
                for name, (lat, lon, dist) in track.sector_intersections.items()
            },
        }
        gen_path = geometry_dir / "generated_track_info.json"
        gen_path.write_text(json.dumps(gen_info, indent=2) + "\n")
        print(f"[Info] Saved to {gen_path}")

    track.save_config(track_dir)
    print(f"[Config] Saved to {track_dir / 'track_config.json'}")

    export_dir = geometry_dir / "export"
    track.export_gpx(export_dir)
    track.export_kml(export_dir / "track.kml")
    track.export_ztracks(export_dir / f"{track_dir.name}.ztracks", venue_name=track.name)

    # Visualize
    from racing_tools.track.visualize_track import plot_track
    plot_track(track, track_dir)
    print(f"[Viz] Saved to {track_dir / 'track_visualization.png'}")


if __name__ == "__main__":
    main()
