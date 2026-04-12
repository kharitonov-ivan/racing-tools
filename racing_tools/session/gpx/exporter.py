from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def export(
    table: pd.DataFrame,
    path: Path,
    *,
    name: str = "",
    description: str = "",
    creator: str = "racing-tools",
    start_time: datetime | None = None,
) -> Path:
    """Export a Session table to GPX 1.1 format.

    Args:
        table: Session DataFrame with GPS Latitude, GPS Longitude columns.
            Optional columns: GPS Altitude, Time, Timestamp.
        path: Output file path.
        name: Track name in GPX metadata.
        description: Track description.
        creator: GPX creator attribute.
        start_time: Absolute start time. If None, uses Timestamp column
            or falls back to current UTC time.

    Returns:
        Path to the written GPX file.
    """
    try:
        import gpxpy
        import gpxpy.gpx
    except ImportError:
        raise ImportError("gpxpy is required to export GPX files. pip install gpxpy")

    lat_col = _pick(table, ["GPS Latitude", "Latitude"])
    lon_col = _pick(table, ["GPS Longitude", "Longitude"])
    if not lat_col or not lon_col:
        raise ValueError("Table must contain GPS Latitude and GPS Longitude columns")

    alt_col = _pick(table, ["GPS Altitude", "Altitude", "Elevation"])
    time_col = "Time" if "Time" in table.columns else None
    ts_col = "Timestamp" if "Timestamp" in table.columns else None

    # Resolve absolute start time for GPX point timestamps
    if ts_col:
        base_ts = float(table[ts_col].iloc[0])
    elif start_time:
        base_ts = start_time.timestamp()
    else:
        base_ts = datetime.now(timezone.utc).timestamp()

    gpx = gpxpy.gpx.GPX()
    gpx.creator = creator
    gpx.name = name
    gpx.description = description

    track = gpxpy.gpx.GPXTrack(name=name)
    gpx.tracks.append(track)

    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    lats = pd.to_numeric(table[lat_col], errors="coerce")
    lons = pd.to_numeric(table[lon_col], errors="coerce")
    alts = pd.to_numeric(table[alt_col], errors="coerce") if alt_col else None
    times = pd.to_numeric(table[time_col], errors="coerce") if time_col else None
    timestamps = pd.to_numeric(table[ts_col], errors="coerce") if ts_col else None

    for i in range(len(table)):
        lat = lats.iloc[i]
        lon = lons.iloc[i]
        if pd.isna(lat) or pd.isna(lon):
            continue

        elevation = float(alts.iloc[i]) if alts is not None and not pd.isna(alts.iloc[i]) else None

        # Compute absolute timestamp for this point
        if timestamps is not None and not pd.isna(timestamps.iloc[i]):
            ts = float(timestamps.iloc[i])
        elif times is not None and not pd.isna(times.iloc[i]):
            ts = base_ts + float(times.iloc[i])
        else:
            ts = None

        point_time = datetime.fromtimestamp(ts, tz=timezone.utc) if ts is not None else None

        point = gpxpy.gpx.GPXTrackPoint(
            latitude=float(lat),
            longitude=float(lon),
            elevation=elevation,
            time=point_time,
        )
        segment.points.append(point)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gpx.to_xml(), encoding="utf-8")
    return path


def _pick(table: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in table.columns:
            return c
    return None
