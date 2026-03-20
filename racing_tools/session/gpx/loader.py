from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from racing_tools.session.distance import ensure_distance
from racing_tools.session.normalizer import ChannelNormalizer


def load(path: Path, normalize: bool = True) -> tuple[pd.DataFrame, dict]:
    try:
        import gpxpy
    except ImportError:
        raise ImportError("gpxpy is required to load GPX files. pip install gpxpy")

    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")

    with path.open("r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    data = []
    start_time = None

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                row = {
                    "GPS Latitude": point.latitude,
                    "GPS Longitude": point.longitude,
                    "GPS Altitude": point.elevation,
                }

                if point.time:
                    ts = point.time.timestamp()
                    if start_time is None:
                        start_time = ts
                    row["Time"] = ts - start_time
                    row["Timestamp"] = ts

                data.append(row)

    if not data:
        raise ValueError(f"No points found in {path}")

    df = pd.DataFrame(data)

    if "Time" not in df.columns:
        raise ValueError("GPX points missing time data")

    df = df.sort_values("Time").reset_index(drop=True)

    if "GPS Speed" not in df.columns:
        lats = np.radians(df["GPS Latitude"])
        lons = np.radians(df["GPS Longitude"])

        dlat = lats.diff()
        dlon = lons.diff()

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lats.shift()) * np.cos(lats) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist = 6371000.0 * c

        dt = df["Time"].diff()

        speed = dist / dt
        speed = speed.fillna(0.0)
        speed[speed > 300] = 0.0

        df["GPS Speed"] = speed * 3.6

    if "Distance" not in df.columns:
        df = ensure_distance(df, distance_keys=["GPS Distance"], speed_keys=["GPS Speed"], frequency=10.0)

    if normalize:
        df = ChannelNormalizer().normalize_dataframe(df)

    meta = {
        "creator": gpx.creator,
        "name": gpx.tracks[0].name if gpx.tracks else "",
        "desc": gpx.tracks[0].description if gpx.tracks else "",
    }

    date_str = ""
    time_str = ""
    if gpx.time:
        date_str = gpx.time.date().isoformat()
        time_str = gpx.time.strftime("%H:%M")
    elif start_time:
        dt = datetime.fromtimestamp(start_time)
        date_str = dt.date().isoformat()
        time_str = dt.strftime("%H:%M")

    return df, {
        "driver": "",
        "venue": "",
        "vehicle": "",
        "event_date": date_str,
        "event_time": time_str,
        "device": "GPX",
        "tags": meta,
    }
