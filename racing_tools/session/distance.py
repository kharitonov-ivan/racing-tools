from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import Transformer


def distance_from_gps(frame: pd.DataFrame) -> np.ndarray | None:
    lat_col = next((c for c in ("GPS Latitude", "Lat.") if c in frame.columns), None)
    lon_col = next((c for c in ("GPS Longitude", "Lon.") if c in frame.columns), None)
    if lat_col is None or lon_col is None:
        return None

    lat = pd.to_numeric(frame[lat_col], errors="coerce").values
    lon = pd.to_numeric(frame[lon_col], errors="coerce").values
    valid = np.isfinite(lat) & np.isfinite(lon) & (lat != 0) & (lon != 0)
    if valid.sum() < 2:
        return None

    med_lon = np.median(lon[valid])
    utm_zone = int((med_lon + 180) / 6) + 1
    hemisphere = "north" if np.median(lat[valid]) >= 0 else "south"
    epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)

    dx = np.diff(x)
    dy = np.diff(y)
    steps = np.sqrt(dx**2 + dy**2)
    median_step = np.median(steps[steps > 0]) if np.any(steps > 0) else 1.0
    steps[steps > median_step * 10] = median_step
    return np.concatenate([[0.0], np.cumsum(steps)])


def ensure_distance(
    frame: pd.DataFrame,
    *,
    distance_keys: Iterable[str],
    speed_keys: Iterable[str],
    frequency: float,
) -> pd.DataFrame:
    for key in distance_keys:
        if key in frame.columns and pd.to_numeric(frame[key], errors="coerce").notna().any():
            frame["Distance"] = pd.to_numeric(frame[key], errors="coerce").ffill().fillna(0.0)
            return frame

    gps_dist = distance_from_gps(frame)
    if gps_dist is not None:
        frame["Distance"] = gps_dist
        return frame

    speed_col = next((k for k in speed_keys if k in frame.columns), None)
    if speed_col is None:
        return frame

    speed = pd.to_numeric(frame[speed_col], errors="coerce").fillna(0.0)
    time = pd.to_numeric(frame.get("Time", pd.Series(range(len(frame)))), errors="coerce")
    delta = time.diff().fillna(0.0)
    positive = delta[delta > 0.0]
    fallback = positive.median() if not positive.empty else 1.0 / max(frequency, 1.0)
    delta[delta <= 0.0] = fallback
    frame["Distance"] = (speed * (1000.0 / 3600.0) * delta).cumsum()
    return frame
