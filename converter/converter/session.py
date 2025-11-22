from __future__ import annotations

from dataclasses import dataclass, field
import math
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from bisect import bisect_right
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parent
MOTEC = ROOT / "motec_log_generator.py"
THIRD_MOTEC = ROOT.parent / "third_party" / "MotecLogGenerator" / "motec_log_generator.py"
WGS84_TO_WEBMERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def motec_script() -> Path:
    for candidate in (MOTEC, THIRD_MOTEC):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("motec_log_generator.py missing")


def load_mapping() -> dict:
    path = ROOT / "channel_mapping.json"
    return json.loads(path.read_text())


@dataclass
class Session:
    table: pd.DataFrame
    driver: str = ""
    venue: str = ""
    vehicle: str = ""
    session: str = ""
    device: str = ""
    event_date: str = ""
    event_time: str = ""
    tags: dict = field(default_factory=dict)

    def copy(self, table: pd.DataFrame | None = None, **kw) -> "Session":
        data = {
            "driver": kw.get("driver", self.driver),
            "venue": kw.get("venue", self.venue),
            "vehicle": kw.get("vehicle", self.vehicle),
            "session": kw.get("session", self.session),
            "device": kw.get("device", self.device),
            "event_date": kw.get("event_date", self.event_date),
            "event_time": kw.get("event_time", self.event_time),
            "tags": kw.get("tags", dict(self.tags)),
        }
        frame = table.copy() if table is not None else self.table.copy()
        return Session(frame, **data)

    def _ordered_table(self) -> pd.DataFrame:
        """Place time/distance columns first so MotecLogGenerator sees a valid timeline."""
        columns = list(self.table.columns)
        priority = [col for col in ("Time", "Distance") if col in columns]
        trailing = [col for col in columns if col not in priority]
        ordered = priority + trailing
        return self.table.loc[:, ordered] if ordered and ordered != columns else self.table

    def to_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._ordered_table()
        data.to_csv(path, index=False)
        return path

    def to_motec(
        self,
        *,
        output: Path,
        frequency: float,
        csv_path: Path | None = None,
        keep_csv: bool = False,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if csv_path is None:
            handle = tempfile.NamedTemporaryFile(suffix=".csv", prefix="tmp_session_", delete=False)
            csv_path = Path(handle.name)
            handle.close()
        self.to_csv(csv_path)

        cmd = [
            sys.executable,
            str(motec_script()),
            str(csv_path),
            "CSV",
            "--output",
            str(output),
            "--frequency",
            str(frequency),
        ]

        meta_flags = {
            "driver": "--driver",
            "venue": "--venue_name",
            "vehicle": "--vehicle_type",
            "session": "--event_session",
            "device": "--vehicle_comment",
        }
        for field, flag in meta_flags.items():
            value = getattr(self, field, "") or ""
            if value:
                cmd.extend([flag, value])

        subprocess.run(cmd, check=True)

        if not keep_csv:
            csv_path.unlink(missing_ok=True)
        return output

    def estimate_laps(
        self,
        start_finish_line: Sequence[tuple[float, float]],
        *,
        distance_threshold: float = 20.0,
        min_lap_time: float = 30.0,
    ) -> None:
        """
        Estimate lap numbers and lap-relative timers using a start/finish line.

        Args:
            start_finish_line: Iterable of (lon, lat) points describing the start/finish line.
            distance_threshold: Maximum lateral distance (meters) from the line to treat a sample as crossing.
            min_lap_time: Minimum seconds between valid crossings to avoid false positives.
        """
        if not start_finish_line or len(start_finish_line) < 2:
            return
        if "Time" not in self.table.columns:
            return

        lat_col = next((c for c in ("GPS Latitude", "Latitude", "Lat.") if c in self.table.columns), None)
        lon_col = next((c for c in ("GPS Longitude", "Longitude", "Lon.") if c in self.table.columns), None)
        if not lat_col or not lon_col:
            return

        time_series = pd.to_numeric(self.table["Time"], errors="coerce")
        lat_series = pd.to_numeric(self.table[lat_col], errors="coerce")
        lon_series = pd.to_numeric(self.table[lon_col], errors="coerce")
        valid_mask = ~(time_series.isna() | lat_series.isna() | lon_series.isna())
        if valid_mask.sum() < 2:
            return

        lon_vals = lon_series[valid_mask].to_numpy()
        lat_vals = lat_series[valid_mask].to_numpy()
        x_vals, y_vals = WGS84_TO_WEBMERC.transform(lon_vals, lat_vals)

        line_lon, line_lat = zip(*start_finish_line)
        line_x, line_y = WGS84_TO_WEBMERC.transform(np.array(line_lon), np.array(line_lat))
        start_vec = np.array([line_x[0], line_y[0]])
        end_vec = np.array([line_x[-1], line_y[-1]])
        line_vec = end_vec - start_vec
        line_len = float(np.hypot(line_vec[0], line_vec[1]))
        if line_len < 1e-3:
            return

        relative = np.column_stack([x_vals - start_vec[0], y_vals - start_vec[1]])
        cross_vals = relative[:, 0] * line_vec[1] - relative[:, 1] * line_vec[0]
        dist_to_line = cross_vals / line_len
        near_line = np.abs(dist_to_line) <= distance_threshold

        valid_indices = np.nonzero(valid_mask.to_numpy())[0]
        time_values = time_series.to_numpy(dtype=float, copy=True)
        start_time = float(time_series[valid_mask].iloc[0])
        if not math.isfinite(start_time):
            start_time = 0.0

        crossings: list[float] = []
        last_cross = None
        for idx in range(1, len(valid_indices)):
            if not (near_line[idx - 1] or near_line[idx]):
                continue
            c0 = cross_vals[idx - 1]
            c1 = cross_vals[idx]
            if math.copysign(1.0, c0) == math.copysign(1.0, c1):
                continue
            i0 = valid_indices[idx - 1]
            i1 = valid_indices[idx]
            t0 = time_values[i0]
            t1 = time_values[i1]
            if not (math.isfinite(t0) and math.isfinite(t1)):
                continue
            if c0 == c1:
                continue
            cross_time = t0 + (c0 / (c0 - c1)) * (t1 - t0)
            if last_cross is not None and cross_time - last_cross < min_lap_time:
                continue
            crossings.append(cross_time)
            last_cross = cross_time

        lap_numbers = np.full(len(time_values), np.nan)
        lap_elapsed = np.full(len(time_values), np.nan)
        lap_durations: list[float] = []

        if not crossings:
            for i, t in enumerate(time_values):
                if not math.isfinite(t):
                    continue
                lap_numbers[i] = 0
                lap_elapsed[i] = max(0.0, t - start_time)
        elif len(crossings) == 1:
            boundary = crossings[0]
            for i, t in enumerate(time_values):
                if not math.isfinite(t):
                    continue
                if t < boundary:
                    lap_numbers[i] = 0
                    lap_elapsed[i] = max(0.0, t - start_time)
                else:
                    lap_numbers[i] = 1
                    lap_elapsed[i] = max(0.0, t - boundary)
        else:
            lap_starts = crossings[:-1]
            last_boundary = crossings[-1]
            lap_durations = [end - start for start, end in zip(lap_starts, crossings[1:])]
            for i, t in enumerate(time_values):
                if not math.isfinite(t):
                    continue
                if t < lap_starts[0]:
                    lap_numbers[i] = 0
                    lap_elapsed[i] = max(0.0, t - start_time)
                    continue
                if t >= last_boundary:
                    lap_numbers[i] = -1
                    lap_elapsed[i] = max(0.0, t - last_boundary)
                    continue
                pos = bisect_right(lap_starts, t) - 1
                pos = max(pos, 0)
                lap_numbers[i] = pos + 1
                lap_elapsed[i] = max(0.0, t - lap_starts[pos])

        lap_series = pd.Series(lap_numbers, index=self.table.index).round().astype("Int64")
        self.table["LapNumber"] = lap_series
        self.table["LapTime"] = lap_elapsed
        if lap_durations:
            self.tags["lap_info"] = {
                "crossings": crossings,
                "lap_durations": lap_durations,
            }


class ChannelNormalizer:
    def __init__(self, mapping_file: str | None = None, device_type: str | None = None):
        config = (
            json.loads(Path(mapping_file).read_text())
            if mapping_file
            else load_mapping()
        )
        self.alias = {}
        for item in config["standard_channels"].values():
            name = item["standard_name"]
            unit = item["unit"]
            for raw in item["aliases"]:
                self.alias[raw.lower()] = (name, unit)

        self.transforms = config.get("transformations", {}).get(device_type or "", {})

    def apply_transformations(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.transforms:
            return frame
        data = frame.copy()
        for key, spec in self.transforms.items():
            if key not in data.columns:
                continue
            series = pd.to_numeric(data[key], errors="coerce").fillna(0.0)
            scale = spec.get("scale", 1.0)
            offset = spec.get("offset", 0.0)
            if scale != 1.0:
                series = series * scale
            if offset:
                series = series + offset
            if spec.get("auto_center"):
                series = series - series.mean()
            data[key] = series
        return data

    def normalize_dataframe(
        self,
        frame: pd.DataFrame,
        add_units_row: bool = False,
        apply_transforms: bool = True,
    ):
        data = self.apply_transformations(frame) if apply_transforms else frame.copy()
        names: list[str] = []
        units: dict[str, str] = {}
        counts: dict[str, int] = {}

        for col in data.columns:
            base, unit = self.alias.get(col.lower(), (col, ""))
            counts[base] = counts.get(base, 0) + 1
            final = base if counts[base] == 1 else f"{base}_{counts[base]}"
            names.append(final)
            units[final] = unit

        normalized = data.copy()
        normalized.columns = names
        return (normalized, units) if add_units_row else normalized


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


def name_tokens(path: Path) -> list[str]:
    import re

    return [p for p in re.split(r"[-_\s]+", path.name) if p]
