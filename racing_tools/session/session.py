from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import math
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from bisect import bisect_right
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import re
import csv
from datetime import datetime
from pyproj import Transformer


@dataclass
class PiecewiseSync:
    """Piecewise linear mapping between video time and telemetry time.

    anchors: list of (video_time, telemetry_time) pairs, sorted by video_time.
    With 1 pair acts as constant offset. Uses np.interp for interpolation
    and linear extrapolation beyond edges.
    """
    anchors: list[tuple[float, float]]  # shape: (N, 2)

    def __post_init__(self) -> None:
        self.anchors.sort(key=lambda a: a[0])
        self._v = np.array([a[0] for a in self.anchors])
        self._t = np.array([a[1] for a in self.anchors])

    def video_to_telemetry(self, video_time: np.ndarray | float) -> np.ndarray:
        """Map video time(s) to telemetry time via piecewise linear interp."""
        return np.interp(video_time, self._v, self._t)

    def telemetry_to_video(self, telem_time: np.ndarray | float) -> np.ndarray:
        """Map telemetry time(s) to video time (inverse mapping)."""
        return np.interp(telem_time, self._t, self._v)

    @classmethod
    def from_offset(cls, offset: float) -> "PiecewiseSync":
        """Create from a single constant offset (telemetry_time = video_time + offset)."""
        return cls(anchors=[(0.0, offset), (1e6, 1e6 + offset)])

    def to_dict(self) -> dict:
        return {
            "type": "piecewise",
            "anchors_video": self._v.tolist(),
            "anchors_telem": self._t.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PiecewiseSync":
        """Load from sidecar dict. Supports legacy {offset: float} format."""
        if data.get("type") == "piecewise":
            return cls(anchors=list(zip(data["anchors_video"], data["anchors_telem"])))
        # Legacy single-offset format
        offset = data.get("offset", 0.0)
        return cls.from_offset(offset)


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


DISTANCE_AIM = ["Distance on GPS Speed", "Distance", "GPS Distance"]
SPEED_AIM = ["GPS Speed", "Speed GPS", "Wheel Speed", "Rear Speed", "Vehicle Speed"]
DISTANCE_ALFANO = ["Distance on GPS Speed", "Distance"]
SPEED_ALFANO = ["Speed GPS", "GPS Speed", "Speed rear", "Wheel Speed"]
DISTANCE_EXCEL = ["Distance"]
SPEED_EXCEL = ["Speed GPS", "Speed rear", "Wheel Speed"]
ALFANO_STEP = 0.1
_LAP_NUM_RE = re.compile(r"LAP_(\d+)_")


def _extract_lap_number(filename: str) -> int:
    """Extract lap number from LAP_N_... filename for natural sorting."""
    m = _LAP_NUM_RE.search(filename)
    return int(m.group(1)) if m else 0


def _detect_alfano_device(files: list[Path]) -> str:
    """Detect Alfano device type (Alfano6/Alfano7) from file names."""
    for f in files:
        upper = f.name.upper()
        if "ALFANO7" in upper:
            return "Alfano7"
        if "ALFANO6" in upper:
            return "Alfano6"
    return "Alfano"
DATE_TOKEN = re.compile(r"\d{6}")
TIME_TOKEN = re.compile(r"\d{2}H\d{2}")


def decode_compact_date(token: str) -> str:
    if not DATE_TOKEN.fullmatch(token or ""):
        return ""
    day = int(token[:2])
    month = int(token[2:4])
    year = int(token[4:])
    year += 2000 if year < 70 else 1900
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return ""


def decode_time_token(token: str) -> str:
    if not TIME_TOKEN.fullmatch(token or ""):
        return ""
    return f"{token[:2]}:{token[-2:]}"


def decode_utc_clock(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 4:
        return ""
    digits = digits.rjust(6, "0")[:6]
    return f"{digits[:2]}:{digits[2:4]}:{digits[4:]}"


def infer_datetime_from_tokens(tokens: list[str]) -> tuple[str, str]:
    date = ""
    time = ""
    for token in tokens:
        if not date:
            date = decode_compact_date(token)
        if not time:
            time = decode_time_token(token)
    return date, time


def infer_datetime_from_path(path: Path) -> tuple[str, str]:
    return infer_datetime_from_tokens(name_tokens(path))


def aim_metadata(lines: list[str]) -> dict[str, str]:
    meta_lines = []
    for line in lines:
        if not line.strip():
            break
        meta_lines.append(line)

    meta: dict[str, str] = {}
    for line in meta_lines:
        row = next(csv.reader([line]))
        if not row:
            continue
        key = row[0].strip().strip('"')
        if not key or key == "Format":
            continue
        value = ",".join(row[1:]).strip().strip('"')
        meta[key] = value
    return meta


def aim_datetime(meta: dict[str, str]) -> tuple[str, str]:
    iso_date = ""
    iso_time = ""
    date_text = meta.get("Date", "")
    time_text = meta.get("Time", "")
    if date_text:
        for fmt in ("%A, %B %d, %Y", "%B %d, %Y"):
            try:
                iso_date = datetime.strptime(date_text, fmt).date().isoformat()
                break
            except ValueError:
                continue
    if time_text:
        for fmt in ("%I:%M %p", "%I %p"):
            try:
                iso_time = datetime.strptime(time_text, fmt).strftime("%H:%M")
                break
            except ValueError:
                continue
    return iso_date, iso_time


def aim_frame(csv_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    lines = csv_path.read_text().splitlines()
    meta = aim_metadata(lines)
    header = None
    for i, raw in enumerate(lines):
        text = raw.strip()
        if not text:
            continue
        normalized = text.lstrip('"')
        if normalized.startswith("Time") and "GPS" in normalized:
            header = i
            break
    if header is None:
        raise ValueError("Time header missing")
    frame = pd.read_csv(csv_path, skiprows=header)
    if len(frame):
        token = str(frame.iloc[0, 0]).replace(".", "").replace("-", "")
        if not token.isdigit():
            frame = frame.iloc[1:].reset_index(drop=True)
    frame.columns = frame.columns.str.strip().str.replace('"', "")
    return frame, meta


def clean_header(raw: str) -> str:
    text = " ".join(str(raw or "").split())
    if ":" in text and not text.lower().startswith("utc time"):
        text = text.split(":")[0].strip()
    return text


def excel_header_clock(csv_path: Path) -> str:
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()
        if not first_line:
            return ""
        for cell in first_line.split(";"):
            if "UTC" in cell:
                return decode_utc_clock(cell.split(":")[-1])
    except Exception:
        pass
    return ""


def excel_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, sep=";", engine="python")
    frame.columns = [clean_header(col) for col in frame.columns]
    frame = frame.dropna(axis=1, how="all").dropna(how="all").copy()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="ignore")
    return frame


def stitch_time(frame: pd.DataFrame) -> pd.DataFrame:
    if "Time" not in frame.columns:
        raise ValueError("Time column missing")
    data = frame.copy()
    data["Time"] = pd.to_numeric(data["Time"], errors="coerce")
    data = data[data["Time"].notna()].reset_index(drop=True)
    laps = pd.to_numeric(data.get("Lap"), errors="coerce").ffill().fillna(1)
    offsets = {}
    total = 0.0
    for lap in laps.dropna().unique():
        mask = laps == lap
        duration = pd.to_numeric(data.loc[mask, "Time"], errors="coerce").max()
        duration = float(duration) if duration and duration == duration else 0.0
        offsets[lap] = total
        total += duration
    data["Time"] = data["Time"] + laps.map(offsets).fillna(0.0)
    return data


def infer_frequency(time_series: pd.Series) -> float:
    deltas = time_series.diff()
    positive = deltas[deltas > 0].dropna()
    if positive.empty:
        return 10.0
    return round(1.0 / positive.median(), 3)


@dataclass
class SessionMetadata:
    """Metadata about the recording session."""
    driver: str = ""
    venue: str = ""
    vehicle: str = ""
    session: str = ""
    device: str = ""
    event_date: str = ""
    event_time: str = ""
    tags: dict = field(default_factory=dict)

    def copy(self, **kw) -> "SessionMetadata":
        return SessionMetadata(
            driver=kw.get("driver", self.driver),
            venue=kw.get("venue", self.venue),
            vehicle=kw.get("vehicle", self.vehicle),
            session=kw.get("session", self.session),
            device=kw.get("device", self.device),
            event_date=kw.get("event_date", self.event_date),
            event_time=kw.get("event_time", self.event_time),
            tags=kw.get("tags", dict(self.tags)),
        )


@dataclass
class Session:
    table: pd.DataFrame
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    # Lap analysis
    track: "TrackGeometry | None" = None
    crossings: list[float] = field(default_factory=list)
    crossings_gps: list[float] = field(default_factory=list)

    # Backward compatibility properties
    @property
    def driver(self) -> str:
        return self.metadata.driver
    
    @driver.setter
    def driver(self, value: str):
        self.metadata.driver = value

    @property
    def venue(self) -> str:
        return self.metadata.venue
    
    @venue.setter
    def venue(self, value: str):
        self.metadata.venue = value

    @property
    def vehicle(self) -> str:
        return self.metadata.vehicle
    
    @vehicle.setter
    def vehicle(self, value: str):
        self.metadata.vehicle = value

    @property
    def session(self) -> str:
        return self.metadata.session
    
    @session.setter
    def session(self, value: str):
        self.metadata.session = value

    @property
    def device(self) -> str:
        return self.metadata.device
    
    @device.setter
    def device(self, value: str):
        self.metadata.device = value

    @property
    def event_date(self) -> str:
        return self.metadata.event_date
    
    @event_date.setter
    def event_date(self, value: str):
        self.metadata.event_date = value

    @property
    def event_time(self) -> str:
        return self.metadata.event_time
    
    @event_time.setter
    def event_time(self, value: str):
        self.metadata.event_time = value

    @property
    def tags(self) -> dict:
        return self.metadata.tags
    
    @tags.setter
    def tags(self, value: dict):
        self.metadata.tags = value

    def _pick_column(self, candidates: list[str]) -> str | None:
        """Find first matching column name from candidates."""
        for name in candidates:
            if name in self.table.columns:
                return name
        return None

    def detect_crossings(self) -> list[float]:
        """Detect start-finish line crossings from GPS data. Requires self.track to be set."""
        if self.track is None or self.track.start_finish_wgs84 is None:
            return []
        
        lat_col = self._pick_column(["GPS Latitude", "Latitude"])
        lon_col = self._pick_column(["GPS Longitude", "Longitude"])
        if not lat_col or not lon_col:
            return []

        lats = pd.to_numeric(self.table[lat_col], errors="coerce").values
        lons = pd.to_numeric(self.table[lon_col], errors="coerce").values
        times = pd.to_numeric(self.table["Time"], errors="coerce").values if "Time" in self.table.columns else self.table.index.values

        sf_points = list(dict.fromkeys(self.track.start_finish_wgs84))
        if len(sf_points) < 2:
            return []
        sf_p1, sf_p2 = sf_points[0], sf_points[-1]

        crossings = []
        for i in range(len(self.table) - 1):
            p1, p2 = (lons[i], lats[i]), (lons[i+1], lats[i+1])
            if p1 == p2:
                continue
            intersects, t = self._segments_intersect(p1, p2, sf_p1, sf_p2)
            if intersects:
                crossings.append(times[i] + t * (times[i+1] - times[i]))
        
        self.crossings = crossings
        return crossings

    @staticmethod
    def _segments_intersect(p1, p2, q1, q2) -> tuple[bool, float]:
        """Check if line segment p1-p2 intersects with q1-q2."""
        px, py = p1
        rx, ry = p2[0] - px, p2[1] - py
        qx, qy = q1
        sx, sy = q2[0] - qx, q2[1] - qy
        cross = rx * sy - ry * sx
        if abs(cross) < 1e-9:
            return False, 0.0
        t = ((qx - px) * sy - (qy - py) * sx) / cross
        u = ((qx - px) * ry - (qy - py) * rx) / cross
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return True, t
        return False, 0.0

    def add_lap_numbers(self) -> None:
        """Add LapNumber and LapTime columns to table based on crossings. Modifies self.table in-place."""
        times = pd.to_numeric(self.table["Time"], errors="coerce").values if "Time" in self.table.columns else self.table.index.values
        lap_numbers = np.zeros(len(self.table), dtype=int)
        lap_times = np.zeros(len(self.table), dtype=float)
        
        crossing_idx = 0
        for i, t in enumerate(times):
            while crossing_idx < len(self.crossings) and t >= self.crossings[crossing_idx]:
                crossing_idx += 1
            lap_numbers[i] = crossing_idx
            
            # Calculate LapTime (elapsed time since lap start)
            if crossing_idx == 0:
                # Before first crossing - time from start
                lap_times[i] = t - times[0] if len(times) > 0 else 0.0
            else:
                # Time since last crossing
                lap_times[i] = t - self.crossings[crossing_idx - 1]
        
        self.table["LapNumber"] = lap_numbers
        self.table["LapTime"] = lap_times

    def get_lap_durations(self) -> dict[int, float]:
        """Calculate lap durations from crossings."""
        if not self.crossings:
            return {}
        start_time = self.table["Time"].iloc[0] if "Time" in self.table.columns else 0.0
        durations = {0: self.crossings[0] - start_time}
        for i in range(1, len(self.crossings)):
            durations[i] = self.crossings[i] - self.crossings[i-1]
        return durations

    def _get_gps_lap_durations(self) -> dict[int, float]:
        """Calculate lap durations from GPS crossings (telemetry time)."""
        if not self.crossings_gps:
            return {}
        durations: dict[int, float] = {}
        for i in range(1, len(self.crossings_gps)):
            durations[i] = self.crossings_gps[i] - self.crossings_gps[i - 1]
        return durations

    def get_lap_stats(self) -> list[dict]:
        """Calculate statistics for each lap. Returns list of dicts with id, time, gps_time, speed/rpm stats."""
        if "LapNumber" not in self.table.columns:
            return []

        lap_durations = self.get_lap_durations()
        gps_durations = self._get_gps_lap_durations()
        speed_col = self._pick_column(["GPS Speed", "Speed", "Vitesse"])
        rpm_col = self._pick_column(["RPM", "Régime"])

        stats = []
        for lap_id in sorted(self.table["LapNumber"].unique()):
            lap_data = self.table[self.table["LapNumber"] == lap_id]
            if lap_data.empty:
                continue

            lap_time = lap_durations.get(int(lap_id), lap_data["LapTime"].max() if "LapTime" in lap_data.columns else 0.0)
            gps_time = gps_durations.get(int(lap_id))

            stat: dict = {"id": int(lap_id), "time": lap_time, "gps_time": gps_time,
                          "min_speed": None, "max_speed": None, "min_rpm": None, "max_rpm": None}
            if speed_col:
                s = pd.to_numeric(lap_data[speed_col], errors="coerce")
                stat["min_speed"], stat["max_speed"] = s.min(), s.max()
            if rpm_col:
                r = pd.to_numeric(lap_data[rpm_col], errors="coerce")
                stat["min_rpm"], stat["max_rpm"] = r.min(), r.max()
            stats.append(stat)
        return stats

    @property
    def best_lap(self) -> dict | None:
        """Return best lap using statistical filtering to reject outliers."""
        stats = self.get_lap_stats()
        valid = [s for s in stats if s["time"] and s["time"] > 20.0]
        if not valid:
            return None
        if len(valid) < 3:
            return min(valid, key=lambda x: x["time"])
        
        times = np.array([s["time"] for s in valid])
        median = np.median(times)
        mad = np.median(np.abs(times - median))
        threshold = 6.0 * mad if mad > 0.001 else 3 * np.std(times)
        lower_bound = median - threshold
        candidates = [s for s in valid if s["time"] >= lower_bound]
        return min(candidates, key=lambda x: x["time"]) if candidates else min(valid, key=lambda x: x["time"])
    
    def __repr__(self) -> str:
        channels_str = ", ".join(self.table.columns.tolist())
        if len(channels_str) > 100:
            channels_str = channels_str[:97] + "..."
            
        return (f"Session(\n"
                f"  Driver: {self.driver}\n"
                f"  Vehicle: {self.vehicle}\n"
                f"  Venue: {self.venue}\n"
                f"  Date: {self.event_date} {self.event_time}\n"
                f"  Device: {self.device}\n"
                f"  Rows: {len(self.table)}\n"
                f"  Channels: [{channels_str}]\n"
                f"  Metadata: {self.tags}\n"
                f")")

    def copy(self, table: pd.DataFrame | None = None, **kw) -> "Session":
        frame = table.copy() if table is not None else self.table.copy()
        new_metadata = self.metadata.copy(**kw)
        return Session(frame, metadata=new_metadata)

    @classmethod
    def load(cls, path_or_folder: str | Path, **kwargs) -> "Session":
        """
        Universal loader that dispatches to specific methods based on file type.
        
        Args:
            path_or_folder: File path or directory to load.
            **kwargs: Additional arguments passed to the specific loader.
            
        Returns:
            Session object.
        """
        path = Path(path_or_folder)
        
        if path.is_dir():
            # Directories are typically Alfano Excel exports folder or AIM CSV folder
            # Simple heuristic: look for Alfano pattern
            if list(path.glob("Excel_*.csv")):
                return cls.load_alfano_csv(path, **kwargs)
            if list(path.glob("LAP_*.csv")):
                return cls.load_alfano_raw(path, **kwargs)
            return cls.load_aim_csv(path, **kwargs)
            
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
            
        suffix = path.suffix.lower()
        
        if suffix == ".gpx":
            return cls.load_gpx(path, **kwargs)
            
        if suffix in (".xrk", ".xrs"):
            return cls.load_aim_raw(path, **kwargs)

        if suffix == ".zip":
            return cls._load_from_zip(path, **kwargs)
            
        if suffix == ".csv":
            # Detect Alfano Excel CSVs by filename pattern
            if path.name.startswith("Excel_"):
                return cls.load_alfano_csv(path, **kwargs)
            return cls.load_aim_csv(path, **kwargs)
            
        raise ValueError(f"Unsupported file extension: {suffix}")

    @classmethod
    def _load_from_zip(cls, path: Path, **kwargs) -> "Session":
        """Extract a ZIP archive to a temp directory and dispatch to the appropriate loader.
        
        Supports Alfano ZIP archives containing LAP_*.csv or Excel_*.csv files.
        """
        path = Path(path)
        if not zipfile.is_zipfile(path):
            raise ValueError(f"{path} is not a valid ZIP file")
        
        tmp_dir = Path(tempfile.mkdtemp(prefix="racing_session_"))
        
        # Schedule cleanup
        def cleanup():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        atexit.register(cleanup)
        
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp_dir)
        
        # Check contents and dispatch
        if list(tmp_dir.glob("LAP_*.csv")):
            return cls.load_alfano_raw(tmp_dir, **kwargs)
        if list(tmp_dir.glob("Excel_*.csv")):
            return cls.load_alfano_csv(tmp_dir, **kwargs)
        
        # Fallback: try treating it as a generic folder
        return cls.load(tmp_dir, **kwargs)

    @classmethod
    def load_aim_raw(cls, path: Path, normalize: bool = True) -> "Session":
        """
        Load Session from AIM XRK file.
        Attempts to use official xrk library (Windows DLL) first,
        then falls back to TrackDataAnalysis (Cython) parser if available.
        """
        path = Path(path)
        if not path.is_file():
             raise FileNotFoundError(f"{path} not found")

        tags = {}
        table = None
        event_date = ""
        event_time = ""
        driver = ""
        venue = ""
        vehicle = ""
        session_type = ""
        
        # Try native Windows DLL wrapper first
        try:
            from racing_tools.third_party.xrk import xrk

            data = xrk.XRK(str(path))
            
            # Extract Metadata
            tags = {
                "championship": data.championship_name,
                "track": data.track_name,
                "vehicle": data.vehicle_name,
                "racer": data.racer_name,
                "venue_type": data.venue_type,
                "datetime": data.datetime,
                "lapcount": data.lapcount,
            }
            
            # Parse datetime
            if data.datetime:
                try:
                     dt = datetime.strptime(data.datetime, "%Y-%m-%d %H:%M:%S")
                     event_date = dt.date().isoformat()
                     event_time = dt.strftime("%H:%M")
                except ValueError:
                    pass
            
            driver = data.racer_name
            venue = data.track_name
            vehicle = data.vehicle_name

            # Extract Channels
            series_list = []
            for name, channel in data.channels.items():
                try:
                    times, values = channel.samples(xtime=True, xabsolute=True)
                except Exception as e:
                    print(f"Warning: failed to load channel {name}: {e}")
                    continue
                
                if not times:
                    continue
                    
                s = pd.Series(values, index=pd.Index(times, name="Time"), name=name)
                if s.index.duplicated().any():
                    s = s[~s.index.duplicated(keep='first')]
                series_list.append(s)
            
            if not series_list:
                raise ValueError(f"No valid channels found in {path}")
                
            table = pd.concat(series_list, axis=1).sort_index()

            # Handle lap info
            if data.lap_info and "Time" in table.columns:
                 # Logic for adding lap info (same as below/previous)
                 # We will reuse the post-processing block
                 pass

        except (ImportError, OSError, AttributeError) as e_dll:
            # Fallback to TrackDataAnalysis Cython parser
            # print(f"Native XRK loader unavailable ({e_dll}), trying TrackDataAnalysis parser...")
            
            try:
                tda_path = Path(__file__).resolve().parents[1] / "third_party" / "TrackDataAnalysis"
                if str(tda_path) not in sys.path:
                    sys.path.append(str(tda_path))
                
                from third_party.TrackDataAnalysis.data import aim_xrk

                def progress(a, b): pass
                
                tda_log = aim_xrk.AIMXRK(str(path), progress)
                
                tags = tda_log.metadata.copy()
                tags["source"] = "TrackDataAnalysis"
                
                # Extract Metadata fields
                driver = tags.get("Driver", "")
                venue = tags.get("Venue", "")
                vehicle = tags.get("Vehicle", "")
                session_type = tags.get("Session", "")
                event_date = str(tags.get("Log Date", ""))
                event_time = str(tags.get("Log Time", ""))

                # Extract Channels
                series_list = []
                for name, channel in tda_log.channels.items():
                     # TDA channels use milliseconds for timecodes based on observation/analysis
                    times_ms = np.array(channel.timecodes)
                    values = np.array(channel.values) 
                    
                    if len(times_ms) == 0:
                        continue

                    # Convert ms to seconds
                    times_sec = times_ms / 1000.0
                    
                    s = pd.Series(values, index=pd.Index(times_sec, name="Time"), name=name)
                    if s.index.duplicated().any():
                        s = s[~s.index.duplicated(keep='first')]
                    series_list.append(s)

                if not series_list:
                    raise ValueError("No channels found using TDA parser")
                
                table = pd.concat(series_list, axis=1).sort_index()
                
            except Exception as e_tda:
                # If both fail, raise informative error
                raise ImportError(f"Failed to load XRK. Native DLL error: {e_dll} | TDA Parser error: {e_tda}")

        # Post-processing (common)
        table = table.ffill().bfill()
        table = table.reset_index()
        
        if normalize:
            table = ChannelNormalizer(device_type="aim").normalize_dataframe(table)
        
        # Ensure Distance column exists
        table = ensure_distance(table, distance_keys=DISTANCE_AIM, speed_keys=SPEED_AIM, frequency=20.0)
            
        session = cls(
            table=table,
            metadata=SessionMetadata(
                driver=driver,
                venue=venue,
                vehicle=vehicle,
                event_date=event_date,
                event_time=event_time,
                device="AIM XRK",
                tags=tags
            )
        )
        
        # Lap Info integration (simplified for now as TDA structures it differently)
        # If we have reliable lap info we should add it.
        # For now, let's rely on basic columns.
        
        return session


    @classmethod
    def load_aim_csv(cls, path_or_folder: Path, frequency: float = 20.0, normalize: bool = True) -> "Session":
        """Load AIM session from CSV export."""
        path = Path(path_or_folder)
        csv_path = path if path.is_file() else path / "aim.csv"
        
        if not csv_path.exists():
             # If passed folder, and aim.csv not found
             raise FileNotFoundError(f"AIM CSV not found at {csv_path}")

        frame, meta = aim_frame(csv_path)
        frame = ensure_distance(frame, distance_keys=DISTANCE_AIM, speed_keys=SPEED_AIM, frequency=frequency)
        if normalize:
            frame = ChannelNormalizer().normalize_dataframe(frame)

        session = cls(frame, metadata=SessionMetadata(device="AIM"))
        
        # Try metadata from parsed header
        if meta:
            session.tags.update({"aim_meta": meta})
            if meta.get("Racer"):
                session.driver = meta["Racer"]
            if meta.get("Vehicle"):
                session.vehicle = meta["Vehicle"]
            if meta.get("Session"):
                session.session = meta["Session"]
            
            iso_date, iso_time = aim_datetime(meta)
            session.event_date = iso_date
            session.event_time = iso_time

        # Infer from path if missing
        if not session.event_date or not session.event_time:
            # use folder name if path is file, take parent
            p = path if path.is_dir() else path.parent
            date_token, time_token = infer_datetime_from_path(p)
            session.event_date = session.event_date or date_token
            session.event_time = session.event_time or time_token
            
            tokens = name_tokens(p)
            if len(tokens) > 3 and not session.venue:
                session.venue = tokens[3]
            if len(tokens) > 4 and not session.vehicle:
                session.vehicle = tokens[4]
            if len(tokens) > 5 and not session.driver:
                session.driver = tokens[5]
            if tokens and not session.session:
                session.session = tokens[-1]

        return session

    @classmethod
    def load_alfano_raw(cls, folder: Path, normalize: bool = True) -> "Session":
        """Load Alfano session from LAP_*.csv files."""
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder} is not a directory")

        files = sorted(
            folder.glob("LAP_*.csv"),
            key=lambda p: _extract_lap_number(p.name),
        )
        frames: list[pd.DataFrame] = []
        for lap_idx, p in enumerate(files, start=1):
            if not p.is_file():
                continue
            df = pd.read_csv(p)
            # Assign lap number from file name order (ignore Partiel column)
            df["Partiel"] = lap_idx
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No LAP_*.csv files in {folder}")

        frame = pd.concat(frames, ignore_index=True)
        frame.insert(0, "Time", np.arange(len(frame)) * ALFANO_STEP)

        if normalize:
            frame = ChannelNormalizer(device_type="alfano").normalize_dataframe(frame)
        frame = ensure_distance(frame, distance_keys=DISTANCE_ALFANO, speed_keys=SPEED_ALFANO, frequency=1.0 / ALFANO_STEP)

        # Detect device type from file names
        device = _detect_alfano_device(files)
        session = cls(frame, metadata=SessionMetadata(device=device))

        tokens = name_tokens(folder)
        if len(tokens) > 1:
            session.driver = tokens[-2]
            session.venue = tokens[-1]

        date_text, time_text = infer_datetime_from_path(folder)
        session.event_date = date_text
        session.event_time = time_text
        return session

    @classmethod
    def load_alfano_csv(cls, path_or_folder: Path, frequency: float = None, normalize: bool = True) -> "Session":
        """Load Alfano session from Excel export (Excel_*.csv).
        
        Accepts either a folder containing Excel_*.csv files,
        or a direct path to an Excel_*.csv file.
        """
        path_or_folder = Path(path_or_folder)
        
        # Accept a direct CSV file path
        if path_or_folder.is_file() and path_or_folder.suffix.lower() == ".csv":
            folder = path_or_folder.parent
        elif path_or_folder.is_dir():
            folder = path_or_folder
        else:
            raise FileNotFoundError(f"{path_or_folder} is not a file or directory")

        files = sorted(folder.glob("Excel_*.csv"))
        if not files:
            raise FileNotFoundError(f"No Excel_*.csv files in {folder}")
            
        csv_path = files[0]
        frame = excel_frame(csv_path)
        frame = stitch_time(frame)
        
        freq = frequency
        if freq is None:
            freq = infer_frequency(frame["Time"])
            
        if normalize:
            frame = ChannelNormalizer(device_type="alfano").normalize_dataframe(frame)
        frame = ensure_distance(frame, distance_keys=DISTANCE_EXCEL, speed_keys=SPEED_EXCEL, frequency=freq)

        session = cls(frame, metadata=SessionMetadata(device="Alfano6 Excel"))
        
        tokens = name_tokens(folder)
        if len(tokens) > 1:
            session.driver = tokens[-2]
            session.venue = tokens[-1]
            
        date_file, time_file = infer_datetime_from_path(csv_path)
        date_folder, time_folder = infer_datetime_from_path(folder)
        utc_clock = excel_header_clock(csv_path)
        
        session.event_date = date_file or date_folder
        session.event_time = utc_clock or time_file or time_folder
        
        return session

    @classmethod
    def load_gpx(cls, path: Path, normalize: bool = True) -> "Session":
        """Load Session from GPX file."""
        try:
            import gpxpy
        except ImportError:
            raise ImportError("gpxpy is required to load GPX files. pip install gpxpy")

        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")

        with path.open("r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)

        data = []
        
        # We assume a single timeline, so flatten all tracks/segments
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
                        # Convert to UTC timestamp
                        # point.time is usually datetime.datetime
                        ts = point.time.timestamp()
                        if start_time is None:
                            start_time = ts
                        row["Time"] = ts - start_time
                        row["Timestamp"] = ts # Keep absolute check
                        
                    # Check for extensions (speed, hr, cadence, temp, depth)
                    # gpxpy parses extensions into an ElementTree object usually
                    # But some standard extensions are accessible
                    
                    # Common extension: speed from various schemas
                    # But often better to trust gpxpy's speed calculation between points if missing?
                    # Let's see if we can extract raw extensions easily if needed.
                    # For now, let's rely on standard fields.
                    
                    data.append(row)

        if not data:
            raise ValueError(f"No points found in {path}")

        df = pd.DataFrame(data)
        
        if "Time" not in df.columns:
            # If no time, make up a time based on index?
            # GPX usually has time.
            raise ValueError("GPX points missing time data")

        # Sort just in case
        df = df.sort_values("Time").reset_index(drop=True)
        
        # Calculate Speed if missing
        # We can use gpxpy functionality or manual
        if "GPS Speed" not in df.columns:
            # Simple 2D speed calculation
            # Use centered difference or just forward diff
            # Let's use simple point-to-point
            
            # Need to re-iterate or use pandas vectorization
            # Haversine formula
            
            # R1 = 6371000
            # phi1 = lat1 * pi/180
            # phi2 = lat2 * pi/180
            # dphi = (lat2 - lat1) * pi/180
            # dlambda = (lon2 - lon1) * pi/180
            # a = sin(dphi/2)^2 + cos(phi1)*cos(phi2)*sin(dlambda/2)^2
            # c = 2 * atan2(sqrt(a), sqrt(1-a))
            # d = R * c
            
            lats = np.radians(df["GPS Latitude"])
            lons = np.radians(df["GPS Longitude"])
            
            dlat = lats.diff()
            dlon = lons.diff()
            
            # semi-haversine
            a = np.sin(dlat/2.0)**2 + np.cos(lats.shift()) * np.cos(lats) * np.sin(dlon/2.0)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            dist = 6371000.0 * c # Meters
            
            dt = df["Time"].diff()
            
            speed = dist / dt
            # fillna(0) for first point
            speed = speed.fillna(0.0)
            
            # Handle inf values or unrealistic speeds
            speed[speed > 300] = 0.0 # > 300m/s is unlikely
            
            df["GPS Speed"] = speed * 3.6 # m/s to km/h usually?
            # session expects m/s usually for "GPS Speed" but check mapping?
            # channel_mapping.json usually has "GPS Speed": "km/h" and unit "km/h"
            # let's look at existing loaders.
            # alfano: ensure_distance uses Speed.
            # ensure_distance calculates Distance += Speed * dt.
            # Convert m/s to km/h?
            # Wait, ensure_distance:
            # frame["Distance"] = (speed * (1000.0 / 3600.0) * delta).cumsum()
            # This implies 'speed' input is in km/h because it multiplies by (1000/3600) to get m/s.
            # So "GPS Speed" should be km/h.
            
        else:
            pass # We have speed

        # Ensure Distance
        # We can use the calculated dist from above or ensure_distance logic
        # If we computed dist above:
        if "Distance" not in df.columns:
             # Let's rely on standard logic
             df = ensure_distance(df, distance_keys=["GPS Distance"], speed_keys=["GPS Speed"], frequency=10.0) # Freq is approx

        if normalize:
             df = ChannelNormalizer().normalize_dataframe(df)

        meta = {
            "creator": gpx.creator,
            "name": gpx.tracks[0].name if gpx.tracks else "",
            "desc": gpx.tracks[0].description if gpx.tracks else "",
        }
        
        # determine date/time
        date_str = ""
        time_str = ""
        if gpx.time:
             date_str = gpx.time.date().isoformat()
             time_str = gpx.time.strftime("%H:%M")
        elif start_time:
             dt = datetime.fromtimestamp(start_time)
             date_str = dt.date().isoformat()
             time_str = dt.strftime("%H:%M")

        session = cls(
            table=df,
            metadata=SessionMetadata(
                device="GPX",
                event_date=date_str,
                event_time=time_str,
                tags=meta
            )
        )
        
        return session

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
            distance_threshold: Unused (kept for API compatibility).
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

        # Transform track path to meters
        lon_vals = lon_series[valid_mask].to_numpy()
        lat_vals = lat_series[valid_mask].to_numpy()
        x_vals, y_vals = WGS84_TO_WEBMERC.transform(lon_vals, lat_vals)
        times = time_series[valid_mask].to_numpy(dtype=float)

        # Transform S/F line to meters
        line_lon, line_lat = zip(*start_finish_line)
        line_x, line_y = WGS84_TO_WEBMERC.transform(np.array(line_lon), np.array(line_lat))
        
        # Define S/F segment (start to end)
        sf_p1 = (line_x[0], line_y[0])
        sf_p2 = (line_x[-1], line_y[-1])
        
        def segments_intersect(p1, p2, q1, q2):
            """Check intersection between p1-p2 and q1-q2. Returns (bool, t)."""
            px, py = p1
            rx, ry = p2[0] - px, p2[1] - py
            qx, qy = q1
            sx, sy = q2[0] - qx, q2[1] - qy

            cross = rx * sy - ry * sx
            if abs(cross) < 1e-9:
                return False, 0.0

            t = ((qx - px) * sy - (qy - py) * sx) / cross
            u = ((qx - px) * ry - (qy - py) * rx) / cross

            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                return True, t
            return False, 0.0

        crossings: list[float] = []
        last_cross = None
        
        for i in range(len(x_vals) - 1):
            p1 = (x_vals[i], y_vals[i])
            p2 = (x_vals[i+1], y_vals[i+1])
            
            if p1 == p2:
                continue
                
            intersects, t = segments_intersect(p1, p2, sf_p1, sf_p2)
            
            if intersects:
                t_cross = times[i] + t * (times[i+1] - times[i])
                
                if last_cross is not None and t_cross - last_cross < min_lap_time:
                    continue
                
                crossings.append(t_cross)
                last_cross = t_cross

        # Assign Lap Numbers
        lap_numbers = np.full(len(time_series), np.nan)
        lap_elapsed = np.full(len(time_series), np.nan)
        lap_durations: list[float] = []
        
        # Valid indices map back to original dataframe
        valid_indices = np.nonzero(valid_mask.to_numpy())[0]
        start_time = times[0]

        if not crossings:
            for i in valid_indices:
                t = time_series.iloc[i]
                lap_numbers[i] = 0
                lap_elapsed[i] = max(0.0, t - start_time)
        else:
            # Re-implement lap assignment logic based on crossings list
            lap_starts = crossings[:-1]
            last_boundary = crossings[-1]
            if len(crossings) > 1:
                lap_durations = [end - start for start, end in zip(lap_starts, crossings[1:])]
                # Insert duration for Lap 0? Usually Lap 0 has no duration as it's partial.
                # Logic below:
            
            # Use bisect for efficiency or simple loop
            import bisect
            
            # Prepare boundaries for bisect: [c0, c1, c2...]
            # Time < c0 -> Lap 0
            # c0 <= Time < c1 -> Lap 1
            # ...
            # Time >= last_c -> In Lap (often -1 or next lap)
            
            boundaries = crossings
            
            for i in valid_indices:
                t = time_series.iloc[i]
                
                if t < boundaries[0]:
                    lap_numbers[i] = 0
                    lap_elapsed[i] = max(0.0, t - start_time)
                elif t >= boundaries[-1]:
                    # After last crossing
                    lap_numbers[i] = len(boundaries) # e.g. Lap 20
                    lap_elapsed[i] = max(0.0, t - boundaries[-1])
                else:
                    # In between
                    pos = bisect.bisect_right(boundaries, t) - 1
                    # pos=0 -> between b[0] and b[1] -> Lap 1
                    lap_num = pos + 1
                    lap_numbers[i] = lap_num
                    lap_elapsed[i] = max(0.0, t - boundaries[pos])

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


def _distance_from_gps(frame: pd.DataFrame) -> np.ndarray | None:
    """Compute cumulative distance from GPS Latitude/Longitude columns (UTM projection)."""
    lat_col = next((c for c in ("GPS Latitude", "Lat.") if c in frame.columns), None)
    lon_col = next((c for c in ("GPS Longitude", "Lon.") if c in frame.columns), None)
    if lat_col is None or lon_col is None:
        return None

    lat = pd.to_numeric(frame[lat_col], errors="coerce").values
    lon = pd.to_numeric(frame[lon_col], errors="coerce").values
    valid = np.isfinite(lat) & np.isfinite(lon) & (lat != 0) & (lon != 0)
    if valid.sum() < 2:
        return None

    # Determine UTM zone from median longitude
    med_lon = np.median(lon[valid])
    utm_zone = int((med_lon + 180) / 6) + 1
    hemisphere = "north" if np.median(lat[valid]) >= 0 else "south"
    epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)

    dx = np.diff(x)
    dy = np.diff(y)
    steps = np.sqrt(dx ** 2 + dy ** 2)
    # Clamp unreasonable jumps (GPS glitches)
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
    # 1. Use existing distance column if present
    for key in distance_keys:
        if key in frame.columns and pd.to_numeric(frame[key], errors="coerce").notna().any():
            frame["Distance"] = pd.to_numeric(frame[key], errors="coerce").ffill().fillna(0.0)
            return frame

    # 2. Compute from GPS coordinates (most accurate for track matching)
    gps_dist = _distance_from_gps(frame)
    if gps_dist is not None:
        frame["Distance"] = gps_dist
        return frame

    # 3. Fallback: integrate speed over time
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


# Import here to avoid circular imports
from racing_tools.session.video_info import probe_video, VideoInfo
from typing import Optional


class VideoSession(Session):
    """Session with video synchronization capabilities."""
    
    def __init__(
        self,
        table: pd.DataFrame,
        video_path: Path,
        metadata: SessionMetadata = None,
        track: "TrackGeometry | None" = None,
        crossings: list = None,
    ):
        super().__init__(
            table=table,
            metadata=metadata if metadata is not None else SessionMetadata(),
            track=track,
            crossings=crossings if crossings is not None else [],
        )
        self.video_path = Path(video_path)
        self._video_info: Optional[VideoInfo] = None
        self.sync_offset: float = 0.0

    @classmethod
    def from_session(cls, session: Session, video_path: Path) -> "VideoSession":
        """Create VideoSession from existing Session and video path."""
        return cls(
            table=session.table,
            video_path=video_path,
            metadata=session.metadata.copy(),
            track=session.track,
            crossings=list(session.crossings),
        )

    @classmethod
    def load_with_video(cls, session_path: Path, video_path: Path, **kwargs) -> "VideoSession":
        """Load session from path and attach video."""
        session = Session.load(session_path, **kwargs)
        return cls.from_session(session, video_path)

    @property
    def info(self) -> VideoInfo:
        """Video metadata (lazy-loaded on first access)."""
        if self._video_info is None:
            if not self.video_path.exists():
                raise FileNotFoundError(f"Video file not found: {self.video_path}")
            self._video_info = probe_video(self.video_path)
        return self._video_info

    def sync(self, interactive: bool = False, force_ui: bool = False) -> float:
        """
        Determine synchronization offset.
        
        1. Checks for .sync-VIDEO_NAME.txt file.
        2. If consistent and not force_ui, uses it.
        3. If interactive or force_ui, launches sync UI.
        """
        sync_file = self.video_path.parent / f".sync-{self.video_path.name}.txt"
        
        if sync_file.exists() and not force_ui:
            try:
                content = sync_file.read_text().strip()
                self.sync_offset = float(content)
                return self.sync_offset
            except ValueError:
                pass

        if interactive:
            from racing_tools.sync_ui import run_interactive_sync
            
            vinfo = self.info
            
            print(f"[VideoSession] Launching Sync UI for {self.video_path.name}...")
            offset = run_interactive_sync(
                    video_path=self.video_path, 
                    session=self,
                    video_duration=vinfo.duration
            )
            if offset is not None:
                self.sync_offset = offset
                self.save_sync(offset)
                return offset
        
        return 0.0

    def save_sync(self, offset: float):
        """Save sync offset to sidecar file."""
        sync_file = self.video_path.parent / f".sync-{self.video_path.name}.txt"
        sync_file.write_text(f"{offset:.4f}")
        self.sync_offset = offset

    def add_video_columns(self):
        """
        Add 'Frame' and 'VideoTime' columns to the session table based on sync offset and video FPS.
        Modifies self.table in-place.
        """
        vinfo = self.info
        fps = vinfo.fps
        
        if "Time" not in self.table.columns:
            return

        times = pd.to_numeric(self.table["Time"], errors="coerce").fillna(0.0)
        video_times = times - self.sync_offset
        
        self.table["VideoTime"] = video_times
        self.table["Frame"] = (video_times * fps).round().astype(int)

    def resample_to_video(self, fps: float, trim_start: float, duration: float,
                           sync: "PiecewiseSync | float" = 0.0) -> pd.DataFrame:
        """
        Resample telemetry to match video frames exactly.

        Args:
            fps: Video frames per second.
            trim_start: Start time of the video segment (seconds).
            duration: Duration of the video segment (seconds).
            sync: PiecewiseSync mapping or float offset (legacy).

        Returns:
            pd.DataFrame: Resampled telemetry with one row per video frame.
        """
        # 1. Create target Video Times (relative to original video start)
        total_frames = int(duration * fps)
        frame_indices = np.arange(total_frames)
        target_video_times = trim_start + (frame_indices / fps)

        # 2. Convert to Telemetry Time
        if isinstance(sync, (int, float)):
            target_telemetry_times = target_video_times + sync
        else:
            target_telemetry_times = sync.video_to_telemetry(target_video_times)
        
        # 3. Interpolate
        df = self.table.copy()
        if "Time" not in df.columns:
            return pd.DataFrame(index=frame_indices)
            
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df = df.sort_values("Time")
        
        src_times = df["Time"].values
        
        resampled = pd.DataFrame({"VideoTime": target_video_times, "TelemetryTime": target_telemetry_times})
        
        interp_cols = [
            "Speed", "RPM", "Throttle", "Brake", "Steer", "G_Lat", "G_Lon", 
            "Distance", "LapNumber", "LapTime", "NormalizedX", "NormalizedY"
        ]
        
        col_mapping = {
            "Speed": ["Speed", "GPS Speed", "Wheel Speed"],
            "RPM": ["RPM", "Engine RPM"],
            "Throttle": ["Throttle", "PedalPosition", "TPS"],
            "Brake": ["Brake", "BrakePressure"],
            "Steer": ["Steer", "SteeringAngle", "Steering"],
            "Distance": ["Distance", "Distance on GPS Speed"],
            "LapNumber": ["LapNumber", "Lap Number"],
            "LapTime": ["LapTime", "Lap Time"],
        }
        
        for col in interp_cols:
            source_col = col
            if col in col_mapping:
                for candidate in col_mapping[col]:
                    if candidate in df.columns:
                        source_col = candidate
                        break
            
            if source_col in df.columns:
                src_vals = pd.to_numeric(df[source_col], errors="coerce").values
                val = np.interp(target_telemetry_times, src_times, src_vals)
                if col == "LapNumber":
                    val = np.round(val)
                resampled[col] = val
            else:
                resampled[col] = 0.0
                
        return resampled
