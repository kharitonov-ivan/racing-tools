from __future__ import annotations

import atexit
import shutil
import subprocess
import sys
import tempfile
import zipfile
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer

from racing_tools.session.normalizer import ChannelNormalizer
from racing_tools.session.utils import segments_intersect
from racing_tools.track.constants import MIN_VALID_LAP_TIME

if TYPE_CHECKING:
    from racing_tools.track.track import Track

WGS84_TO_WEBMERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def create_session_from_crossings(video_info: "VideoInfo", crossing_times: list[float]) -> "Session":
    """
    Create a Session object from video info and manual crossing times (no telemetry).

    Creates one row per video frame with:
    - Time: frame_number / fps
    - Duration: cumulative time
    - LapNumber: assigned from crossing_times (before first crossing = Lap 0)

    Args:
        video_info: VideoInfo with fps, nb_frames, etc.
        crossing_times: Sorted list of lap crossing times in seconds.

    Returns:
        Session object with minimal frame-based data.
    """
    fps = video_info.fps
    nb_frames = video_info.nb_frames

    times = np.arange(nb_frames) / fps

    sorted_crossings = sorted(crossing_times) if crossing_times else []

    def get_lap_number(t: float) -> int:
        if not sorted_crossings:
            return 0
        return bisect_right(sorted_crossings, t)

    lap_numbers = np.array([get_lap_number(t) for t in times])

    df = pd.DataFrame(
        {
            "Time": times,
            "Duration": times,
            "LapNumber": lap_numbers,
        }
    )

    return Session(table=df)


@dataclass
class PiecewiseSync:
    anchors: list[tuple[float, float]]

    def __post_init__(self) -> None:
        self.anchors.sort(key=lambda a: a[0])
        self._v = np.array([a[0] for a in self.anchors])
        self._t = np.array([a[1] for a in self.anchors])
        print("[DEBUG] Synchronization anchors initialized:")
        for i, (video, telem) in enumerate(self.anchors):
            print(f"  Anchor {i + 1}: video={video:.3f}s, telemetry={telem:.3f}s")

    def video_to_telemetry(self, video_time: np.ndarray | float) -> np.ndarray:
        return np.interp(video_time, self._v, self._t)

    def telemetry_to_video(self, telem_time: np.ndarray | float) -> np.ndarray:
        return np.interp(telem_time, self._t, self._v)

    @classmethod
    def from_offset(cls, offset: float) -> "PiecewiseSync":
        return cls(anchors=[(0.0, offset), (1e6, 1e6 + offset)])

    def to_dict(self) -> dict:
        return {
            "type": "piecewise",
            "anchors_video": self._v.tolist(),
            "anchors_telem": self._t.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PiecewiseSync":
        if data.get("type") == "piecewise":
            return cls(anchors=list(zip(data["anchors_video"], data["anchors_telem"])))
        offset = data.get("offset", 0.0)
        return cls.from_offset(offset)


@dataclass
class SessionMetadata:
    driver: str = ""
    venue: str = ""
    vehicle: str = ""
    session: str = ""
    device: str = ""
    event_date: str = ""
    event_time: str = ""
    tags: dict = field(default_factory=dict)

    def copy(self, **kw) -> "SessionMetadata":
        import dataclasses

        return dataclasses.replace(self, **kw)


@dataclass
class Session:
    table: pd.DataFrame
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    track: "Track | None" = None
    crossings: list[float] = field(default_factory=list)
    crossings_gps: list[float] = field(default_factory=list)

    def __getattr__(self, name: str):
        if name in ("driver", "venue", "vehicle", "session", "device", "event_date", "event_time", "tags"):
            return getattr(self.metadata, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name in ("driver", "venue", "vehicle", "session", "device", "event_date", "event_time", "tags"):
            setattr(self.metadata, name, value)
        else:
            super().__setattr__(name, value)

    def _pick_column(self, candidates: list[str]) -> str | None:
        for name in candidates:
            if name in self.table.columns:
                return name
        return None

    def detect_crossings(self) -> list[float]:
        if self.track is None or self.track.start_finish_wgs84 is None:
            print("[DEBUG] No track or start/finish line defined.")
            return []

        lat_col = self._pick_column(["GPS Latitude", "Latitude"])
        lon_col = self._pick_column(["GPS Longitude", "Longitude"])
        if not lat_col or not lon_col:
            print("[DEBUG] Missing GPS Latitude/Longitude columns.")
            return []

        lats = pd.to_numeric(self.table[lat_col], errors="coerce").values
        lons = pd.to_numeric(self.table[lon_col], errors="coerce").values
        times = pd.to_numeric(self.table["Time"], errors="coerce").values if "Time" in self.table.columns else self.table.index.values

        sf_points = list(dict.fromkeys(self.track.start_finish_wgs84))
        if len(sf_points) < 2:
            print("[DEBUG] Insufficient start/finish line points.")
            return []
        sf_p1, sf_p2 = sf_points[0], sf_points[-1]

        sf_dx = sf_p2[0] - sf_p1[0]
        sf_dy = sf_p2[1] - sf_p1[1]
        sf_len = (sf_dx**2 + sf_dy**2) ** 0.5

        print(f"[DEBUG] Start-finish line: length={sf_len:.6f}° (≈{sf_len * 111000:.1f}m)")

        crossings = []
        prev_sign: int | None = None

        for i in range(1, len(self.table)):
            if times[i] <= 0.0:
                continue

            # Check if GPS trajectory segment crosses the start-finish line
            gps_p1 = (lons[i - 1], lats[i - 1])
            gps_p2 = (lons[i], lats[i])

            intersects, t_param, sign = segments_intersect(
                (sf_p1[0], sf_p1[1]),
                (sf_p2[0], sf_p2[1]),
                gps_p1, gps_p2
            )

            if intersects and t_param is not None:
                # Check direction consistency (skip reverse crossings from GPS jitter)
                if prev_sign is not None and sign != prev_sign:
                    continue

                # Interpolate crossing time along the GPS segment
                crossing_time = times[i - 1] + t_param * (times[i] - times[i - 1])

                print(f"[DEBUG] Valid crossing detected at time {crossing_time:.2f}s (sign={sign})")
                crossings.append(crossing_time)
                prev_sign = sign

        print(f"[DEBUG] Total valid crossings detected: {len(crossings)}")
        self.crossings = crossings
        return crossings

    def add_lap_numbers(self) -> None:
        times = pd.to_numeric(self.table["Time"], errors="coerce").values if "Time" in self.table.columns else self.table.index.values
        lap_numbers = np.zeros(len(self.table), dtype=int)
        lap_times = np.zeros(len(self.table), dtype=float)

        crossing_idx = 0
        for i, t in enumerate(times):
            while crossing_idx < len(self.crossings) and t >= self.crossings[crossing_idx]:
                crossing_idx += 1
            lap_numbers[i] = crossing_idx

            if crossing_idx == 0:
                lap_times[i] = t - times[0] if len(times) > 0 else 0.0
            else:
                lap_times[i] = t - self.crossings[crossing_idx - 1]

        self.table["LapNumber"] = lap_numbers
        self.table["LapTime"] = lap_times

        print("[DEBUG] Lap numbers and times added to the session table.")
        print(self.table[["Time", "LapNumber", "LapTime"]].head())

    def get_lap_durations(self) -> dict[int, float]:
        if not self.crossings:
            return {}
        start_time = self.table["Time"].iloc[0] if "Time" in self.table.columns else 0.0
        durations = {0: self.crossings[0] - start_time}
        for i in range(1, len(self.crossings)):
            durations[i] = self.crossings[i] - self.crossings[i - 1]
        return durations

    def _get_gps_lap_durations(self) -> dict[int, float]:
        if not self.crossings_gps:
            return {}
        durations: dict[int, float] = {}
        for i in range(1, len(self.crossings_gps)):
            durations[i] = self.crossings_gps[i] - self.crossings_gps[i - 1]
        return durations

    def get_lap_stats(self) -> list[dict]:
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

            stat: dict = {
                "id": int(lap_id),
                "time": lap_time,
                "gps_time": gps_time,
                "min_speed": None,
                "max_speed": None,
                "min_rpm": None,
                "max_rpm": None,
            }
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
        stats = self.get_lap_stats()
        valid = [s for s in stats if s["time"] and s["time"] > MIN_VALID_LAP_TIME]
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

        return (
            f"Session(\n"
            f"  Driver: {self.driver}\n"
            f"  Vehicle: {self.vehicle}\n"
            f"  Venue: {self.venue}\n"
            f"  Date: {self.event_date} {self.event_time}\n"
            f"  Device: {self.device}\n"
            f"  Rows: {len(self.table)}\n"
            f"  Channels: [{channels_str}]\n"
            f"  Metadata: {self.tags}\n"
            f")"
        )

    def copy(self, table: pd.DataFrame | None = None, **kw) -> "Session":
        frame = table.copy() if table is not None else self.table.copy()
        new_metadata = self.metadata.copy(**kw)
        return Session(frame, metadata=new_metadata)

    @classmethod
    def load(cls, path_or_folder: str | Path, **kwargs) -> "Session":
        path = Path(path_or_folder)

        if path.is_dir():
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
            if path.name.startswith("Excel_"):
                return cls.load_alfano_csv(path, **kwargs)
            return cls.load_aim_csv(path, **kwargs)

        raise ValueError(f"Unsupported file extension: {suffix}")

    @classmethod
    def _load_from_zip(cls, path: Path, **kwargs) -> "Session":
        path = Path(path)
        if not zipfile.is_zipfile(path):
            raise ValueError(f"{path} is not a valid ZIP file")

        tmp_dir = Path(tempfile.mkdtemp(prefix="racing_session_"))

        def cleanup():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        atexit.register(cleanup)

        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp_dir)

        if list(tmp_dir.glob("LAP_*.csv")):
            return cls.load_alfano_raw(tmp_dir, **kwargs)
        if list(tmp_dir.glob("Excel_*.csv")):
            return cls.load_alfano_csv(tmp_dir, **kwargs)

        return cls.load(tmp_dir, **kwargs)

    @classmethod
    def load_aim_raw(cls, path: Path, normalize: bool = True) -> "Session":
        from racing_tools.session.aim.loader import load_raw

        table, meta = load_raw(Path(path), normalize=normalize)
        return cls(
            table=table,
            metadata=SessionMetadata(
                driver=meta.get("driver", ""),
                venue=meta.get("venue", ""),
                vehicle=meta.get("vehicle", ""),
                event_date=meta.get("event_date", ""),
                event_time=meta.get("event_time", ""),
                device=meta.get("device", "AIM XRK"),
                tags=meta.get("tags", {}),
            ),
        )

    @classmethod
    def load_aim_csv(cls, path_or_folder: Path, frequency: float = 20.0, normalize: bool = True) -> "Session":
        from racing_tools.session.aim.loader import load_csv

        table, meta = load_csv(Path(path_or_folder), frequency=frequency, normalize=normalize)
        return cls(
            table=table,
            metadata=SessionMetadata(
                driver=meta.get("driver", ""),
                venue=meta.get("venue", ""),
                vehicle=meta.get("vehicle", ""),
                session=meta.get("session", ""),
                event_date=meta.get("event_date", ""),
                event_time=meta.get("event_time", ""),
                device=meta.get("device", "AIM"),
                tags=meta.get("tags", {}),
            ),
        )

    @classmethod
    def load_alfano_raw(cls, folder: Path, normalize: bool = True) -> "Session":
        from racing_tools.session.alfano.loader import load_raw

        table, meta = load_raw(Path(folder), normalize=normalize)
        return cls(
            table=table,
            metadata=SessionMetadata(
                driver=meta.get("driver", ""),
                venue=meta.get("venue", ""),
                event_date=meta.get("event_date", ""),
                event_time=meta.get("event_time", ""),
                device=meta.get("device", "Alfano"),
                tags=meta.get("tags", {}),
            ),
        )

    @classmethod
    def load_alfano_csv(cls, path_or_folder: Path, frequency: float = None, normalize: bool = True) -> "Session":
        from racing_tools.session.alfano.loader import load_csv

        table, meta = load_csv(Path(path_or_folder), frequency=frequency, normalize=normalize)
        return cls(
            table=table,
            metadata=SessionMetadata(
                driver=meta.get("driver", ""),
                venue=meta.get("venue", ""),
                event_date=meta.get("event_date", ""),
                event_time=meta.get("event_time", ""),
                device=meta.get("device", "Alfano6 Excel"),
                tags=meta.get("tags", {}),
            ),
        )

    @classmethod
    def load_gpx(cls, path: Path, normalize: bool = True) -> "Session":
        from racing_tools.session.gpx.loader import load

        table, meta = load(Path(path), normalize=normalize)
        return cls(
            table=table,
            metadata=SessionMetadata(
                event_date=meta.get("event_date", ""),
                event_time=meta.get("event_time", ""),
                device=meta.get("device", "GPX"),
                tags=meta.get("tags", {}),
            ),
        )

    def _ordered_table(self) -> pd.DataFrame:
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
        from racing_tools.session.aim.loader import motec_script

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
        for fld, flag in meta_flags.items():
            value = getattr(self, fld, "") or ""
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
        min_lap_time: float = 30.0,
    ) -> None:
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
        times = time_series[valid_mask].to_numpy(dtype=float)

        line_lon, line_lat = zip(*start_finish_line)
        line_x, line_y = WGS84_TO_WEBMERC.transform(np.array(line_lon), np.array(line_lat))

        sf_p1 = (line_x[0], line_y[0])
        sf_p2 = (line_x[-1], line_y[-1])

        crossings: list[float] = []
        last_cross = None

        for i in range(len(x_vals) - 1):
            p1 = (x_vals[i], y_vals[i])
            p2 = (x_vals[i + 1], y_vals[i + 1])

            if p1 == p2:
                continue

            intersects, t = segments_intersect(p1, p2, sf_p1, sf_p2)

            if intersects:
                t_cross = times[i] + t * (times[i + 1] - times[i])

                if last_cross is not None and t_cross - last_cross < min_lap_time:
                    continue

                crossings.append(t_cross)
                last_cross = t_cross

        lap_numbers = np.full(len(time_series), np.nan)
        lap_elapsed = np.full(len(time_series), np.nan)
        lap_durations: list[float] = []

        valid_indices = np.nonzero(valid_mask.to_numpy())[0]
        start_time = times[0]

        if not crossings:
            for i in valid_indices:
                t = time_series.iloc[i]
                lap_numbers[i] = 0
                lap_elapsed[i] = max(0.0, t - start_time)
        else:
            import bisect

            boundaries = crossings

            for i in valid_indices:
                t = time_series.iloc[i]

                if t < boundaries[0]:
                    lap_numbers[i] = 0
                    lap_elapsed[i] = max(0.0, t - start_time)
                elif t >= boundaries[-1]:
                    lap_numbers[i] = len(boundaries)
                    lap_elapsed[i] = max(0.0, t - boundaries[-1])
                else:
                    pos = bisect.bisect_right(boundaries, t) - 1
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


from racing_tools.video.video_info import probe_video, VideoInfo


class VideoSession(Session):
    def __init__(
        self,
        table: pd.DataFrame,
        video_path: Path,
        metadata: SessionMetadata = None,
        track: "Track | None" = None,
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
        return cls(
            table=session.table,
            video_path=video_path,
            metadata=session.metadata.copy(),
            track=session.track,
            crossings=list(session.crossings),
        )

    @classmethod
    def load_with_video(cls, session_path: Path, video_path: Path, **kwargs) -> "VideoSession":
        session = Session.load(session_path, **kwargs)
        return cls.from_session(session, video_path)

    @property
    def info(self) -> VideoInfo:
        if self._video_info is None:
            if not self.video_path.exists():
                raise FileNotFoundError(f"Video file not found: {self.video_path}")
            self._video_info = probe_video(self.video_path)
        return self._video_info

    def sync(self, interactive: bool = False, force_ui: bool = False) -> float:
        sync_file = self.video_path.parent / f".sync-{self.video_path.name}.txt"

        if sync_file.exists() and not force_ui:
            try:
                content = sync_file.read_text().strip()
                self.sync_offset = float(content)
                return self.sync_offset
            except ValueError:
                pass

        if interactive:
            from racing_tools.utils.sync_ui import run_interactive_sync

            vinfo = self.info

            print(f"[VideoSession] Launching Sync UI for {self.video_path.name}...")
            offset = run_interactive_sync(video_path=self.video_path, session=self, video_duration=vinfo.duration)
            if offset is not None:
                self.sync_offset = offset
                self.save_sync(offset)
                return offset

        return 0.0

    def save_sync(self, offset: float):
        sync_file = self.video_path.parent / f".sync-{self.video_path.name}.txt"
        sync_file.write_text(f"{offset:.4f}")
        self.sync_offset = offset

    def add_video_columns(self):
        vinfo = self.info
        fps = vinfo.fps

        if "Time" not in self.table.columns:
            return

        times = pd.to_numeric(self.table["Time"], errors="coerce").fillna(0.0)
        video_times = times - self.sync_offset

        self.table["VideoTime"] = video_times
        self.table["Frame"] = (video_times * fps).round().astype(int)

    def resample_to_video(self, fps: float, trim_start: float, duration: float, sync: "PiecewiseSync | float" = 0.0) -> pd.DataFrame:
        total_frames = int(duration * fps)
        frame_indices = np.arange(total_frames)
        target_video_times = trim_start + (frame_indices / fps)

        if isinstance(sync, (int, float)):
            target_telemetry_times = target_video_times + sync
        else:
            target_telemetry_times = sync.video_to_telemetry(target_video_times)

        df = self.table.copy()
        if "Time" not in df.columns:
            return pd.DataFrame(index=frame_indices)

        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df = df.sort_values("Time")

        src_times = df["Time"].values

        resampled = pd.DataFrame({"VideoTime": target_video_times, "TelemetryTime": target_telemetry_times})

        interp_cols = [
            "Speed",
            "RPM",
            "Throttle",
            "Brake",
            "Steer",
            "G_Lat",
            "G_Lon",
            "Distance",
            "LapNumber",
            "LapTime",
            "NormalizedX",
            "NormalizedY",
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
