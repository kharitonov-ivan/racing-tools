from __future__ import annotations

from dataclasses import dataclass, field
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
MOTEC = ROOT / "motec_log_generator.py"
THIRD_MOTEC = ROOT.parent / "third_party" / "MotecLogGenerator" / "motec_log_generator.py"


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
