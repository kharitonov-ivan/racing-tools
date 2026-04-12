from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from racing_tools.session.alfano.utils import (
    DISTANCE_KEYS,
    SPEED_KEYS,
    STEP,
    detect_device,
    excel_frame,
    extract_lap_number,
    header_clock,
    infer_frequency,
    stitch_time,
)
from racing_tools.session.distance import ensure_distance
from racing_tools.session.normalizer import ChannelNormalizer
from racing_tools.session.utils import infer_datetime_from_path, name_tokens


def _to_signed16(v: int) -> int:
    return v - 65536 if v > 32767 else v


_RPM_SUB_COLS = ["RPM 1 20Hz", "RPM 2 50Hz", "RPM 3 50Hz", "RPM 4 50Hz", "RPM 5 50Hz"]
_RPM_SUB_OFFSETS = [0.02, 0.04, 0.06, 0.08, 0.10]  # seconds relative to previous row


def _expand_subchannels(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand 25Hz GPS/Speed and 50Hz RPM sub-channels into separate rows.

    Sub-channel values in row N were measured between rows N-1 and N.
    All expanded rows carry only the sub-channel value; other columns are NaN
    and will be filled by the 100Hz resampling step later.
    """
    has_gps = "Lat. 25Hz" in frame.columns and "Lon. 25Hz" in frame.columns
    has_speed = "Speed GPS 25Hz" in frame.columns
    has_rpm = all(c in frame.columns for c in _RPM_SUB_COLS)

    if not has_gps and not has_speed and not has_rpm:
        return frame

    extra_frames = []

    # GPS 25Hz midpoint: measured at -0.05s from current row
    if has_gps or has_speed:
        mid = pd.DataFrame(index=range(1, len(frame)))
        mid["Time"] = frame["Time"].values[1:] - 0.05
        mid["Partiel"] = frame["Partiel"].values[1:]
        if has_gps:
            mid["Lat."] = frame["Lat."].values[1:] + frame["Lat. 25Hz"].apply(_to_signed16).values[1:]
            mid["Lon."] = frame["Lon."].values[1:] + frame["Lon. 25Hz"].apply(_to_signed16).values[1:]
        if has_speed:
            mid["Speed GPS"] = frame["Speed GPS 25Hz"].values[1:]
        extra_frames.append(mid)

    # RPM 50Hz: 5 sub-samples at +0.02s intervals between rows N-1 and N
    if has_rpm:
        for col, offset in zip(_RPM_SUB_COLS, _RPM_SUB_OFFSETS):
            rpm_row = pd.DataFrame(index=range(1, len(frame)))
            rpm_row["Time"] = frame["Time"].values[:-1] + offset
            rpm_row["Partiel"] = frame["Partiel"].values[1:]
            rpm_row["RPM"] = frame[col].values[1:]
            extra_frames.append(rpm_row)

    if not extra_frames:
        return frame

    # Drop consumed sub-channel columns from base frame
    drop_cols = ["Lat. 25Hz", "Lon. 25Hz", "Speed GPS 25Hz"] + _RPM_SUB_COLS
    frame = frame.drop(columns=[c for c in drop_cols if c in frame.columns])

    # If RPM sub-channels are present, drop the main 10Hz RPM (Excel does the same)
    if has_rpm:
        frame = frame.drop(columns=["RPM"], errors="ignore")

    expanded = pd.concat([frame] + extra_frames, ignore_index=True)
    expanded = expanded.sort_values("Time", kind="mergesort").reset_index(drop=True)
    return expanded


def _resample_100hz(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample a Time-indexed DataFrame to a uniform 100 Hz grid."""
    frame = frame.set_index("Time")
    if len(frame) < 2:
        return frame.reset_index()
    # Merge duplicate timestamps (e.g. RPM sub-channel rows on same time as base row)
    if frame.index.duplicated().any():
        frame = frame.groupby(level=0).first().combine_first(
            frame.groupby(level=0).last()
        )
    t = frame.index
    uniform_t = np.arange(t.min(), t.max() + 0.005, 0.01)
    frame = frame.reindex(frame.index.union(uniform_t)).interpolate(method="index").reindex(uniform_t)
    frame = frame.ffill().bfill()
    frame.index.name = "Time"
    return frame.reset_index()


def _read_summary_lap_times(folder: Path) -> list[float] | None:
    """Read real lap durations (seconds) from summary CSV if present."""
    candidates = list(folder.glob("SN*.csv"))
    if not candidates:
        return None
    try:
        lines = candidates[0].read_text().splitlines()
        durations = []
        for line in lines[1:]:  # skip header line
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip().isdigit():
                durations.append(int(parts[1]) / 1000.0)
        return durations if durations else None
    except Exception:
        return None


def load_raw(folder: Path, normalize: bool = True) -> tuple[pd.DataFrame, dict]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")

    files = sorted(
        folder.glob("LAP_*.csv"),
        key=lambda p: extract_lap_number(p.name),
    )
    frames: list[pd.DataFrame] = []
    for lap_idx, p in enumerate(files, start=1):
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        df["Partiel"] = lap_idx
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No LAP_*.csv files in {folder}")

    # Build per-row timestamps using real lap durations from summary CSV
    # when available, falling back to fixed STEP otherwise.
    lap_times = _read_summary_lap_times(folder)
    time_arrays = []
    offset = 0.0
    for i, df in enumerate(frames):
        n = len(df)
        if lap_times and i < len(lap_times):
            step = lap_times[i] / n
        else:
            step = STEP
        time_arrays.append(np.arange(n) * step + offset)
        offset = time_arrays[-1][-1] + step

    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "Time", np.concatenate(time_arrays))

    # Expand 25Hz GPS/Speed and 50Hz RPM sub-channels into separate rows.
    # See experiments/alfano-log-zip-format/ALFANO7_FORMAT.md for protocol docs.
    frame = _expand_subchannels(frame)

    if normalize:
        frame = ChannelNormalizer(device_type="alfano").normalize_dataframe(frame)

    frame = _resample_100hz(frame)
    frame = ensure_distance(frame, distance_keys=DISTANCE_KEYS, speed_keys=SPEED_KEYS, frequency=100.0)

    device = detect_device(files)
    driver = ""
    venue = ""
    event_date = ""
    event_time = ""

    tokens = name_tokens(folder)
    if len(tokens) > 1:
        driver = tokens[-2]
        venue = tokens[-1]

    date_text, time_text = infer_datetime_from_path(folder)
    event_date = date_text
    event_time = time_text

    return frame, {
        "driver": driver,
        "venue": venue,
        "vehicle": "",
        "event_date": event_date,
        "event_time": event_time,
        "device": device,
        "tags": {},
    }


def load_csv(
    path_or_folder: Path,
    frequency: float = None,
    normalize: bool = True,
) -> tuple[pd.DataFrame, dict]:
    path_or_folder = Path(path_or_folder)

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

    if normalize:
        frame = ChannelNormalizer(device_type="alfano").normalize_dataframe(frame)

    frame = _resample_100hz(frame)
    frame = ensure_distance(frame, distance_keys=DISTANCE_KEYS, speed_keys=SPEED_KEYS, frequency=100.0)

    driver = ""
    venue = ""
    event_date = ""
    event_time = ""

    tokens = name_tokens(folder)
    if len(tokens) > 1:
        driver = tokens[-2]
        venue = tokens[-1]

    date_file, time_file = infer_datetime_from_path(csv_path)
    date_folder, time_folder = infer_datetime_from_path(folder)
    utc_clock = header_clock(csv_path)

    event_date = date_file or date_folder
    event_time = utc_clock or time_file or time_folder

    return frame, {
        "driver": driver,
        "venue": venue,
        "vehicle": "",
        "event_date": event_date,
        "event_time": event_time,
        "device": "Alfano6 Excel",
        "tags": {},
    }
