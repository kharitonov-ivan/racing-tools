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


def _expand_25hz(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Interleave 25Hz GPS/Speed midpoint samples with 10Hz rows to get ~20Hz.

    Returns (expanded_frame, effective_frequency_hz).
    If no 25Hz columns are present, returns the frame unchanged at 10Hz.
    """
    has_gps = "Lat. 25Hz" in frame.columns and "Lon. 25Hz" in frame.columns
    has_speed = "Speed GPS 25Hz" in frame.columns

    if not has_gps and not has_speed:
        return frame, 1.0 / STEP

    # Build midpoint rows from row 1 onwards (row 0 has no previous row)
    mid = pd.DataFrame(index=range(1, len(frame)))
    mid["Time"] = frame["Time"].values[1:] - 0.05

    # Copy Partiel (lap number) from the current row
    mid["Partiel"] = frame["Partiel"].values[1:]

    if has_gps:
        lat_raw = frame["Lat."].values
        lon_raw = frame["Lon."].values
        lat_delta = frame["Lat. 25Hz"].apply(_to_signed16).values
        lon_delta = frame["Lon. 25Hz"].apply(_to_signed16).values
        mid["Lat."] = lat_raw[1:] + lat_delta[1:]
        mid["Lon."] = lon_raw[1:] + lon_delta[1:]

    if has_speed:
        mid["Speed GPS"] = frame["Speed GPS 25Hz"].values[1:]

    # For other columns (RPM, Altitude, Gf, Orientation, etc.) interpolate later via NaN
    # Tag rows for identification
    frame = frame.copy()
    frame["_is_mid"] = False
    mid["_is_mid"] = True

    # Interleave and sort by time
    expanded = pd.concat([frame, mid], ignore_index=True)
    expanded = expanded.sort_values("Time", kind="mergesort").reset_index(drop=True)

    # Interpolate non-GPS columns that are NaN in midpoint rows
    for col in expanded.columns:
        if col in ("Time", "Partiel", "_is_mid", "Lat.", "Lon.", "Speed GPS",
                   "Lat. 25Hz", "Lon. 25Hz", "Speed GPS 25Hz"):
            continue
        if expanded[col].dtype.kind in "fi":
            expanded[col] = expanded[col].interpolate(method="linear")

    # Drop consumed 25Hz columns and tag
    for col in ("Lat. 25Hz", "Lon. 25Hz", "Speed GPS 25Hz", "_is_mid"):
        if col in expanded.columns:
            expanded.drop(columns=col, inplace=True)

    return expanded, 1.0 / 0.05


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

    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "Time", np.arange(len(frame)) * STEP)

    # Expand 25Hz GPS/Speed sub-channels to ~20Hz by interleaving midpoint samples.
    # 25Hz value in row N was measured between rows N-1 and N (at time - 0.05s).
    # See experiments/alfano-log-zip-format/ALFANO7_FORMAT.md for protocol docs.
    frame, effective_freq = _expand_25hz(frame)

    if normalize:
        frame = ChannelNormalizer(device_type="alfano").normalize_dataframe(frame)
    frame = ensure_distance(frame, distance_keys=DISTANCE_KEYS, speed_keys=SPEED_KEYS, frequency=effective_freq)

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
    # TODO: Fix European decimal format handling in Alfano7 Excel CSV export.
    # RPM and Orientation use comma (4,562) while Speed GPS and others use point (28.1).
    # Current code parses RPM as strings, resulting in NaN after to_numeric().
    # Need to either: 1) Use decimal=',' in excel_frame() and post-process point-separated cols,
    # or 2) Parse numeric columns individually with appropriate decimal separators.
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

    freq = frequency
    if freq is None:
        freq = infer_frequency(frame["Time"])

    if normalize:
        frame = ChannelNormalizer(device_type="alfano").normalize_dataframe(frame)
    frame = ensure_distance(frame, distance_keys=DISTANCE_KEYS, speed_keys=SPEED_KEYS, frequency=freq)

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
