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

    if normalize:
        frame = ChannelNormalizer(device_type="alfano").normalize_dataframe(frame)
    frame = ensure_distance(frame, distance_keys=DISTANCE_KEYS, speed_keys=SPEED_KEYS, frequency=1.0 / STEP)

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
