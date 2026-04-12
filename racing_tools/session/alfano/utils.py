from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


_LAP_NUM_RE = re.compile(r"LAP_(\d+)_")
DISTANCE_KEYS = ["Distance on GPS Speed", "Distance"]
SPEED_KEYS = ["Speed GPS", "GPS Speed", "Speed rear", "Wheel Speed"]
STEP = 0.1


def extract_lap_number(filename: str) -> int:
    m = _LAP_NUM_RE.search(filename)
    return int(m.group(1)) if m else 0


def detect_device(files: list[Path]) -> str:
    for f in files:
        upper = f.name.upper()
        if "ALFANO7" in upper:
            return "Alfano7"
        if "ALFANO6" in upper:
            return "Alfano6"
    return "Alfano"


def clean_header(raw: str) -> str:
    text = " ".join(str(raw or "").split())
    if ":" in text and not text.lower().startswith("utc time"):
        text = text.split(":")[0].strip()
    return text


def header_clock(csv_path: Path) -> str:
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()
        if not first_line:
            return ""
        for cell in first_line.split(";"):
            if "UTC" in cell:
                from racing_tools.session.utils import decode_utc_clock

                return decode_utc_clock(cell.split(":")[-1])
    except Exception:
        pass
    return ""


def excel_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, sep=";", engine="python")
    frame.columns = [clean_header(col) for col in frame.columns]
    frame = frame.dropna(axis=1, how="all").dropna(how="all").copy()
    for col in frame.columns:
        series = frame[col]
        if series.dtype == object:
            # Alfano Excel CSV uses European formatting: semicolon field separator,
            # comma as thousand separator (e.g. RPM "8,198" = 8198, "13,879" = 13879).
            # Detect: thousand-separator pattern is digit,digit{3} (e.g. "8,198").
            # Decimal comma would be digit,digit{1-2} (e.g. "12,5").
            sample = series.dropna().head(20)
            has_thousands = sample.str.match(r"^\d{1,3}(,\d{3})+$").any()
            if has_thousands:
                cleaned = series.str.replace(",", "", regex=False)
                converted = pd.to_numeric(cleaned, errors="coerce")
                if converted.notna().any():
                    series = converted
        frame[col] = pd.to_numeric(series, errors="coerce")
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
