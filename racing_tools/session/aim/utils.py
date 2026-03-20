from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd


def metadata(lines: list[str]) -> dict[str, str]:
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


def datetime_from_meta(meta: dict[str, str]) -> tuple[str, str]:
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


def frame(csv_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    lines = csv_path.read_text().splitlines()
    meta = metadata(lines)
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
    df = pd.read_csv(csv_path, skiprows=header)
    if len(df):
        token = str(df.iloc[0, 0]).replace(".", "").replace("-", "")
        if not token.isdigit():
            df = df.iloc[1:].reset_index(drop=True)
    df.columns = df.columns.str.strip().str.replace('"', "")
    return df, meta


DISTANCE_KEYS = ["Distance on GPS Speed", "Distance", "GPS Distance"]
SPEED_KEYS = ["GPS Speed", "Speed GPS", "Wheel Speed", "Rear Speed", "Vehicle Speed"]
