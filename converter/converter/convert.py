#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from session import (
    ChannelNormalizer,
    Session,
    ensure_distance,
    load_mapping,
    name_tokens,
)

DISTANCE_AIM = ["Distance on GPS Speed", "Distance", "GPS Distance"]
SPEED_AIM = ["GPS Speed", "Speed GPS", "Wheel Speed", "Rear Speed", "Vehicle Speed"]
DISTANCE_ALFANO = ["Distance on GPS Speed", "Distance"]
SPEED_ALFANO = ["Speed GPS", "GPS Speed", "Speed rear", "Wheel Speed"]
DISTANCE_EXCEL = ["Distance"]
SPEED_EXCEL = ["Speed GPS", "Speed rear", "Wheel Speed"]
ALFANO_STEP = 0.1
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


def aim_session(folder: Path, frequency: float, normalize: bool) -> Session:
    csv_path = folder / "aim.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    frame, meta = aim_frame(csv_path)
    frame = ensure_distance(frame, distance_keys=DISTANCE_AIM, speed_keys=SPEED_AIM, frequency=frequency)
    if normalize:
        frame = ChannelNormalizer().normalize_dataframe(frame)

    session = Session(frame, device="AIM")
    tokens = name_tokens(folder)
    if len(tokens) > 3:
        session.venue = tokens[3]
    if len(tokens) > 4:
        session.vehicle = tokens[4]
    if len(tokens) > 5:
        session.driver = tokens[5]
    if tokens:
        session.session = tokens[-1]

    if meta:
        session.tags.update({"aim_meta": meta})
        if meta.get("Racer"):
            session.driver = meta["Racer"]
        if meta.get("Vehicle"):
            session.vehicle = meta["Vehicle"]
        if meta.get("Session"):
            session.session = meta["Session"]
        date_text, time_text = aim_datetime(meta)
        if date_text:
            session.event_date = date_text
        if time_text:
            session.event_time = time_text
    if not session.event_date or not session.event_time:
        date_token, time_token = infer_datetime_from_path(folder)
        if date_token and not session.event_date:
            session.event_date = date_token
        if time_token and not session.event_time:
            session.event_time = time_token
    return session


def alfano_session(folder: Path, normalize: bool) -> Session:
    files = sorted(folder.glob("LAP_*.csv"))
    frames = [pd.read_csv(path) for path in files if path.is_file()]
    if not frames:
        raise FileNotFoundError("LAP_*.csv missing")
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "Time", np.arange(len(frame)) * ALFANO_STEP)
    frame = ensure_distance(frame, distance_keys=DISTANCE_ALFANO, speed_keys=SPEED_ALFANO, frequency=1.0 / ALFANO_STEP)
    if normalize:
        frame = ChannelNormalizer(device_type="alfano").normalize_dataframe(frame)

    session = Session(frame, device="Alfano6")
    tokens = name_tokens(folder)
    if len(tokens) > 1:
        session.driver = tokens[-2]
        session.venue = tokens[-1]
    date_text, time_text = infer_datetime_from_path(folder)
    if date_text:
        session.event_date = date_text
    if time_text:
        session.event_time = time_text
    return session


def clean_header(raw: str) -> str:
    text = " ".join(str(raw or "").split())
    if ":" in text and not text.lower().startswith("utc time"):
        text = text.split(":")[0].strip()
    return text


def excel_header_clock(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return ""
    for cell in first_line.split(";"):
        if "UTC" in cell:
            return decode_utc_clock(cell.split(":")[-1])
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


def alfano_excel_session(folder: Path, normalize: bool) -> tuple[Session, float]:
    files = sorted(folder.glob("Excel_*.csv"))
    if not files:
        raise FileNotFoundError("Excel_*.csv missing")
    csv_path = files[0]
    frame = excel_frame(csv_path)
    frame = stitch_time(frame)
    freq = infer_frequency(frame["Time"])
    frame = ensure_distance(frame, distance_keys=DISTANCE_EXCEL, speed_keys=SPEED_EXCEL, frequency=freq)
    if normalize:
        frame = ChannelNormalizer().normalize_dataframe(frame)

    session = Session(frame, device="Alfano6 Excel")
    tokens = name_tokens(folder)
    if len(tokens) > 1:
        session.driver = tokens[-2]
        session.venue = tokens[-1]
    date_file, time_file = infer_datetime_from_path(csv_path)
    date_folder, time_folder = infer_datetime_from_path(folder)
    utc_clock = excel_header_clock(csv_path)
    session.event_date = date_file or date_folder or session.event_date
    session.event_time = utc_clock or time_file or time_folder or session.event_time
    return session, freq


def run_session(
    session: Session,
    *,
    folder: Path,
    freq: float,
    tmp_prefix: str,
    output: str | None,
    keep: bool,
) -> Path:
    tmp = folder / f"tmp_{tmp_prefix}_{folder.name}.csv"
    out = Path(output).expanduser() if output else folder / f"{folder.name}.ld"
    session.to_motec(output=out, frequency=freq, csv_path=tmp, keep_csv=keep)
    print(out)
    return out


def detect(folder: Path) -> str | None:
    if (folder / "aim.csv").is_file():
        return "aim"
    if list(folder.glob("LAP_*.csv")):
        return "alfano"
    if list(folder.glob("Excel_*.csv")):
        return "alfano_excel"
    return None


def handle_aim(args):
    folder = Path(args.directory).expanduser()
    if not folder.is_dir():
        raise SystemExit(f"{folder} missing")
    session = aim_session(folder, args.frequency, not args.raw)
    run_session(
        session,
        folder=folder,
        freq=args.frequency,
        tmp_prefix="aim",
        output=args.output,
        keep=args.keep,
    )


def handle_alfano(args):
    folder = Path(args.directory).expanduser()
    if not folder.is_dir():
        raise SystemExit(f"{folder} missing")
    session = alfano_session(folder, not args.raw)
    run_session(
        session,
        folder=folder,
        freq=args.frequency,
        tmp_prefix="alfano",
        output=args.output,
        keep=args.keep,
    )


def handle_alfano_excel(args):
    folder = Path(args.directory).expanduser()
    if not folder.is_dir():
        raise SystemExit(f"{folder} missing")
    session, inferred = alfano_excel_session(folder, not args.raw)
    freq = args.frequency or inferred
    run_session(
        session,
        folder=folder,
        freq=freq,
        tmp_prefix="alfano_excel",
        output=args.output,
        keep=args.keep,
    )


def handle_batch(args):
    root = Path(args.directory).expanduser()
    if not root.is_dir():
        raise SystemExit(f"{root} missing")

    if not args.keep_tmp:
        for tmp in root.rglob("tmp_*.csv"):
            try:
                tmp.unlink(missing_ok=True)
            except PermissionError:
                print(f"skip locked tmp: {tmp}")

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        kind = detect(folder)
        if kind == "aim":
            session = aim_session(folder, args.aim_frequency, True)
            run_session(
                session,
                folder=folder,
                freq=args.aim_frequency,
                tmp_prefix="aim",
                output=None,
                keep=args.keep_tmp,
            )
        elif kind == "alfano":
            session = alfano_session(folder, True)
            run_session(
                session,
                folder=folder,
                freq=args.alfano_frequency,
                tmp_prefix="alfano",
                output=None,
                keep=args.keep_tmp,
            )
        elif kind == "alfano_excel":
            session, inferred = alfano_excel_session(folder, True)
            run_session(
                session,
                folder=folder,
                freq=args.excel_frequency or inferred,
                tmp_prefix="alfano_excel",
                output=None,
                keep=args.keep_tmp,
            )


def handle_mapping(_args):
    rows = load_mapping()["standard_channels"].values()
    for item in sorted(rows, key=lambda r: r["standard_name"]):
        name = item["standard_name"]
        unit = item["unit"]
        aliases = ", ".join(item["aliases"])
        print(f"{name:24} | {unit:6} | {aliases}")


def main():
    parser = argparse.ArgumentParser(description="Telemetry converters")
    sub = parser.add_subparsers(dest="cmd", required=True)

    aim_cmd = sub.add_parser("aim", help="convert AIM session")
    aim_cmd.add_argument("directory")
    aim_cmd.add_argument("--frequency", type=float, default=20.0)
    aim_cmd.add_argument("--output")
    aim_cmd.add_argument("--raw", action="store_true", help="skip normalization")
    aim_cmd.add_argument("--keep", action="store_true", help="keep tmp csv")
    aim_cmd.set_defaults(func=handle_aim)

    alfano_cmd = sub.add_parser("alfano", help="convert Alfano LAP files")
    alfano_cmd.add_argument("directory")
    alfano_cmd.add_argument("--frequency", type=float, default=10.0)
    alfano_cmd.add_argument("--output")
    alfano_cmd.add_argument("--raw", action="store_true")
    alfano_cmd.add_argument("--keep", action="store_true")
    alfano_cmd.set_defaults(func=handle_alfano)

    excel_cmd = sub.add_parser("alfano-excel", help="convert Alfano Excel export")
    excel_cmd.add_argument("directory")
    excel_cmd.add_argument("--frequency", type=float)
    excel_cmd.add_argument("--output")
    excel_cmd.add_argument("--raw", action="store_true")
    excel_cmd.add_argument("--keep", action="store_true")
    excel_cmd.set_defaults(func=handle_alfano_excel)

    batch_cmd = sub.add_parser("batch", help="convert each session under a folder")
    batch_cmd.add_argument("directory")
    batch_cmd.add_argument("--keep-tmp", action="store_true")
    batch_cmd.add_argument("--aim-frequency", type=float, default=20.0)
    batch_cmd.add_argument("--alfano-frequency", type=float, default=10.0)
    batch_cmd.add_argument("--excel-frequency", type=float)
    batch_cmd.set_defaults(func=handle_batch)

    map_cmd = sub.add_parser("mapping", help="list normalized channel names")
    map_cmd.set_defaults(func=handle_mapping)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
