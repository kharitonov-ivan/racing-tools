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
    Session,
    load_mapping,
    name_tokens,
    infer_frequency,
)

def aim_session(folder: Path, frequency: float, normalize: bool) -> Session:
    return Session.load_aim_csv(folder, frequency=frequency, normalize=normalize)


def alfano_session(folder: Path, normalize: bool) -> Session:
    return Session.load_alfano_raw(folder, normalize=normalize)


def alfano_excel_session(folder: Path, normalize: bool) -> tuple[Session, float]:
    session = Session.load_alfano_csv(folder, normalize=normalize)
    freq = infer_frequency(session.table["Time"])
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
