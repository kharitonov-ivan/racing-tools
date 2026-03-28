#!/usr/bin/env python3
"""Batch process racing sessions from newest folders."""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from racing_tools.session.aim.utils import metadata as aim_metadata


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRACK_DIR = PROJECT_ROOT / "racing_tools" / "track" / "data" / "RIMSportKarting"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}|\w{2}-\w{2})")


def is_aim_csv(path: Path) -> bool:
    """Check if a CSV file is AIM format by reading its metadata header."""
    lines = path.read_text(errors="replace").splitlines()[:20]
    meta = aim_metadata(lines)
    return bool(meta)


def parse_folder_datetime(folder: Path) -> datetime:
    """Extract datetime from folder name like '2026-03-17_17-51_...'.

    Folders with unparseable time (XX-XX) get time 00:00.
    Folders with no date match sort to the very beginning (oldest).
    """
    m = _DATE_RE.match(folder.name)
    if not m:
        return datetime.min
    date_str = m.group(1)
    time_str = m.group(2)
    try:
        return datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H-%M")
    except ValueError:
        # Handles XX-XX or other non-numeric time
        return datetime.strptime(date_str, "%Y-%m-%d")


def find_video(folder: Path) -> Path | None:
    for ext in VIDEO_EXTS:
        files = list(folder.glob(f"*{ext}")) + list(folder.glob(f"*{ext.upper()}"))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
    return None


def _newest(files: list[Path]) -> Path:
    return max(files, key=lambda p: p.stat().st_mtime)


def _glob_ci(folder: Path, pattern: str) -> list[Path]:
    return list(folder.glob(pattern)) + list(folder.glob(pattern.upper()))


def find_telemetry(folder: Path) -> tuple[str, Path] | None:
    """Detect telemetry format and return (format_name, path).

    Priority: aim_xrk > aim_csv > alfano_zip > alfano_csv
    """
    # aim_xrk: .xrk / .xrs files
    xrk = _glob_ci(folder, "*.xrk") + _glob_ci(folder, "*.xrs")
    if xrk:
        return "aim_xrk", _newest(xrk)

    # aim_csv: CSV with AIM metadata header
    csvs = _glob_ci(folder, "*.csv")
    aim_csvs = [f for f in csvs if is_aim_csv(f)]
    if aim_csvs:
        return "aim_csv", _newest(aim_csvs)

    # alfano_zip: .zip containing alfano data
    zips = _glob_ci(folder, "*.zip")
    if zips:
        return "alfano_zip", _newest(zips)

    # alfano_csv: Excel_*.csv or LAP_*.csv
    excel = [f for f in csvs if f.name.startswith("Excel_")]
    if excel:
        return "alfano_csv", _newest(excel)

    lap = [f for f in csvs if f.name.startswith("LAP_")]
    if lap:
        return "alfano_csv", _newest(lap)

    return None


def process_folder(folder: Path, resolution: int, stabilisation: bool, telemetry_only: bool) -> bool:
    folder = Path(folder)
    result = find_telemetry(folder)
    video = None if telemetry_only else find_video(folder)

    print(f"\n{'=' * 60}")
    print(f"Processing: {folder.name}")
    print(f"{'=' * 60}")

    if not result:
        print(f"[SKIP] No telemetry found in {folder}")
        return False

    fmt, telemetry = result
    print(f"[FOUND] Telemetry: {telemetry.name} ({fmt})")
    print(f"[FOUND] Video: {video.name if video else 'None'}")

    cmd = [
        sys.executable,
        "-m",
        "racing_tools.run",
        "--telemetry",
        str(telemetry),
        "--track",
        str(TRACK_DIR),
    ]

    if not telemetry_only:
        cmd.extend(["--resolution", str(resolution)])

    if video and not telemetry_only:
        cmd.extend(["--in", str(video)])

    if stabilisation:
        cmd.append("--stabilise")

    print(f"[RUN] {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(description="Batch process racing sessions")
    p.add_argument("folders", nargs="*", default=[str(PROJECT_ROOT / "data" / "new")], help="Folders to process (default: data/new)")
    p.add_argument("--n", type=int, default=None, help="Process N newest folders")
    p.add_argument("--resolution", type=int, default=720, help="Video resolution height (default: 720)")
    p.add_argument("--stabilise", action="store_true", help="Enable video stabilisation")
    p.add_argument("--telemetry-only", action="store_true", help="Only export telemetry (skip video)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    args = p.parse_args()

    folders = [Path(f) for f in args.folders]

    all_dirs: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            if find_telemetry(folder):
                # Folder itself contains telemetry — treat as session folder
                all_dirs.append(folder)
            else:
                # Parent folder — look at subdirectories
                subdirs = [d for d in folder.iterdir() if d.is_dir()]
                all_dirs.extend(subdirs)
        else:
            all_dirs.append(folder.parent)

    # Sort by date/time parsed from folder name, newest first
    all_dirs = sorted(all_dirs, key=parse_folder_datetime, reverse=True)

    if args.n:
        all_dirs = all_dirs[: args.n]

    if args.dry_run:
        print("Would process:")
        for d in all_dirs:
            video = find_video(d)
            result = find_telemetry(d)
            fmt, telem = result if result else ("none", None)
            print(f"  {d.name}: video={video.name if video else 'None'}, telemetry={telem.name if telem else 'None'} ({fmt})")
        return 0

    success = 0
    failed = 0

    for folder in all_dirs:
        ok = process_folder(folder, args.resolution, args.stabilise, args.telemetry_only)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success} succeeded, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
