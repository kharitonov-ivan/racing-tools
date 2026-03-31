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


def has_export(folder: Path) -> bool:
    """Check if folder already has a MoTeC .ld export."""
    return bool(list(folder.glob("*.ld")))


def process_folder(folder: Path, args: argparse.Namespace) -> bool:
    folder = Path(folder)
    result = find_telemetry(folder)
    video = find_video(folder)
    want_video = args.ass or args.render

    print(f"\n{'=' * 60}")
    print(f"Processing: {folder.name}")
    print(f"{'=' * 60}")

    if not result:
        if not args.no_interactive and video:
            print(f"[INFO] No telemetry found — interactive lap marking mode")
        else:
            print(f"[SKIP] No telemetry found in {folder}")
            return False

    if args.no_overwrite and has_export(folder):
        print(f"[SKIP] Already exported (remove --no-overwrite to re-process)")
        return True

    if result:
        fmt, telemetry = result
        print(f"[FOUND] Telemetry: {telemetry.name} ({fmt})")

    if want_video and not video:
        if args.telemetry:
            print(f"[WARN] No video found, exporting telemetry only")
        else:
            print(f"[SKIP] No video found (required for --ass/--render)")
            return False

    use_video = video and want_video
    if use_video:
        print(f"[FOUND] Video: {video.name}")

    cmd = [sys.executable, "-m", "racing_tools.run"]

    if result:
        cmd.extend(["--telemetry", str(telemetry.resolve())])

    cmd.extend(["--track", str(TRACK_DIR)])

    if args.no_interactive:
        cmd.append("--no-interactive")

    if use_video:
        cmd.extend(["--in", str(video.resolve())])
        cmd.extend(["--resolution", str(args.resolution)])

    if use_video and args.ass and not args.render:
        cmd.append("--no-render")

    if args.stabilise:
        cmd.append("--stabilise")

    print(f"[RUN] {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(description="Batch process racing sessions")
    p.add_argument("folders", nargs="*", default=[str(PROJECT_ROOT / "data" / "new")], help="Folders to process (default: data/new)")
    p.add_argument("--n", type=int, default=None, help="Process N newest folders")
    p.add_argument("--telemetry", action="store_true", help="Export telemetry (.ld)")
    p.add_argument("--ass", action="store_true", help="Generate ASS overlay for source video")
    p.add_argument("--render", action="store_true", help="Render video with ASS overlay and trimming")
    p.add_argument("--resolution", type=int, default=720, help="Video resolution height (default: 720)")
    p.add_argument("--stabilise", action="store_true", help="Enable video stabilisation")
    p.add_argument("--no-overwrite", action="store_true", help="Skip folders that already have exports")
    p.add_argument("--no-interactive", action="store_true", help="Skip all interactive prompts (crossings, lap marking)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    args = p.parse_args()

    # Default: telemetry only if no stage flags specified
    if not args.telemetry and not args.ass and not args.render:
        args.telemetry = True

    folders = [Path(f) for f in args.folders]

    all_dirs: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            is_session = find_telemetry(folder) or (not args.no_interactive and find_video(folder))
            if is_session:
                # Folder itself is a session folder
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
        ok = process_folder(folder, args)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success} succeeded, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
