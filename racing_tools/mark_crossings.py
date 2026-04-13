#!/usr/bin/env python3
"""Pre-mark video lap crossings for batch processing.

Opens the interactive sync UI for each session's video, saves crossings
to sidecar files so that batch_process.py --no-interactive can pick them up.
"""

import argparse
from pathlib import Path

from racing_tools.batch_process import (
    PROJECT_ROOT,
    find_telemetry,
    find_video,
    parse_folder_datetime,
)
from racing_tools.utils.sync_ui import run_manual_lap_marking
from racing_tools.video.trim import VideoSidecar


def discover_sessions(folders: list[Path]) -> list[tuple[Path, Path]]:
    """Find (folder, video) pairs from input folders."""
    all_dirs: list[Path] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        is_session = find_telemetry(folder) or find_video(folder)
        if is_session:
            all_dirs.append(folder)
        else:
            all_dirs.extend(d for d in folder.iterdir() if d.is_dir())

    all_dirs = sorted(all_dirs, key=parse_folder_datetime, reverse=True)

    sessions = []
    for d in all_dirs:
        video = find_video(d)
        if video:
            sessions.append((d, video))
    return sessions


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-mark video lap crossings for batch processing")
    p.add_argument("folders", nargs="*", default=[str(PROJECT_ROOT / "data" / "new")], help="Session folders or parent dirs (default: data/new)")
    p.add_argument("--n", type=int, default=None, help="Process N newest sessions")
    p.add_argument("--force", action="store_true", help="Re-mark sessions that already have crossings")
    p.add_argument("--dry-run", action="store_true", help="Show sessions and their sidecar status")
    args = p.parse_args()

    sessions = discover_sessions([Path(f) for f in args.folders])

    if args.n:
        sessions = sessions[: args.n]

    if not sessions:
        print("No sessions with video found.")
        return 0

    if args.dry_run:
        print(f"Found {len(sessions)} sessions with video:")
        for folder, video in sessions:
            sidecar = VideoSidecar.load(video, "crossings")
            n = len(sidecar.get("times", []))
            status = f"{n} crossings" if sidecar.exists else "no crossings"
            print(f"  {folder.name}: {video.name} ({status})")
        return 0

    to_mark = []
    skipped = 0
    for folder, video in sessions:
        sidecar = VideoSidecar.load(video, "crossings")
        if sidecar.exists and not args.force:
            n = len(sidecar.get("times", []))
            print(f"[SKIP] {folder.name}: already has {n} crossings")
            skipped += 1
        else:
            to_mark.append((folder, video, sidecar))

    if not to_mark:
        print(f"All {skipped} sessions already have crossings (use --force to re-mark)")
        return 0

    marked = 0
    for i, (folder, video, sidecar) in enumerate(to_mark, 1):
        print(f"\n[{i}/{len(to_mark)}] {folder.name}")
        print(f"  Video: {video.name}")

        existing = sidecar.get("times", []) if sidecar.exists else None
        times = run_manual_lap_marking(str(video), start_time=0.0, existing_boundaries=existing or None)

        if times:
            VideoSidecar(video, "crossings").save({"times": times})
            marked += 1
            print(f"  Saved {len(times)} crossings")
        else:
            print("  Skipped (cancelled)")

    print(f"\nDone: {marked}/{len(to_mark)} marked, {skipped} already had crossings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
