"""Scan video files for PTS discontinuities (gaps from broken concat).

Usage:
    uv run python -m racing_tools.video.check_pts <folder>
    uv run python -m racing_tools.video.check_pts <video.mp4>

Reports any video where consecutive packet PTS differ by more than the
expected frame interval — a sign that ffmpeg's concat demuxer with -c copy
joined GoPro chapters and left a gap (typically ~0.5s at the seam). Such
videos give wrong frame_index/fps timing and break crossing validation; they
should be re-concatenated via TS-protocol (the new path in concat.py).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV", ".MKV", ".AVI"}
# Only flag gaps larger than this — small drifts from rounding/B-frames don't
# break sync (sync_ui uses real PTS now). The broken-concat artifact is ~0.5s.
MIN_GAP_SECONDS = 0.1


def _scan_video(path: Path) -> tuple[int, list[tuple[float, float]]] | None:
    """Return (n_frames, [(pts_at_gap, dt), ...]) or None on probe failure.

    Sorts PTS first to be robust to B-frame reorder (ffprobe returns packets
    in DTS storage order, which differs from display/PTS order with B-frames).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path),
            ],
            capture_output=True, text=True, check=True, timeout=300,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    pts: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pts.append(float(line))
        except ValueError:
            continue

    if len(pts) < 2:
        return None

    arr = np.sort(np.array(pts, dtype=np.float64))
    diffs = np.diff(arr)
    gap_idx = np.where(diffs > MIN_GAP_SECONDS)[0]
    gaps = [(float(arr[i]), float(diffs[i])) for i in gap_idx]
    return len(arr), gaps


def _format_time(t: float) -> str:
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:06.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Video file or folder to scan")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recurse into subfolders")
    args = parser.parse_args()

    if args.path.is_file():
        videos = [args.path]
    elif args.path.is_dir():
        if args.recursive:
            videos = [p for p in args.path.rglob("*") if p.suffix in VIDEO_EXTENSIONS]
        else:
            videos = [p for p in args.path.iterdir() if p.suffix in VIDEO_EXTENSIONS]
        videos.sort()
    else:
        print(f"Error: {args.path} is not a file or directory", file=sys.stderr)
        return 2

    if not videos:
        print("No videos found.")
        return 0

    print(f"Scanning {len(videos)} video(s)...")
    bad: list[tuple[Path, list[tuple[float, float]]]] = []
    for v in videos:
        result = _scan_video(v)
        if result is None:
            print(f"  ?  {v}: probe failed")
            continue
        n_frames, gaps = result
        if gaps:
            bad.append((v, gaps))
            total_drift = sum(dt for _, dt in gaps) - len(gaps) * gaps[0][1]
            print(f"  ✗  {v}: {len(gaps)} gap(s), {n_frames} frames")
            for pts, dt in gaps:
                print(f"       gap @ {_format_time(pts)} (PTS={pts:.3f}s) dt={dt:.4f}s")
        else:
            print(f"  ✓  {v}: clean ({n_frames} frames)")

    print()
    if bad:
        print(f"Found {len(bad)} video(s) with PTS gaps. To fix:")
        print("  1. Re-concat from /100GOPRO/ source via the new concat.py (TS-protocol)")
        print("  2. Re-run trim if applicable")
        print("  3. Delete .crossings-*.json sidecar and re-mark crossings")
        return 1
    print("All videos clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
