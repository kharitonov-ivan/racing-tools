from __future__ import annotations

import argparse
import ffmpeg
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import List, Optional, TypedDict


class VideoData(TypedDict):
    """Type annotation for video metadata dictionary.

    Keys:
        file: Path to video file
        duration: Video duration in seconds
        start_time: Optional datetime when video started
        end_time: Optional datetime when video ended
    """

    file: Path
    duration: float
    start_time: Optional[datetime]
    end_time: Optional[datetime]


import pytesseract
from PIL import Image, ImageOps
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from racing_tools.video.video_info import probe_video

console = Console()

# Constants
CONTINUITY_TOLERANCE_SECONDS = 4.0
BRIDGE_TOLERANCE_SECONDS = 5.0
DEFAULT_CROP_RATIOS = (0.60, 0.90, 1.0, 1.0)


def check_system_dependencies():
    """Check that required system binaries (tesseract, ffmpeg/ffprobe) are installed."""
    missing = []

    try:
        subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        missing.append("tesseract-ocr")

    for binary in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run(
                [binary, "-version"],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            missing.append(binary)

    if missing:
        console.print(f"[red bold]Missing system dependencies: {', '.join(missing)}[/red bold]")
        console.print(
            "[yellow]Install them:\n"
            "  Ubuntu/Debian: sudo apt install tesseract-ocr ffmpeg\n"
            "  macOS:         brew install tesseract ffmpeg\n"
            "  Docker:        docker compose run --rm app uv run ...[/yellow]"
        )
        sys.exit(1)


def extract_frame(file_path: Path, time_offset: float) -> Optional[Image.Image]:
    """Extract single frame from video at given time offset."""
    try:
        out, _ = (
            ffmpeg.input(str(file_path), ss=time_offset)
            .output("pipe:", vframes=1, format="image2pipe", vcodec="png")
            .run(capture_stdout=True, capture_stderr=True)
        )
        return Image.open(BytesIO(out))
    except (ffmpeg.Error, OSError, IOError):
        return None


def parse_timestamp(text: str) -> Optional[datetime]:
    text = text.strip()
    time_match = re.search(r"(\d{2})[:\'\.](\d{2})[:\'\.](\d{2})", text)
    date_match = re.search(r"(\d{1,4})[/\-\.](\d{1,2})[/\-\.](\d{1,4})", text)

    if not (time_match and date_match):
        return None

    try:
        h, m, s = map(int, time_match.groups())
        d1, d2, d3 = map(int, date_match.groups())

        if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
            return None

        # Heuristic date parsing
        year, month, day = 0, 0, 0

        def is_valid(y, m, d):
            try:
                datetime(y, m, d)
                return True
            except ValueError:
                return False

        if d1 > 1000 and is_valid(d1, d2, d3):  # YYYY-MM-DD
            year, month, day = d1, d2, d3
        elif d3 > 1000:  # MM-DD-YYYY or DD-MM-YYYY
            # User specified MM/DD/YYYY, so try that first
            if is_valid(d3, d1, d2):
                year, month, day = d3, d1, d2  # MM-DD-YYYY
            elif is_valid(d3, d2, d1):
                year, month, day = d3, d2, d1  # DD-MM-YYYY
        else:  # Ambiguous 2-digit year
            y = 2000 + d3 if d3 < 100 else d3
            if is_valid(y, d1, d2):
                year, month, day = y, d1, d2  # MM-DD-YY
            elif is_valid(y, d2, d1):
                year, month, day = y, d2, d1  # DD-MM-YY

        if year > 0:
            return datetime(year, month, day, h, m, s)
    except ValueError:
        pass
    return None


# OCR Crop Ratios for bottom-right timestamp (date + time)
# Resolution-specific crop parameters: (left, top, right, bottom)
CROP_RATIOS = {
    # 1920x1080 (Full HD) - tighter crop, timestamp on far right
    (1920, 1080): (0.80, 0.935, 1.0, 1.0),
    # 1280x720 (HD) - wider crop needed to capture full date+time
    (1280, 720): (0.60, 0.90, 1.0, 1.0),
}
# Default fallback for unknown resolutions
DEFAULT_CROP_RATIOS = (0.60, 0.90, 1.0, 1.0)


def detect_timestamp_from_image(image: Image.Image, debug_save_path: Optional[Path] = None) -> Optional[datetime]:
    w, h = image.size
    configs = []

    # Get resolution-specific crop ratios
    ratios = CROP_RATIOS.get((w, h), DEFAULT_CROP_RATIOS)
    left_r, top_r, right_r, bottom_r = ratios

    # Generate configs: (crop_box, threshold, invert, psm)
    crop_box = (
        int(w * left_r),
        int(h * top_r),
        int(w * right_r),
        int(h * bottom_r),
    )
    crops = [crop_box]
    thresholds = [200, 220, 180]

    for crop in crops:
        for thresh in thresholds:
            for invert in [False, True]:
                for psm in [6, 11, 3]:
                    configs.append((crop, thresh, invert, psm))

    for crop, thresh, invert, psm in configs:
        processed = image.crop(crop).convert("L").point(lambda p: 255 if p > thresh else 0)
        if invert:
            processed = ImageOps.invert(processed)

        try:
            text = pytesseract.image_to_string(processed, config=f"--psm {psm}")
            dt = parse_timestamp(text)
            if dt:
                if debug_save_path:
                    processed.save(debug_save_path)
                return dt
        except (pytesseract.TesseractError, ValueError):
            continue

    if debug_save_path:
        # Save the last processed image as failure debug
        fail_path = debug_save_path.parent / f"{debug_save_path.stem}_fail.jpg"
        image.save(fail_path)

    return None


def sample_timestamps(
    file_path: Path,
    duration: float,
    fps: float,
    num_samples: int = 10,
    debug_folder: Optional[Path] = None,
) -> List[Tuple[float, datetime]]:
    """
    Samples random frames from the video and attempts to detect timestamps.
    Returns a list of (time_offset, detected_datetime).
    """
    samples = []
    # Always include start and end (with some buffer)
    offsets = [0.0, max(0.0, duration - 1.0)]

    # Add random samples
    for _ in range(num_samples):
        offsets.append(random.uniform(0.0, duration))

    offsets = sorted(list(set(offsets)))

    # Use tqdm for sampling progress, leave=False to clear after completion
    for i, offset in enumerate(tqdm(offsets, desc=f"Sampling {file_path.name}", leave=False)):
        img = extract_frame(file_path, offset)
        if not img:
            continue

        debug_path = debug_folder / f"{file_path.stem}_sample_{i}_{offset:.1f}.jpg" if debug_folder else None
        dt = detect_timestamp_from_image(img, debug_path)

        if dt:
            samples.append((offset, dt))

    return samples


def estimate_start_time(samples: List[Tuple[float, datetime]], fps: float) -> Optional[datetime]:
    """
    Estimates the video start time from samples using robust consensus (RANSAC-like).
    """
    if not samples:
        return None

    # Calculate implied start time for each sample
    # implied_start = sample_time - offset
    # Note: offset is in seconds. sample_time is datetime.
    candidates = []
    for offset, dt in samples:
        candidates.append(dt - timedelta(seconds=offset))

    # Outlier rejection:
    # We look for the largest cluster of start times that are within a small tolerance (e.g. 5 seconds)
    # Since we might have year/month errors, we first group by date.

    # 1. Group by Date
    date_counts = Counter(c.date() for c in candidates)
    if not date_counts:
        return None

    most_common_date = date_counts.most_common(1)[0][0]
    valid_candidates = [c for c in candidates if c.date() == most_common_date]

    if not valid_candidates:
        return None

    # 2. Find consensus on time
    # We can use a sliding window or just pairwise distances.
    # Let's use a simple clustering: sort and find longest sequence with diff < tolerance
    valid_candidates.sort()

    best_cluster = []
    current_cluster = []
    tolerance = timedelta(seconds=5)

    for c in valid_candidates:
        if not current_cluster:
            current_cluster.append(c)
            continue

        # Check if consistent with cluster mean or just last item?
        # Using last item is simple for chaining.
        if c - current_cluster[-1] < tolerance:
            current_cluster.append(c)
        else:
            if len(current_cluster) > len(best_cluster):
                best_cluster = current_cluster
            current_cluster = [c]

    if len(current_cluster) > len(best_cluster):
        best_cluster = current_cluster

    if not best_cluster:
        return None

    # Return the median of the best cluster
    return best_cluster[len(best_cluster) // 2]


def get_video_files(folder: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"}
    return sorted([f for f in folder.iterdir() if f.suffix in exts])


def analyze_video(file_path: Path, debug_folder: Optional[Path]) -> VideoData:
    """Analyze video file to extract metadata and timestamp samples."""
    try:
        info = probe_video(file_path)
        duration, fps, nb_frames = info.duration, info.fps, info.nb_frames
    except (FileNotFoundError, KeyError, AttributeError):
        duration, fps, nb_frames = 0.0, 0.0, 0

    # Sample timestamps
    samples = sample_timestamps(file_path, duration, fps, num_samples=50, debug_folder=debug_folder)

    start_time = estimate_start_time(samples, fps)

    end_time = None
    if start_time:
        end_time = start_time + timedelta(seconds=duration)

    return VideoData(
        file=file_path,
        duration=duration,
        start_time=start_time,
        end_time=end_time,
    )


def check_theoretical_continuity(prev: VideoData, curr: VideoData) -> bool:
    """Check if prev start + duration matches curr start."""
    if not (prev["start_time"] and curr["start_time"]):
        return False

    expected = prev["start_time"] + timedelta(seconds=prev["duration"])
    diff = (curr["start_time"] - expected).total_seconds()
    return abs(diff) < CONTINUITY_TOLERANCE_SECONDS


def check_explicit_continuity(prev: VideoData, curr: VideoData) -> bool:
    """Check if prev end matches curr start."""
    if not (prev["end_time"] and curr["start_time"]):
        return False

    return abs((curr["start_time"] - prev["end_time"]).total_seconds()) < CONTINUITY_TOLERANCE_SECONDS


def check_gap_filling(prev: VideoData, curr: VideoData) -> bool:
    """Check if curr can fill gap between prev end and curr end."""
    if not (prev["end_time"] and curr["end_time"]):
        return False

    expected_start = curr["end_time"] - timedelta(seconds=curr["duration"])
    if abs((expected_start - prev["end_time"]).total_seconds()) >= CONTINUITY_TOLERANCE_SECONDS:
        return False

    if not curr["start_time"]:
        curr["start_time"] = expected_start
    elif abs((curr["start_time"] - expected_start).total_seconds()) > CONTINUITY_TOLERANCE_SECONDS:
        console.print(f"[yellow]Correcting start time for {curr['file'].name} based on end time[/yellow]")
        curr["start_time"] = expected_start
    return True


def check_date_correction(prev: VideoData, curr: VideoData) -> bool:
    """Check if curr date can be corrected to match continuity."""
    if not (prev["start_time"] and curr["start_time"]):
        return False

    expected = prev["start_time"] + timedelta(seconds=prev["duration"])

    try:
        # Try correcting date while keeping time
        curr_corrected = curr["start_time"].replace(year=expected.year, month=expected.month, day=expected.day)

        if abs((curr_corrected - expected).total_seconds()) < CONTINUITY_TOLERANCE_SECONDS:
            console.print(f"[yellow]Correcting date for {curr['file'].name}: {curr['start_time'].date()} -> {expected.date()}[/yellow]")
            curr["start_time"] = curr_corrected
            if curr["end_time"]:
                curr["end_time"] = curr["end_time"].replace(year=expected.year, month=expected.month, day=expected.day)
            return True

        # Handle day rollover (e.g. expected 23:59, curr 00:01)
        for offset in [1, -1]:
            check_date = expected.date() + timedelta(days=offset)
            curr_corrected = curr["start_time"].replace(year=check_date.year, month=check_date.month, day=check_date.day)
            if abs((curr_corrected - expected).total_seconds()) < CONTINUITY_TOLERANCE_SECONDS:
                console.print(f"[yellow]Correcting date (rollover) for {curr['file'].name}: {curr['start_time'].date()} -> {check_date}[/yellow]")
                curr["start_time"] = curr_corrected
                if curr["end_time"]:
                    curr["end_time"] = curr["end_time"].replace(
                        year=check_date.year,
                        month=check_date.month,
                        day=check_date.day,
                    )
                return True

    except ValueError:
        pass

    return False


def is_continuous(prev: VideoData, curr: VideoData) -> bool:
    """Check if two videos are continuous using multiple strategies."""
    # Try each continuity check in order
    if check_theoretical_continuity(prev, curr):
        return True

    if check_explicit_continuity(prev, curr):
        return True

    if check_gap_filling(prev, curr):
        return True

    if check_date_correction(prev, curr):
        return True

    return False


def group_videos(video_data: List[VideoData]) -> List[List[VideoData]]:
    # Initial pass to fill missing times
    for v in video_data:
        if v["start_time"] and not v["end_time"]:
            v["end_time"] = v["start_time"] + timedelta(seconds=v["duration"])
        elif v["end_time"] and not v["start_time"]:
            v["start_time"] = v["end_time"] - timedelta(seconds=v["duration"])

    groups = []
    current_group = []

    i = 0
    while i < len(video_data):
        v = video_data[i]

        if not current_group:
            current_group.append(v)
            i += 1
            continue

        prev = current_group[-1]

        # Check continuity
        if is_continuous(prev, v):
            current_group.append(v)
            i += 1
        else:
            # Try Bridge Logic: Check if v fits between prev and next
            # Even if v has timestamps, if they are wrong (discontinuous), we might want to override them.
            # We check if we have a next video and if the gap matches v's duration.
            bridged = False
            if i + 1 < len(video_data):
                next_v = video_data[i + 1]
                # We need reliable timestamps for prev and next
                # prev is in current_group, so it should be reliable (or at least consistent with group)
                # next_v might be the start of a new group, but we check if it fits the gap.

                if next_v["start_time"] and prev["end_time"]:
                    # Check gap duration
                    gap = (next_v["start_time"] - prev["end_time"]).total_seconds()
                    # Expected gap is v['duration']
                    if abs(gap - v["duration"]) < BRIDGE_TOLERANCE_SECONDS:
                        console.print(
                            f"[yellow]Bridging gap (override) for {v['file'].name} (duration={v['duration']:.1f}s, gap={gap:.1f}s)[/yellow]"
                        )
                        # Interpolate timestamps
                        v["start_time"] = prev["end_time"]
                        v["end_time"] = next_v["start_time"]
                        current_group.append(v)

                        # We also need to add next_v to the group
                        current_group.append(next_v)
                        i += 2
                        bridged = True

            if not bridged:
                # If bridge failed, start new group
                groups.append(current_group)
                current_group = [v]
                i += 1

    if current_group:
        groups.append(current_group)

    # Fill gaps within groups (for minor missing timestamps)
    for group in groups:
        for i in range(1, len(group)):
            if group[i - 1]["end_time"] and not group[i]["start_time"]:
                group[i]["start_time"] = group[i - 1]["end_time"]

    return groups


def export_group(group: List[VideoData], output_folder: Path) -> None:
    first = group[0]
    name = first["start_time"].strftime("%Y-%m-%d_%H-%M-%S") if first["start_time"] else f"unknown_{first['file'].stem}"
    out_path = output_folder / f"{name}.mp4"

    if len(group) == 1:
        console.print(f"Copying {first['file'].name} -> {out_path.name}")
        shutil.copy2(first["file"], out_path)
    else:
        console.print(f"Concatenating {len(group)} files -> {out_path.name}")
        list_file = output_folder / "concat_list.txt"
        with open(list_file, "w") as f:
            for v in group:
                safe_path = str(v["file"].absolute()).replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        try:
            (
                ffmpeg.input(str(list_file), format="concat", safe=0)
                .output(str(out_path), c="copy", y=None)
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            console.print(f"[red]Concat failed: {e.stderr.decode() if e.stderr else str(e)}[/red]")
        list_file.unlink()


def main():
    check_system_dependencies()

    parser = argparse.ArgumentParser(description="Concatenate racing videos.")
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.input_folder.exists():
        console.print("[red]Input folder not found[/red]")
        sys.exit(1)

    out_folder = args.output or args.input_folder
    if args.debug:
        (out_folder / "debug").mkdir(parents=True, exist_ok=True)

    files = get_video_files(args.input_folder)
    if not files:
        console.print("[yellow]No videos found[/yellow]")
        return

    video_data = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task("Processing...", total=len(files))
        for f in files:
            progress.update(task, description=f"Analyzing {f.name}")
            video_data.append(analyze_video(f, out_folder / "debug" if args.debug else None))
            progress.advance(task)

    groups = group_videos(video_data)

    console.print(f"\nFound {len(groups)} groups:")
    for i, g in enumerate(groups):
        start = g[0]["start_time"]
        s_str = start.strftime("%Y-%m-%d %H:%M:%S") if start else "?"
        console.print(f"  Group {i + 1}: {len(g)} files, Start: {s_str}")

    if not args.dry_run:
        out_folder.mkdir(parents=True, exist_ok=True)
        for g in groups:
            export_group(g, out_folder)


if __name__ == "__main__":
    main()
