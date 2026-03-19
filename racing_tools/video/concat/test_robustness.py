from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

# Add current directory to sys.path
sys.path.append(str(Path(__file__).parent))

from concat import analyze_video, group_videos


def split_video(file_path: Path, output_dir: Path) -> list[Path]:
    """
    Splits the first 3 minutes of a video into random chunks (30s +/- 15s).
    Returns a list of paths to the chunks.
    """
    import random
    import subprocess

    if not file_path.exists():
        return []

    # Get duration
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        total_duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return []

    # Limit to 3 minutes max for testing
    test_duration = min(total_duration, 180.0)
    if test_duration < 10:
        return []

    chunks = []
    current_time = 0.0
    part_idx = 1

    while current_time < test_duration:
        # Random duration: 30s +/- 15s (min 15s, max 45s)
        chunk_dur = 30.0 + random.uniform(-15.0, 15.0)

        # Clamp to remaining time
        if current_time + chunk_dur > test_duration:
            chunk_dur = test_duration - current_time

        # If remaining is too small (<5s), just add to previous or skip
        if chunk_dur < 5.0:
            break

        output_name = f"{file_path.stem}_part_{part_idx:03d}.mp4"
        output_path = output_dir / output_name

        # Extract chunk
        # Use mpeg4 and -an to avoid keyframe/audio issues in testing
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            str(current_time),
            "-t",
            str(chunk_dur),
            "-i",
            str(file_path),
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-an",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)

        chunks.append(output_path)
        current_time += chunk_dur
        part_idx += 1

    return chunks


@pytest.fixture
def temp_dir():
    """Fixture to create and clean up a temporary directory."""
    path = Path("tests/temp_robustness").resolve()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "debug").mkdir(exist_ok=True)
    yield path
    # Cleanup after test
    if path.exists():
        shutil.rmtree(path)


def test_all_videos_robustness(temp_dir):
    """Test video splitting and grouping algorithm on all videos.

    Iterates through all videos, splits them into random chunks,
    and verifies that the algorithm groups them back correctly.
    """
    # Locate VIDEO directory relative to this test file
    video_dir = Path(__file__).parent.parent / "VIDEO"

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"}
    videos = sorted([f for f in video_dir.iterdir() if f.suffix in video_extensions])

    if not videos:
        pytest.skip("No videos found in render/VIDEO")

    print(f"\nFound {len(videos)} videos to test.")

    results = []

    for video in videos:
        print(f"\nTesting {video.name}...")

        # Clean temp dir for each video to save space
        # We keep the debug folder structure but empty it
        for f in temp_dir.glob("*.mp4"):
            f.unlink()
        for f in (temp_dir / "debug").glob("*"):
            f.unlink()

        parts = split_video(video, temp_dir)
        if not parts:
            print(f"Skipping {video.name} (too short or split failed)")
            continue

        print(f"  Split into {len(parts)} parts.")

        video_data = []
        for p in parts:
            data = analyze_video(p, debug_folder=temp_dir / "debug")
            video_data.append(data)

        for p, data in zip(parts, video_data):
            s = data["start_time"]
            e = data["end_time"]
            print(f"    {p.name}: Start={s}, End={e}")

        groups = group_videos(video_data)

        success = False
        msg = ""

        if len(groups) != 1:
            msg = f"Failed: Expected 1 group, got {len(groups)}"
        elif len(groups[0]) != len(parts):
            msg = f"Failed: Expected {len(parts)} videos, got {len(groups[0])}"
        else:
            # Check continuity
            is_continuous = True
            for i in range(len(groups[0]) - 1):
                curr = groups[0][i]
                next_v = groups[0][i + 1]
                if curr["start_time"] and next_v["start_time"]:
                    diff = (next_v["start_time"] - curr["start_time"]).total_seconds()
                    expected = curr["duration"]
                    # Tolerance increased to 4s as per recent fixes
                    if abs(diff - expected) > 4:
                        is_continuous = False
                        msg = f"Failed: Discontinuity at index {i} (diff={diff:.1f}s, expected={expected:.1f}s)"
                        break
                else:
                    is_continuous = False
                    msg = f"Failed: Missing timestamps at index {i}"
                    break

            if is_continuous:
                success = True
                msg = "Passed"

        print(f"  Result: {msg}")
        results.append((video.name, success, msg))

    # Summary Report
    print("\n" + "=" * 40)
    print("Summary:")
    print("=" * 40)
    passed = 0
    for name, success, msg in results:
        status = "PASS" if success else "FAIL"
        if success:
            passed += 1
        print(f"{name:<20} | {status} | {msg}")

    print("-" * 40)
    print(f"Total: {len(results)}, Passed: {passed}, Failed: {len(results) - passed}")

    assert passed == len(results), "Not all videos passed robustness test"
