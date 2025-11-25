#!/usr/bin/env python3
"""
Video Split Script

This script splits a video file into two parts at a specified time point.

Usage:
    python video_split.py <video_path> <split_time> [--output-dir OUTPUT_DIR]

Arguments:
    video_path: Path to the input video file
    split_time: Time to split at (format: HH:MM:SS or MM:SS or seconds)
    --output-dir: Optional directory for output files (default: same as input)

Examples:
    python video_split.py video.mp4 01:30:00
    python video_split.py video.mp4 5:30
    python video_split.py video.mp4 125
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_time_to_seconds(time_str: str) -> float:
    """
    Parse time string to seconds.
    
    Supports formats:
    - HH:MM:SS or HH:MM:SS.mmm
    - MM:SS or MM:SS.mmm
    - seconds (integer or float)
    
    Args:
        time_str: Time string to parse
        
    Returns:
        Time in seconds
    """
    try:
        # Try parsing as float (seconds)
        return float(time_str)
    except ValueError:
        pass
    
    # Parse as time format
    parts = time_str.split(':')
    if len(parts) == 3:
        # HH:MM:SS
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        # MM:SS
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def get_video_duration(video_path: str) -> float:
    """
    Get video duration using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_video(video_path: str, split_time: float, output_dir: str = None):
    """
    Split video into two parts at the specified time.
    
    Args:
        video_path: Path to input video
        split_time: Time to split at (in seconds)
        output_dir: Optional output directory (default: same as input)
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video duration
    print(f"Analyzing video: {video_path}")
    duration = get_video_duration(str(video_path))
    print(f"Video duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Split time: {split_time:.2f} seconds ({split_time/60:.2f} minutes)")
    
    # Validate split time
    if split_time <= 0:
        raise ValueError("Split time must be positive")
    if split_time >= duration:
        raise ValueError(f"Split time ({split_time}s) must be less than video duration ({duration}s)")
    
    # Determine output directory
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames
    stem = video_path.stem
    ext = video_path.suffix
    output_part1 = output_dir / f"{stem}_part1{ext}"
    output_part2 = output_dir / f"{stem}_part2{ext}"
    
    print(f"\nOutput files:")
    print(f"  Part 1: {output_part1}")
    print(f"  Part 2: {output_part2}")
    
    # Split video - Part 1 (from start to split_time)
    print(f"\nCreating part 1 (0 to {split_time}s)...")
    cmd_part1 = [
        'ffmpeg',
        '-i', str(video_path),
        '-t', str(split_time),
        '-c', 'copy',  # Copy codec (fast, no re-encoding)
        '-avoid_negative_ts', 'make_zero',
        '-y',  # Overwrite output file
        str(output_part1)
    ]
    
    subprocess.run(cmd_part1, check=True)
    print(f"✓ Part 1 created successfully")
    
    # Split video - Part 2 (from split_time to end)
    print(f"\nCreating part 2 ({split_time}s to end)...")
    cmd_part2 = [
        'ffmpeg',
        '-i', str(video_path),
        '-ss', str(split_time),
        '-c', 'copy',  # Copy codec (fast, no re-encoding)
        '-avoid_negative_ts', 'make_zero',
        '-y',  # Overwrite output file
        str(output_part2)
    ]
    
    subprocess.run(cmd_part2, check=True)
    print(f"✓ Part 2 created successfully")
    
    # Display file sizes
    size1 = output_part1.stat().st_size
    size2 = output_part2.stat().st_size
    size_orig = video_path.stat().st_size
    
    print(f"\nFile sizes:")
    print(f"  Original: {size_orig / 1024 / 1024:.2f} MB")
    print(f"  Part 1:   {size1 / 1024 / 1024:.2f} MB ({size1/size_orig*100:.1f}%)")
    print(f"  Part 2:   {size2 / 1024 / 1024:.2f} MB ({size2/size_orig*100:.1f}%)")
    print(f"\n✓ Video split completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Split a video file into two parts at a specified time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4 01:30:00
  %(prog)s video.mp4 5:30
  %(prog)s video.mp4 125
  %(prog)s video.mp4 01:30:00 --output-dir /path/to/output
        """
    )
    
    parser.add_argument(
        'video_path',
        help='Path to the input video file'
    )
    
    parser.add_argument(
        'split_time',
        help='Time to split at (format: HH:MM:SS, MM:SS, or seconds)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for split files (default: same as input)',
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        # Parse split time
        split_seconds = parse_time_to_seconds(args.split_time)
        
        # Split video
        split_video(args.video_path, split_seconds, args.output_dir)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
