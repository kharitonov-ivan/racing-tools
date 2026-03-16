#!/usr/bin/env python3
"""
Batch Transcode Script
======================

Transcodes all videos in a folder to AV1/HEVC.

Usage:
    # GPU AV1 (fast, good quality)
    python racing_tools/transcode_folder.py -i /path/to/videos

    # CPU AV1 (slower, better compression)
    python racing_tools/transcode_folder.py -i /path/to/videos --codec libsvtav1

    # CPU HEVC (good balance)
    python racing_tools/transcode_folder.py -i /path/to/videos --codec libx265
    # No audio (video only)
    python racing_tools/transcode_folder.py -i /path/to/videos --no-audio
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".ts", ".m4v"}


def get_video_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    files = []
    search_fn = input_dir.rglob if recursive else input_dir.glob
    for ext in VIDEO_EXTENSIONS:
        files.extend(search_fn(f"*{ext}"))
        files.extend(search_fn(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def transcode_file(
    input_path: Path,
    output_path: Path,
    codec: str = "av1_nvenc",
    cq: int = 20,
    preset: str = "p7",
    overwrite: bool = False,
    no_audio: bool = False,
) -> bool:
    """
    Transcode a single file using ffmpeg.

    Quality recommendations:
    - av1_nvenc: cq=25 (good), cq=20 (excellent), cq=15 (near-lossless)
    - libsvtav1: cq=30 (good), cq=25 (excellent), cq=20 (near-lossless)
    - libx265: cq=26 (good), cq=22 (excellent), cq=18 (near-lossless)
    """
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path.name}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_args = ["-c:v", codec]

    if codec == "av1_nvenc":
        video_args.extend(
            [
                "-preset",
                preset,
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(cq),
                "-multipass",
                "qres",
                "-maxrate",
                "30M",
                "-bufsize",
                "60M",
                "-rc-lookahead",
                "60",
                "-spatial_aq",
                "1",
                "-temporal_aq",
                "1",
                "-aq-strength",
                "12",
                "-g",
                "300",
                "-bf",
                "5",
                "-pix_fmt",
                "yuv420p10le",
            ]
        )
    elif codec == "libsvtav1":
        preset_map = {"p7": "2", "p6": "3", "p5": "4", "p4": "5", "p3": "6", "p2": "7", "p1": "8"}
        sv_preset = preset_map.get(preset, "4")
        video_args.extend(
            [
                "-preset",
                sv_preset,
                "-crf",
                str(cq),
                "-svtav1-params",
                "fast-decode=1:tune=0",
                "-g",
                "300",
                "-pix_fmt",
                "yuv420p10le",
            ]
        )
    elif codec == "libx265":
        preset_map = {"p7": "veryslow", "p6": "slower", "p5": "slow", "p4": "medium", "p3": "fast", "p2": "faster", "p1": "veryfast"}
        x265_preset = preset_map.get(preset, "slow")
        video_args.extend(
            [
                "-preset",
                x265_preset,
                "-crf",
                str(cq),
                "-x265-params",
                "aq-mode=3:psy-rd=1.0",
                "-g",
                "300",
                "-pix_fmt",
                "yuv420p10le",
            ]
        )
    else:
        print(f"[Warning] Unknown codec '{codec}', using default settings")
        video_args.extend(["-q:v", str(cq)])

    hwaccel_args = ["-hwaccel", "cuda"] if "nvenc" in codec else []

    cmd = [
        "ffmpeg",
        *hwaccel_args,
        "-y" if overwrite else "-n",
        "-v",
        "error",
        "-stats",
        "-i",
        str(input_path),
        *video_args,
    ]

    if not no_audio:
        cmd.extend(["-c:a", "libopus", "-b:a", "192k"])
    else:
        cmd.append("-an")

    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error transcoding {input_path.name}: {e}")
        return False
    except KeyboardInterrupt:
        print("\nTranscoding interrupted by user.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Batch transcode videos to AV1/HEVC.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input folder containing videos.")
    parser.add_argument("--output", "-o", type=Path, help="Output folder. Defaults to 'transcoded' inside input folder.")
    parser.add_argument(
        "--codec",
        type=str,
        default="av1_nvenc",
        help="Video codec: av1_nvenc (GPU), libsvtav1 (CPU AV1), libx265 (CPU HEVC). Default: av1_nvenc",
    )
    parser.add_argument(
        "--cq",
        type=int,
        default=32,
        help="Quality. av1_nvenc: 15-35, libsvtav1: 20-35, libx265: 18-26. Lower = better. Default: 32",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="p7",
        help="NVENC: p1-p7 (p7=best). SVT-AV1: mapped to 2-8. Default: p7",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--no-audio", action="store_true", help="Discard audio stream.")
    parser.add_argument("--recursive", action="store_true", help="Process files in subdirectories recursively.")

    args = parser.parse_args()

    input_dir = args.input.expanduser().resolve()
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if args.output:
        output_dir = args.output.expanduser().resolve()
    else:
        output_dir = input_dir / "transcoded"

    print(f"Scanning {input_dir} for videos...")
    videos = get_video_files(input_dir, recursive=args.recursive)

    if not videos:
        print("No video files found.")
        return

    print(f"Found {len(videos)} video(s).")
    print(f"Output directory: {output_dir}")
    print(f"Settings: Codec={args.codec}, Preset={args.preset}, CQ={args.cq}")
    print("-" * 40)

    success_count = 0
    iterator = tqdm(videos, unit="video", desc="Transcoding")

    for video_path in iterator:
        rel_path = video_path.relative_to(input_dir)
        new_filename = f"{rel_path.stem}_transcoded{rel_path.suffix}"
        out_file = output_dir / rel_path.parent / new_filename

        if iterator is videos:
            print(f"Processing: {video_path.name} -> {out_file.name}")

        if transcode_file(
            video_path,
            out_file,
            codec=args.codec,
            cq=args.cq,
            preset=args.preset,
            overwrite=args.overwrite,
            no_audio=args.no_audio,
        ):
            success_count += 1

    print("-" * 40)
    print(f"Done. Successfully processed {success_count}/{len(videos)} files.")


if __name__ == "__main__":
    main()
