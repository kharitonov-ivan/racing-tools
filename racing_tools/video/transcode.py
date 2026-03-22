#!/usr/bin/env python3
"""Transcode Script.

Transcodes video(s) to HEVC/AV1.

Usage:
    # Single file (GPU HEVC)
    python racing_tools/transcode.py video.mp4

    # Folder (all videos)
    python racing_tools/transcode.py /path/to/videos

    # GPU AV1 (better compression, but OpenCV seek may not work)
    python racing_tools/transcode.py video.mp4 --codec av1_nvenc

    # CPU AV1 (slower, better compression)
    python racing_tools/transcode.py video.mp4 --codec libsvtav1

    # No audio (video only)
    python racing_tools/transcode.py video.mp4 --no-audio

    # Scale to 720p
    python racing_tools/transcode.py video.mp4 --resolution 720
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".ts", ".m4v"}

# Constants for encoder settings
NVENC_BUFSIZE = "30M"
NVENC_LOOKAHEAD = 60
NVENC_AQ_STRENGTH = 15
NVENC_B_FRAMES = 5
NVENC_GOP_SIZE = 60
SVTAV1_GOP_SIZE = 60
X265_GOP_SIZE = 60


def get_video_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Get all video files from directory, optionally recursively."""
    files = []
    search_fn = input_dir.rglob if recursive else input_dir.glob
    for ext in VIDEO_EXTENSIONS:
        files.extend(search_fn(f"*{ext}"))
        files.extend(search_fn(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def build_av1_nvenc_args(cq: int, preset: str, bitrate: str, maxrate: str) -> list[str]:
    """Build ffmpeg arguments for AV1 NVENC encoder."""
    return [
        "-preset", preset,
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", str(cq),
        "-b:v", bitrate,
        "-maxrate:v", maxrate,
        "-bufsize:v", NVENC_BUFSIZE,
        "-multipass", "fullres",
        "-rc-lookahead", str(NVENC_LOOKAHEAD),
        "-spatial_aq", "1",
        "-temporal_aq", "1",
        "-aq-strength", str(NVENC_AQ_STRENGTH),
        "-bf", str(NVENC_B_FRAMES),
        "-g", str(NVENC_GOP_SIZE),
        "-pix_fmt", "yuv420p10le",
    ]


def build_hevc_nvenc_args(cq: int, preset: str, bitrate: str, maxrate: str) -> list[str]:
    """Build ffmpeg arguments for HEVC NVENC encoder."""
    return [
        "-preset", preset,
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", str(cq),
        "-b:v", bitrate,
        "-maxrate:v", maxrate,
        "-bufsize:v", NVENC_BUFSIZE,
        "-multipass", "qres",
        "-rc-lookahead", str(NVENC_LOOKAHEAD),
        "-spatial_aq", "1",
        "-temporal_aq", "1",
        "-aq-strength", str(NVENC_AQ_STRENGTH),
        "-bf", str(NVENC_B_FRAMES),
        "-g", str(NVENC_GOP_SIZE),
        "-pix_fmt", "yuv420p10le",
    ]


def build_svtav1_args(cq: int, preset: str) -> list[str]:
    """Build ffmpeg arguments for SVT-AV1 encoder."""
    preset_map = {"p7": "2", "p6": "3", "p5": "4", "p4": "5", "p3": "6", "p2": "7", "p1": "8"}
    sv_preset = preset_map.get(preset, "4")
    return [
        "-preset", sv_preset,
        "-crf", str(cq),
        "-svtav1-params", "fast-decode=1:tune=0",
        "-g", str(SVTAV1_GOP_SIZE),
        "-pix_fmt", "yuv420p10le",
    ]


def build_x265_args(cq: int, preset: str) -> list[str]:
    """Build ffmpeg arguments for x265 encoder."""
    preset_map = {
        "p7": "veryslow", "p6": "slower", "p5": "slow",
        "p4": "medium", "p3": "fast", "p2": "faster", "p1": "veryfast"
    }
    x265_preset = preset_map.get(preset, "slow")
    return [
        "-preset", x265_preset,
        "-crf", str(cq),
        "-x265-params", "aq-mode=3:psy-rd=1.0",
        "-g", str(X265_GOP_SIZE),
        "-pix_fmt", "yuv420p10le",
    ]


def get_video_codec_args(codec: str, cq: int, preset: str, bitrate: str, maxrate: str) -> list[str]:
    """Get encoder-specific arguments for video codec.

    Quality recommendations:
    - hevc_nvenc: cq=22 (good), cq=18 (excellent), cq=15 (near-lossless)
    - av1_nvenc: cq=20 (good), cq=15 (excellent), cq=12 (near-lossless)
    - libsvtav1: cq=30 (good), cq=25 (excellent), cq=20 (near-lossless)
    - libx265: cq=26 (good), cq=22 (excellent), cq=18 (near-lossless)
    """
    base_args = ["-c:v", codec]

    if codec == "av1_nvenc":
        codec_args = build_av1_nvenc_args(cq, preset, bitrate, maxrate)
    elif codec == "hevc_nvenc":
        codec_args = build_hevc_nvenc_args(cq, preset, bitrate, maxrate)
    elif codec == "libsvtav1":
        codec_args = build_svtav1_args(cq, preset)
    elif codec == "libx265":
        codec_args = build_x265_args(cq, preset)
    else:
        print(f"[Warning] Unknown codec '{codec}', using default settings")
        codec_args = ["-q:v", str(cq)]

    return base_args + codec_args


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    codec: str,
    cq: int,
    preset: str,
    overwrite: bool,
    no_audio: bool,
    bitrate: str,
    maxrate: str,
    resolution: int | None = None,
) -> list[str]:
    """Build complete ffmpeg command line arguments."""
    hwaccel_args = ["-hwaccel", "cuda"] if "nvenc" in codec else []
    video_args = get_video_codec_args(codec, cq, preset, bitrate, maxrate)

    cmd = [
        "ffmpeg",
        *hwaccel_args,
        "-y" if overwrite else "-n",
        "-v", "error",
        "-stats",
        "-i", str(input_path),
    ]

    # Add scale filter if resolution specified
    if resolution:
        cmd.extend(["-vf", f"scale=-1:{resolution}"])

    cmd.extend(video_args)

    if not no_audio:
        cmd.extend(["-c:a", "libopus", "-b:a", "192k"])
    else:
        cmd.append("-an")

    cmd.append(str(output_path))
    return cmd


def transcode_file(
    input_path: Path,
    output_path: Path,
    codec: str = "hevc_nvenc",
    cq: int = 18,
    preset: str = "p7",
    overwrite: bool = False,
    no_audio: bool = False,
    bitrate: str = "15M",
    maxrate: str = "20M",
    resolution: int | None = None,
) -> bool:
    """Transcode a single file using ffmpeg."""
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path.name}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_ffmpeg_command(
        input_path, output_path, codec, cq, preset,
        overwrite, no_audio, bitrate, maxrate, resolution
    )

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error transcoding {input_path.name}: {e}")
        return False
    except KeyboardInterrupt:
        print("\nTranscoding interrupted by user.")
        sys.exit(1)


def transcode_single_file(args) -> bool:
    """Transcode a single video file."""
    input_path = args.input.expanduser().resolve()

    if args.output:
        output_path = args.output.expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.stem}_transcoded{input_path.suffix}"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Settings: Codec={args.codec}, Preset={args.preset}, CQ={args.cq}")
    print("-" * 40)

    success = transcode_file(
        input_path,
        output_path,
        codec=args.codec,
        cq=args.cq,
        preset=args.preset,
        overwrite=args.overwrite,
        no_audio=args.no_audio,
        bitrate=args.bitrate,
        maxrate=args.maxrate,
        resolution=args.resolution,
    )
    print("-" * 40)
    print("Done." if success else "Failed.")
    return success


def transcode_multiple_files(args) -> int:
    """Transcode multiple video files in a directory."""
    input_path = args.input.expanduser().resolve()

    if args.output:
        output_dir = args.output.expanduser().resolve()
    else:
        output_dir = input_path / "transcoded"

    print(f"Scanning {input_path} for videos...")
    videos = get_video_files(input_path, recursive=args.recursive)

    if not videos:
        print("No video files found.")
        return 0

    print(f"Found {len(videos)} video(s).")
    print(f"Output directory: {output_dir}")
    print(f"Settings: Codec={args.codec}, Preset={args.preset}, CQ={args.cq}")
    print("-" * 40)

    success_count = 0
    iterator = tqdm(videos, unit="video", desc="Transcoding")

    for video_path in iterator:
        rel_path = video_path.relative_to(input_path)
        new_filename = f"{rel_path.stem}_transcoded{rel_path.suffix}"
        out_file = output_dir / rel_path.parent / new_filename

        # Update progress bar description
        iterator.set_description(f"{video_path.name}")

        if transcode_file(
            video_path,
            out_file,
            codec=args.codec,
            cq=args.cq,
            preset=args.preset,
            overwrite=args.overwrite,
            no_audio=args.no_audio,
            bitrate=args.bitrate,
            maxrate=args.maxrate,
            resolution=args.resolution,
        ):
            success_count += 1

    print("-" * 40)
    print(f"Done. Successfully processed {success_count}/{len(videos)} files.")
    return success_count


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transcode video(s) to HEVC/AV1.")
    parser.add_argument("input", type=Path, help="Input video file or folder.")
    parser.add_argument("--output", "-o", type=Path, help="Output file (for single) or folder. Default: same dir with suffix.")
    parser.add_argument(
        "--codec",
        type=str,
        default="hevc_nvenc",
        help="Video codec: av1_nvenc, hevc_nvenc (GPU), libsvtav1 (CPU AV1), libx265 (CPU HEVC). Default: hevc_nvenc",
    )
    parser.add_argument(
        "--cq",
        type=int,
        default=18,
        help="Quality. av1_nvenc: 10-25, hevc_nvenc: 15-28, libsvtav1: 20-35, libx265: 18-26. Lower = better. Default: 18",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="p7",
        help="NVENC: p1-p7 (p7=best). SVT-AV1: mapped to 2-8. Default: p7",
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        default="15M",
        help="Target bitrate for VBR. Default: 15M",
    )
    parser.add_argument(
        "--maxrate",
        type=str,
        default="20M",
        help="Max bitrate for VBR. Default: 20M",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--no-audio", action="store_true", help="Discard audio stream.")
    parser.add_argument("--recursive", action="store_true", help="Process files in subdirectories recursively (folder mode only).")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Output height in pixels (e.g., 1080, 720). Width auto-calculated to maintain aspect ratio.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for transcode script."""
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print(f"Error: Input '{input_path}' does not exist.")
        sys.exit(1)

    if input_path.is_file():
        transcode_single_file(args)
    else:
        transcode_multiple_files(args)


if __name__ == "__main__":
    main()
