#!/usr/bin/env python3
"""Transcode Script.

Transcodes video(s) to HEVC/AV1/H.264.

Usage:
    # Single file (GPU HEVC)
    python racing_tools/transcode.py video.mp4

    # Folder (all videos)
    python racing_tools/transcode.py /path/to/videos

    # GPU AV1 (better compression, but OpenCV seek may not work)
    python racing_tools/transcode.py video.mp4 --codec av1_nvenc

    # CPU HEVC (no GPU required)
    python racing_tools/transcode.py video.mp4 --codec libx265

    # CPU H.264 (faster, wider compatibility)
    python racing_tools/transcode.py video.mp4 --codec libx264

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

from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".ts", ".m4v"}

# Constants for encoder settings
NVENC_BUFSIZE = "30M"
NVENC_LOOKAHEAD = 60
NVENC_AQ_STRENGTH = 15
DEFAULT_BITRATE = "15M"
DEFAULT_MAXRATE = "20M"
RESOLUTION_BITRATE_PRESETS = {
    720: ("10M", "16M"),
    1080: (DEFAULT_BITRATE, DEFAULT_MAXRATE),
    1440: ("25M", "35M"),
    2160: ("45M", "60M"),
}

# Seek-friendly defaults: short GOP + no B-frames keeps OpenCV frame seeking
# responsive (keyframe ~every 0.5-1s, no reorder ambiguity). Raise GOP / add
# B-frames for smaller files when accurate random seek is not needed.
DEFAULT_GOP_SIZE = 30
DEFAULT_B_FRAMES = 2


def get_video_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Get all video files from directory, optionally recursively."""
    files = []
    search_fn = input_dir.rglob if recursive else input_dir.glob
    for ext in VIDEO_EXTENSIONS:
        files.extend(search_fn(f"*{ext}"))
        files.extend(search_fn(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def build_av1_nvenc_args(cq: int, preset: str, bitrate: str, maxrate: str, gop: int, bframes: int) -> list[str]:
    """Build ffmpeg arguments for AV1 NVENC encoder."""
    return [
        "-preset",
        preset,
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(cq),
        "-b:v",
        bitrate,
        "-maxrate:v",
        maxrate,
        "-bufsize:v",
        NVENC_BUFSIZE,
        "-multipass",
        "fullres",
        "-rc-lookahead",
        str(NVENC_LOOKAHEAD),
        "-spatial_aq",
        "1",
        "-temporal_aq",
        "1",
        "-aq-strength",
        str(NVENC_AQ_STRENGTH),
        "-bf",
        str(bframes),
        "-g",
        str(gop),
        "-pix_fmt",
        "yuv420p10le",
    ]


def build_hevc_nvenc_args(cq: int, preset: str, bitrate: str, maxrate: str, gop: int, bframes: int) -> list[str]:
    """Build ffmpeg arguments for HEVC NVENC encoder."""
    return [
        "-preset",
        preset,
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(cq),
        "-b:v",
        bitrate,
        "-maxrate:v",
        maxrate,
        "-bufsize:v",
        NVENC_BUFSIZE,
        "-multipass",
        "qres",
        "-rc-lookahead",
        str(NVENC_LOOKAHEAD),
        "-spatial_aq",
        "1",
        "-temporal_aq",
        "1",
        "-aq-strength",
        str(NVENC_AQ_STRENGTH),
        "-bf",
        str(bframes),
        "-g",
        str(gop),
        "-pix_fmt",
        "yuv420p10le",
    ]


def build_svtav1_args(cq: int, preset: str, gop: int) -> list[str]:
    """Build ffmpeg arguments for SVT-AV1 encoder."""
    preset_map = {"p7": "2", "p6": "3", "p5": "4", "p4": "5", "p3": "6", "p2": "7", "p1": "8"}
    sv_preset = preset_map.get(preset, "4")
    return [
        "-preset",
        sv_preset,
        "-crf",
        str(cq),
        "-svtav1-params",
        "fast-decode=1:tune=0",
        "-g",
        str(gop),
        "-pix_fmt",
        "yuv420p10le",
    ]


def build_x265_args(cq: int, preset: str, gop: int) -> list[str]:
    """Build ffmpeg arguments for x265 encoder."""
    preset_map = {"p7": "veryslow", "p6": "slower", "p5": "slow", "p4": "medium", "p3": "fast", "p2": "faster", "p1": "veryfast"}
    x265_preset = preset_map.get(preset, "slow")
    return [
        "-preset",
        x265_preset,
        "-crf",
        str(cq),
        "-x265-params",
        "aq-mode=3:psy-rd=1.0",
        "-g",
        str(gop),
        "-pix_fmt",
        "yuv420p10le",
    ]


def build_x264_args(cq: int, preset: str, gop: int) -> list[str]:
    """Build ffmpeg arguments for x264 encoder."""
    preset_map = {"p7": "veryslow", "p6": "slower", "p5": "slow", "p4": "medium", "p3": "fast", "p2": "faster", "p1": "veryfast"}
    x264_preset = preset_map.get(preset, "slow")
    return [
        "-preset",
        x264_preset,
        "-crf",
        str(cq),
        "-tune",
        "film",
        "-g",
        str(gop),
        "-pix_fmt",
        "yuv420p",
    ]


def get_video_codec_args(
    codec: str, cq: int, preset: str, bitrate: str, maxrate: str, gop: int, bframes: int
) -> list[str]:
    """Get encoder-specific arguments for video codec.

    Quality recommendations:
    - hevc_nvenc: cq=22 (good), cq=18 (excellent), cq=15 (near-lossless)
    - av1_nvenc: cq=20 (good), cq=15 (excellent), cq=12 (near-lossless)
    - libsvtav1: cq=30 (good), cq=25 (excellent), cq=20 (near-lossless)
    - libx265: cq=26 (good), cq=22 (excellent), cq=18 (near-lossless)
    - libx264: cq=23 (good), cq=20 (excellent), cq=16 (near-lossless)
    """
    base_args = ["-c:v", codec]

    if codec == "av1_nvenc":
        codec_args = build_av1_nvenc_args(cq, preset, bitrate, maxrate, gop, bframes)
    elif codec == "hevc_nvenc":
        codec_args = build_hevc_nvenc_args(cq, preset, bitrate, maxrate, gop, bframes)
    elif codec == "libsvtav1":
        codec_args = build_svtav1_args(cq, preset, gop)
    elif codec == "libx265":
        codec_args = build_x265_args(cq, preset, gop)
    elif codec == "libx264":
        codec_args = build_x264_args(cq, preset, gop)
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
    gop: int,
    bframes: int,
    resolution: int | None = None,
) -> list[str]:
    """Build complete ffmpeg command line arguments."""
    hwaccel_args = ["-hwaccel", "cuda"] if "nvenc" in codec else []
    video_args = get_video_codec_args(codec, cq, preset, bitrate, maxrate, gop, bframes)

    cmd = [
        "ffmpeg",
        *hwaccel_args,
        "-y" if overwrite else "-n",
        "-v",
        "error",
        "-stats",
        "-i",
        str(input_path),
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
    gop: int = DEFAULT_GOP_SIZE,
    bframes: int = DEFAULT_B_FRAMES,
    resolution: int | None = None,
) -> bool:
    """Transcode a single file using ffmpeg."""
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path.name}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_ffmpeg_command(
        input_path, output_path, codec, cq, preset, overwrite, no_audio, bitrate, maxrate, gop, bframes, resolution
    )

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error transcoding {input_path.name}: {e}")
        # Suggest CPU codec if CUDA/NVENC failed
        if "nvenc" in codec.lower() and ("cuda" in str(e).lower() or "nvenc" in str(e).lower() or "device" in str(e).lower()):
            print("\n[TIP] CUDA/NVENC encoder failed. Try CPU encoding instead:")
            print(f"      {sys.argv[0]} {input_path} --codec libx265  # HEVC")
            print(f"      {sys.argv[0]} {input_path} --codec libx264  # H.264")
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
    print(f"Settings: Codec={args.codec}, Preset={args.preset}, CQ={args.cq}, Bitrate={args.bitrate}, Maxrate={args.maxrate}")
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
        gop=args.gop,
        bframes=args.bframes,
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
    print(f"Settings: Codec={args.codec}, Preset={args.preset}, CQ={args.cq}, Bitrate={args.bitrate}, Maxrate={args.maxrate}")
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
            gop=args.gop,
            bframes=args.bframes,
            resolution=args.resolution,
        ):
            success_count += 1

    print("-" * 40)
    print(f"Done. Successfully processed {success_count}/{len(videos)} files.")
    return success_count


def resolve_bitrate_settings(args: argparse.Namespace) -> None:
    """Apply bitrate defaults, lowering them automatically for lower output resolutions."""
    if args.bitrate is not None and args.maxrate is not None:
        return

    preset_bitrate, preset_maxrate = RESOLUTION_BITRATE_PRESETS.get(
        args.resolution,
        (DEFAULT_BITRATE, DEFAULT_MAXRATE),
    )
    if args.bitrate is None:
        args.bitrate = preset_bitrate
    if args.maxrate is None:
        args.maxrate = preset_maxrate


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transcode video(s) to HEVC/AV1.")
    parser.add_argument("input", type=Path, help="Input video file or folder.")
    parser.add_argument("--output", "-o", type=Path, help="Output file (for single) or folder. Default: same dir with suffix.")
    parser.add_argument(
        "--codec",
        type=str,
        default="hevc_nvenc",
        help="Video codec: av1_nvenc, hevc_nvenc (GPU), libsvtav1 (CPU AV1), libx265 (CPU HEVC), libx264 (CPU H.264). Default: hevc_nvenc",
    )
    parser.add_argument(
        "--cq",
        type=int,
        default=18,
        help="Quality. av1_nvenc: 10-25, hevc_nvenc: 15-28, libsvtav1: 20-35, libx265: 18-26, libx264: 16-28. Lower = better. Default: 18",
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
        default=None,
        help=f"Target bitrate for VBR. Default: auto by resolution ({DEFAULT_BITRATE} for 1080p, 12M for 720p)",
    )
    parser.add_argument(
        "--maxrate",
        type=str,
        default=None,
        help=f"Max bitrate for VBR. Default: auto by resolution ({DEFAULT_MAXRATE} for 1080p, 18M for 720p)",
    )
    parser.add_argument(
        "--gop",
        type=int,
        default=DEFAULT_GOP_SIZE,
        help=f"Keyframe interval (frames). Smaller = faster/more accurate OpenCV seek, larger files. Default: {DEFAULT_GOP_SIZE}",
    )
    parser.add_argument(
        "--bframes",
        type=int,
        default=DEFAULT_B_FRAMES,
        help=f"B-frames (NVENC only). 0 = best for OpenCV frame seeking; raise for smaller files. Default: {DEFAULT_B_FRAMES}",
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

    args = parser.parse_args()
    resolve_bitrate_settings(args)
    return args


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
