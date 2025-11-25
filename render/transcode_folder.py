#!/usr/bin/env python3
"""
Batch Transcode Script
======================

Transcodes all videos in a folder to AV1 using NVIDIA's hardware encoder (av1_nvenc).
Requires an NVIDIA RTX 40-series GPU or newer.

Usage:
    python3 render/transcode_folder.py --input /path/to/videos --output /path/to/output
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Try to import tqdm for progress bars, fallback if missing
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".ts", ".m4v"}

def get_video_files(input_dir: Path) -> List[Path]:
    """Recursively find all video files in the input directory."""
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))
        files.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(list(set(files)))

def transcode_file(
    input_path: Path,
    output_path: Path,
    codec: str = "av1_nvenc",
    cq: int = 20,
    preset: str = "p7",
    overwrite: bool = False
) -> bool:
    """
    Transcode a single file using ffmpeg.
    Returns True if successful, False otherwise.
    """
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path.name}")
        return True

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Adjust flags based on codec
    video_args = ["-c:v", codec]
    
    if "nvenc" in codec:
        video_args.extend([
            "-preset", preset,
            "-cq", str(cq),
            "-b:v", "0",
        ])
    elif codec == "libx264" or codec == "libx265":
        # Map p1-p7 to something reasonable or just ignore if user passed p7
        # For testing, we'll just use a standard preset if it looks like an nvenc preset
        use_preset = preset
        if preset.startswith("p") and preset[1:].isdigit():
             use_preset = "medium"
        
        video_args.extend([
            "-preset", use_preset,
            "-crf", str(cq), # Map CQ to CRF roughly
        ])
    else:
        # Generic fallback
        video_args.extend(["-q:v", str(cq)])

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-v", "error",
        "-stats",
        "-i", str(input_path),
        *video_args,
        "-c:a", "copy",
        str(output_path)
    ]

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
    parser = argparse.ArgumentParser(description="Batch transcode videos to AV1 (NVIDIA).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input folder containing videos.")
    parser.add_argument("--output", "-o", type=Path, help="Output folder. Defaults to 'transcoded' inside input folder.")
    parser.add_argument("--codec", type=str, default="av1_nvenc", help="Video codec (default: av1_nvenc).")
    parser.add_argument("--cq", type=int, default=30, help="Constant Quality value (default: 30). Lower = better quality, larger file.")
    parser.add_argument("--preset", type=str, default="p7", help="NVENC Preset (p1-p7). Default: p7 (best quality).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    
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
    videos = get_video_files(input_dir)
    
    if not videos:
        print("No video files found.")
        return

    print(f"Found {len(videos)} video(s).")
    print(f"Output directory: {output_dir}")
    print(f"Settings: Codec={args.codec}, Preset={args.preset}, CQ={args.cq}")
    print("-" * 40)

    success_count = 0
    
    # Use tqdm if available, otherwise simple loop
    iterator = tqdm(videos, unit="video", desc="Transcoding")
    
    for video_path in iterator:
        # Calculate relative path to maintain structure if needed, 
        # or just flat output. Let's do flat output for simplicity unless collision.
        # Actually, preserving structure is safer for "folder" inputs.
        
        rel_path = video_path.relative_to(input_dir)
        out_file = output_dir / rel_path.with_suffix(".mp4") # Force mp4 container
        
        # If not using tqdm, print current file
        if iterator is videos: 
            print(f"Processing: {video_path.name} -> {out_file.name}")

        if transcode_file(video_path, out_file, codec=args.codec, cq=args.cq, preset=args.preset, overwrite=args.overwrite):
            success_count += 1

    print("-" * 40)
    print(f"Done. Successfully processed {success_count}/{len(videos)} files.")

if __name__ == "__main__":
    main()
