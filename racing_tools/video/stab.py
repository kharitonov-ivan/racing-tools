#!/usr/bin/env python3
"""
Video Stabilization Module
==========================

Stabilizes onboard video using ffmpeg's vidstab filters (vidstabdetect + vidstabtransform).
Performs a 2-pass stabilization.
Uses NVIDIA Hardware Acceleration (decoding & encoding) or CPU fallback.

Usage:
    python -m racing_tools.video.stab input.mp4 [output.mp4]
    # or
    python racing_tools/video/stab.py input.mp4 [output.mp4]
"""

import argparse
import sys
from pathlib import Path
import ffmpeg

from racing_tools.utils import check_cuda_availability


def generate_transforms(input_path: Path, shakiness: int = 10, accuracy: int = 15, stepsize: int = 32, overwrite: bool = False) -> Path | None:
    """
    Run pass 1 detection and return path to transforms file.
    Returns None on failure.
    """
    transforms_file = input_path.with_name(f".{input_path.stem}_transforms.trf")
    
    if transforms_file.exists():
        if overwrite:
            print(f"[stab] Overwriting existing transforms: {transforms_file.name}")
        else:
            print(f"[stab] Found existing transforms: {transforms_file.name}")
            try:
                choice = input("Re-calculate transforms? [y/N]: ").strip().lower()
                if choice != 'y':
                    return transforms_file
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                sys.exit(0)

    print(f"[stab] Pass 1: Analyzing video motion (vidstabdetect) Shakiness={shakiness}...")
    
    try:
        # Build ffmpeg command using ffmpeg-python
        stream = ffmpeg.input(str(input_path))
        stream = ffmpeg.filter(
            stream,
            'format',
            'yuv420p'
        )
        stream = ffmpeg.filter(
            stream,
            'vidstabdetect',
            stepsize=stepsize,
            shakiness=shakiness,
            accuracy=accuracy,
            result=str(transforms_file.absolute())
        )
        stream = ffmpeg.output(stream, '-', format='null')
        
        # Run with stats and error logging
        ffmpeg.run(
            stream,
            overwrite_output=True
        )
        
        return transforms_file
    except ffmpeg.Error as e:
        print(f"[stab] Error during Pass 1: {e.stderr.decode() if e.stderr else str(e)}")
        return None


def get_transform_filter(transforms_file: Path, smoothing: int = 10, zoom: int = 0, optzoom: int = 0, unsharp: bool = True, crop: str = "keep") -> dict:
    """Return the vidstabtransform filter parameters and unsharp if needed."""
    return {
        'transform': {
            'input': str(transforms_file.absolute()),
            'zoom': zoom,
            'smoothing': smoothing,
            'optzoom': optzoom,
            'crop': crop
        },
        'unsharp': unsharp
    }


def stabilize_video(
    input_path: Path,
    output_path: Path,
    smoothing: int = 10,
    zoom: int = 0,
    optzoom: int = 0,
    shakiness: int = 5,
    accuracy: int = 15,
    stepsize: int = 6,
    unsharp: bool = False,
    crop: str = "keep",
    interpol: str = "bilinear",
    cq: int = 19,
    preset: str = "p6",
    overwrite: bool = False,
    lossless: bool = False
):
    """
    Stabilize video using ffmpeg vidstab filters.
    Automatically detects CUDA availability.
    """
    if output_path.exists() and not overwrite:
        print(f"Error: Output file '{output_path}' already exists. Use --overwrite to replace.")
        return False

    has_cuda = check_cuda_availability()
    
    print(f"Stabilizing: {input_path}")
    print(f"Output to:   {output_path}")
    print(f"Params:      Shakiness={shakiness}, Smoothing={smoothing}, Unsharp={unsharp}")
    
    if lossless:
        print(f"Mode:        LOSSLESS (CPU libx264 -qp 0)")
    elif has_cuda:
        print(f"Mode:        GPU (CUDA + h264_nvenc)")
    else:
        print(f"Mode:        CPU (libx264)")

    print("-" * 40)

    # 1. Detection Pass
    transforms_file = generate_transforms(input_path, shakiness=shakiness, accuracy=accuracy, stepsize=stepsize, overwrite=False)
    if not transforms_file:
        return False

    # 2. Transform Pass
    print("[stab] Pass 2: Applying stabilization (vidstabtransform) & Encoding...")
    
    try:
        # Build ffmpeg command using ffmpeg-python
        stream = ffmpeg.input(str(input_path))
        
        # Apply format and stabilization filters
        stream = ffmpeg.filter(stream, 'format', 'yuv420p')
        stream = ffmpeg.filter(
            stream,
            'vidstabtransform',
            input=str(transforms_file.absolute()),
            zoom=zoom,
            smoothing=smoothing,
            optzoom=optzoom,
            crop=crop,
            interpol=interpol
        )
        
        # Apply unsharp if needed
        if unsharp:
            # Default unsharp parameters from vidstab example
            # luma_msize_x:luma_msize_y:luma_amount:chroma_msize_x:chroma_msize_y:chroma_amount
            stream = ffmpeg.filter(stream, 'unsharp', lx=5, ly=5, la=0.8, cx=3, cy=3, ca=0.4)
        
        # Set up encoding options
        output_kwargs = {
            'acodec': 'copy',
            'pix_fmt': 'yuv420p'
        }
        
        if lossless:
            # Lossless CPU Encoding
            output_kwargs.update({
                'vcodec': 'libx264',
                'preset': 'ultrafast',
                'qp': 0
            })
        elif has_cuda:
            # GPU Encoding
            output_kwargs.update({
                'vcodec': 'h264_nvenc',
                'preset': preset,
                'tune': 'hq',
                'rc': 'vbr',
                'cq': cq,
                'multipass': '2',
                'maxrate': '40M',
                'bufsize': '80M',
                'rc-lookahead': 64,
                'spatial_aq': 1,
                'temporal_aq': 1,
                'aq-strength': 15,
                'g': 480,
                'bf': 3
            })
        else:
            # CPU Encoding Fallback (High Quality H.264)
            output_kwargs.update({
                'vcodec': 'libx264',
                'preset': 'slow',
                'crf': 18
            })
        
        stream = ffmpeg.output(stream, str(output_path), **output_kwargs)
        
        # Run with stats
        ffmpeg.run(
            stream,
            overwrite_output=True
        )
        
    except ffmpeg.Error as e:
        print(f"[stab] Error during Pass 2: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    
    # No cleanup of transforms_file as requested

    print("-" * 40)
    print("Stabilization complete!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Stabilize onboard video using ffmpeg and NVIDIA AV1.")
    parser.add_argument("input", type=Path, help="Input video file path")
    parser.add_argument("output", type=Path, nargs="?", help="Output video file path (optional)")
    # Transform params
    parser.add_argument("--smoothing", "-s", type=int, default=10, help="Smoothing window size (default: 10).")
    parser.add_argument("--zoom", "-z", type=int, default=0, help="Zoom percentage (default: 0).")
    parser.add_argument("--optzoom", type=int, default=0, choices=[0, 1, 2], help="Optimal zoom (0=off).")
    parser.add_argument("--unsharp", default=True, action="store_true", help="Apply unsharp mask (recommended for 2-pass).")
    # Detection params
    parser.add_argument("--shakiness", type=int, default=10, help="Shakiness 1-10 (default: 5).")
    parser.add_argument("--accuracy", type=int, default=15, help="Accuracy 1-15 (default: 15).")
    parser.add_argument("--stepsize", type=int, default=32, help="Step size (default: 3).")
    parser.add_argument("--crop", default="keep", choices=["black", "keep"], help="Border cropping: black (default) or keep.")
    parser.add_argument("--interpol", default="bilinear", choices=["no", "linear", "bilinear", "bicubic"], help="Interpolation: no, linear, bilinear, bicubic (default).")
    # Encoding params
    parser.add_argument("--cq", type=int, default=19, help="CQ value (default: 19).")
    parser.add_argument("--preset", type=str, default="p7", help="NVENC Preset (default: p7).")
    parser.add_argument("--overwrite", "-y", action="store_true", help="Overwrite output file if exists.")
    parser.add_argument("--lossless", action="store_true", help="Use lossless encoding (libx264 -qp 0).")

    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    if args.output:
        output_path = args.output.expanduser().resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_stab.mp4")

    stabilize_video(
        input_path, 
        output_path, 
        smoothing=args.smoothing, 
        zoom=args.zoom,
        optzoom=args.optzoom,
        shakiness=args.shakiness,
        accuracy=args.accuracy,
        stepsize=args.stepsize,
        unsharp=args.unsharp,
        crop=args.crop,
        interpol=args.interpol,
        cq=args.cq, 
        preset=args.preset, 
        overwrite=args.overwrite,
        lossless=args.lossless
    )

if __name__ == "__main__":
    main()
