from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from concat import extract_frame, detect_timestamp_from_image


def debug_crop(video_path: Path, output_dir: Path) -> None:
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    print(f"Processing {video_path.name}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check multiple frames to verify crop consistency
    timestamps = [0.0, 10.0, 30.0, 60.0]

    for t in timestamps:
        print(f"\nChecking frame at {t}s...")
        img = extract_frame(video_path, t)
        if not img:
            print(f"Failed to extract frame at {t}s")
            continue

        # Save original
        original_path = output_dir / f"{video_path.stem}_original_{int(t)}.jpg"
        img.save(original_path)

        # Visualize crop on original
        w, h = img.size
        # Ratios from concat_videos.py: 0.80729, 0.935185, 1.0, 1.0
        crop_box = (int(w * 0.80729), int(h * 0.935185), w, h)

        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        draw.rectangle(crop_box, outline="red", width=5)
        vis_path = output_dir / f"{video_path.stem}_visualized_{int(t)}.jpg"
        vis_img.save(vis_path)
        print(f"Saved visualization: {vis_path}")

        # Perform actual crop and OCR
        cropped = img.crop(crop_box)
        crop_path = output_dir / f"{video_path.stem}_cropped_{int(t)}.jpg"
        cropped.save(crop_path)

        # Run OCR
        dt = detect_timestamp_from_image(
            img,
            debug_save_path=output_dir / f"{video_path.stem}_ocr_debug_{int(t)}.jpg",
        )
        print(f"Detected Timestamp at {t}s: {dt}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug video crop and OCR")
    parser.add_argument("video", type=Path, nargs="?", help="Path to video file")
    parser.add_argument("--output", "-o", type=Path, default=Path("debug_crop_output"), help="Output directory")

    args = parser.parse_args()

    if args.video:
        video = args.video
    else:
        # Fallback to old path for backwards compatibility
        video_dir = Path(__file__).parent.parent.parent / "VIDEO"
        video = video_dir / "FHD0002.MOV"

    if video.exists():
        debug_crop(video, args.output)
    else:
        print(f"Video not found: {video}")
        sys.exit(1)
