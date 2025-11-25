# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytesseract",
#     "Pillow",
#     "rich",
# ]
# ///

import argparse
import subprocess
import sys
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import shutil

import pytesseract
from PIL import Image, ImageOps
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

console = Console()

def get_video_duration(file_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        console.print(f"[red]Error getting duration for {file_path}: {e}[/red]")
        return 0.0

def extract_frame(file_path: Path, time_offset: float) -> Optional[Image.Image]:
    """Extract a frame at a specific time offset using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-ss", str(time_offset),
        "-i", str(file_path),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        from io import BytesIO
        try:
            return Image.open(BytesIO(result.stdout))
        except Exception as e:
            # console.print(f"[red]Error opening image from {file_path} at {time_offset}: {e}[/red]")
            return None
    except subprocess.CalledProcessError as e:
        # console.print(f"[red]Error extracting frame from {file_path} at {time_offset}: {e}[/red]")
        return None

def detect_timestamp_from_image(image: Image.Image, debug_save_path: Optional[Path] = None) -> Optional[datetime]:
    """
    Try multiple OCR configurations to detect a timestamp.
    Returns the first valid datetime found, or None.
    """
    w, h = image.size
    
    # Crop candidates (left, top, right, bottom)
    # 1. Right 50%, Bottom 20% (Safer)
    # 2. Right 40%, Bottom 15% (Tighter)
    crops = [
        (int(w * 0.50), int(h * 0.80), w, h),
        (int(w * 0.60), int(h * 0.85), w, h),
    ]
    
    # Threshold candidates
    thresholds = [200, 220, 180, 150]
    
    # Inversion candidates
    inversions = [False, True] # Normal (White on Black), Inverted (Black on White)
    
    # PSM candidates
    psms = [6, 11, 3]
    
    for crop_box in crops:
        cropped = image.crop(crop_box)
        gray = cropped.convert('L')
        
        for thresh in thresholds:
            binary = gray.point(lambda p: 255 if p > thresh else 0)
            
            for invert in inversions:
                if invert:
                    processed = ImageOps.invert(binary)
                else:
                    processed = binary
                
                for psm in psms:
                    config = f'--psm {psm}'
                    try:
                        text = pytesseract.image_to_string(processed, config=config)
                        dt = parse_timestamp(text)
                        if dt:
                            if debug_save_path:
                                # Save the successful image
                                processed.save(debug_save_path)
                            return dt
                    except Exception:
                        continue
                        
    # If we reach here, no timestamp found.
    # Save the last processed image for debug if requested
    if debug_save_path and 'processed' in locals():
        processed.save(debug_save_path)
        
    return None

# Removed separate preprocess_image_for_ocr and ocr_frame functions
# as they are now integrated into detect_timestamp_from_image

from collections import Counter

def detect_timestamp_robust(
    file_path: Path, 
    base_offset: float, 
    num_frames: int = 5, 
    interval: float = 1.0, 
    debug_folder: Optional[Path] = None,
    suffix: str = ""
) -> Optional[datetime]:
    """
    Detect timestamp by checking multiple frames and aggregating results.
    Normalizes all detected times to the base_offset time.
    """
    candidates = []
    
    for i in range(num_frames):
        offset = base_offset + (i * interval)
        
        # Ensure we don't go past video duration (though extract_frame handles this gracefully usually)
        # But we should check if we can get duration here? 
        # We'll rely on extract_frame failing or returning None if out of bounds, 
        # or just returning the last frame.
        
        img = extract_frame(file_path, offset)
        if not img:
            continue
            
        debug_path = None
        if debug_folder:
            debug_folder.mkdir(parents=True, exist_ok=True)
            debug_path = debug_folder / f"{file_path.stem}_{suffix}_{i}.jpg"
            
        dt = detect_timestamp_from_image(img, debug_path)
        if dt:
            # Normalize to base_offset
            # If we are at base_offset + 1s, the time should be T + 1s.
            # So base time T = dt - 1s.
            normalized_dt = dt - timedelta(seconds=i * interval)
            candidates.append(normalized_dt)
            
    if not candidates:
        return None
        
    # Voting logic
    # 1. Vote on Date
    dates = [c.date() for c in candidates]
    most_common_date = Counter(dates).most_common(1)[0][0]
    
    # Filter candidates that match the most common date
    valid_candidates = [c for c in candidates if c.date() == most_common_date]
    
    # 2. Select Time (Median)
    # Sort by time
    valid_candidates.sort()
    mid = len(valid_candidates) // 2
    return valid_candidates[mid]

def parse_timestamp(text: str) -> Optional[datetime]:
    """Parse timestamp from OCR text."""
    # Clean up text
    text = text.strip()
    
    # Try to find date and time
    # Patterns:
    # Date: DD/MM/YYYY or YYYY-MM-DD or DD-MM-YYYY
    # Time: HH:MM:SS, HH:MM'SS, HH.MM.SS
    
    # Regex for time
    # Allow : or ' or . as separators (removed space to avoid matching year parts like 2025 17:25)
    time_pattern = r'(\d{2})[:\'\.](\d{2})[:\'\.](\d{2})'
    time_match = re.search(time_pattern, text)
    
    # Regex for date
    # Allow / or - or . as separators
    # Allow 1 digit for day/month (e.g. 4/18/2025 instead of 14/18/2025 if OCR missed a digit)
    date_pattern = r'(\d{1,4})[/\-\.](\d{1,2})[/\-\.](\d{1,4})'
    date_match = re.search(date_pattern, text)
    
    if time_match and date_match:
        try:
            h, m, s = map(int, time_match.groups())
            d1, d2, d3 = map(int, date_match.groups())
            
            # Validate time
            if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
                return None

            # Figure out year, month, day
            year, month, day = 0, 0, 0
            
            # Helper to validate date
            def is_valid_date(y, m, d):
                try:
                    datetime(y, m, d)
                    return True
                except ValueError:
                    return False

            if d1 > 1000: # YYYY-MM-DD
                if is_valid_date(d1, d2, d3):
                    year, month, day = d1, d2, d3
            elif d3 > 1000: # DD-MM-YYYY or MM-DD-YYYY
                # Prefer DD-MM-YYYY
                if is_valid_date(d3, d2, d1):
                    year, month, day = d3, d2, d1
                elif is_valid_date(d3, d1, d2):
                    year, month, day = d3, d1, d2
            else:
                # Ambiguous 2-digit year
                # Assume d3 is year
                y = 2000 + d3 if d3 < 100 else d3
                if is_valid_date(y, d2, d1):
                     year, month, day = y, d2, d1
                elif is_valid_date(y, d1, d2):
                     year, month, day = y, d1, d2
            
            if year > 0:
                return datetime(year, month, day, h, m, s)
                
        except ValueError:
            pass
            
    return None

def process_videos(folder: Path, output_folder: Path, dry_run: bool = False, debug: bool = False):
    """Process videos in the folder."""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV'}
    files = sorted([f for f in folder.iterdir() if f.suffix in video_extensions])
    
    if not files:
        console.print("[yellow]No video files found.[/yellow]")
        return

    console.print(f"Found {len(files)} video files.")
    
    video_data = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Analyzing videos...", total=len(files))
        
        for i, file_path in enumerate(files):
            progress.update(task, description=f"Analyzing {file_path.name}...")
            progress.advance(task) # Added this line as per instruction's snippet
            
            duration = get_video_duration(file_path)
            
            # Robust detection for start time
            # Check 5 frames at 0, 1, 2, 3, 4 seconds
            start_time = detect_timestamp_robust(
                file_path, 
                base_offset=0, 
                num_frames=5, 
                interval=1.0, 
                debug_folder=output_folder / "debug" if debug else None,
                suffix="start"
            )
            
            if debug and start_time:
                console.print(f"  [dim]{file_path.name} Start: {start_time}[/dim]")

            # Robust detection for end time
            # Check 5 frames ending at duration-0.5s
            # e.g. if duration is 100, check 99.5, 98.5, 97.5, 96.5, 95.5
            # We start at duration - 0.5 and go backwards? 
            # Or better: start at duration - 4.5 and go forward to duration - 0.5
            
            end_base_offset = max(0, duration - 5.0)
            detected_end_time = detect_timestamp_robust(
                file_path,
                base_offset=end_base_offset,
                num_frames=5,
                interval=1.0,
                debug_folder=output_folder / "debug" if debug else None,
                suffix="end"
            )

            if detected_end_time:
                # Project to the actual end of the video
                # detected_end_time is at end_base_offset
                # We want time at duration
                end_time = detected_end_time + timedelta(seconds=(duration - end_base_offset))
                if debug:
                    console.print(f"  [dim]{file_path.name} End (projected): {end_time}[/dim]")
            else:
                end_time = None
            
            # Fallback if end time not detected: start_time + duration
            if start_time and not end_time:
                end_time = start_time + timedelta(seconds=duration)
            
            video_data.append({
                "file": file_path,
                "duration": duration,
                "start_time": start_time,
                "end_time": end_time
            })
            
            progress.advance(task)

    # Group videos
    # First, try to infer missing timestamps based on continuity
    # We can assume continuity if:
    # 1. Timestamps match (Start ~ Prev End)
    # 2. Timestamps + Duration match (End ~ Prev End + Duration)
    # 3. File modification times? (Maybe unreliable if copied)
    
    # Let's do a forward pass to fill Start from Prev End if it looks like a sequence
    # But we need to be careful not to merge separate sessions.
    
    # Better approach:
    # 1. Calculate theoretical start/end for everyone based on available data.
    #    If we have End, Start = End - Duration.
    #    If we have Start, End = Start + Duration.
    #    (We already did this partially)
    
    for i in range(len(video_data)):
        curr = video_data[i]
        if curr['start_time'] and not curr['end_time']:
            curr['end_time'] = curr['start_time'] + timedelta(seconds=curr['duration'])
        elif curr['end_time'] and not curr['start_time']:
            curr['start_time'] = curr['end_time'] - timedelta(seconds=curr['duration'])

    groups = []
    current_group = []
    
    for i, data in enumerate(video_data):
        if not current_group:
            current_group.append(data)
            continue
            
        prev = current_group[-1]
        
        is_continuous = False
        
        # Calculate theoretical end of previous video based on start + duration
        # This is often more reliable than OCR of the last frame
        prev_theoretical_end = None
        if prev['start_time']:
            prev_theoretical_end = prev['start_time'] + timedelta(seconds=prev['duration'])
        elif prev['end_time']:
            prev_theoretical_end = prev['end_time']

        is_continuous = False
        
        # Check 1: Continuity based on theoretical end (Start + Duration) vs Next Start
        if prev_theoretical_end and data['start_time']:
            diff = (data['start_time'] - prev_theoretical_end).total_seconds()
            if -2 < diff < 2: # Allow 2s gap/overlap
                is_continuous = True
            else:
                # Check for year OCR error (e.g. 2026 instead of 2025)
                try:
                    corrected_start = data['start_time'].replace(year=prev_theoretical_end.year)
                    diff_corrected = (corrected_start - prev_theoretical_end).total_seconds()
                    if -2 < diff_corrected < 2:
                        console.print(f"[yellow]Correcting year for {data['file'].name}: {data['start_time'].year} -> {prev_theoretical_end.year}[/yellow]")
                        data['start_time'] = corrected_start
                        # Also correct end_time if present
                        if data['end_time']:
                             data['end_time'] = data['end_time'].replace(year=prev_theoretical_end.year)
                        is_continuous = True
                except ValueError:
                    pass

        # Check 2: Explicit timestamps match (fallback if theoretical calculation failed or missing)
        elif prev['end_time'] and data['start_time']:
            diff = (data['start_time'] - prev['end_time']).total_seconds()
            if -2 < diff < 2:
                is_continuous = True
        
        # Check 3: If one timestamp is missing, check if the other aligns with duration
        # e.g. Prev End is known. Data Start is unknown. Data End is known.
        elif prev['end_time'] and data['end_time']:
             expected_end = prev['end_time'] + timedelta(seconds=data['duration'])
             diff = (data['end_time'] - expected_end).total_seconds()
             if -2 < diff < 2:
                 is_continuous = True
                 if not data['start_time']:
                     data['start_time'] = prev['end_time']
        
        # Check 4: If Prev End is missing, but Prev Start is known
        elif prev['start_time'] and data['start_time']:
            # This is covered by Check 1 usually, but just in case
            expected_start = prev['start_time'] + timedelta(seconds=prev['duration'])
            diff = (data['start_time'] - expected_start).total_seconds()
            if -2 < diff < 2:
                is_continuous = True
                if not prev['end_time']:
                    prev['end_time'] = expected_start

        if is_continuous:
            current_group.append(data)
        else:
            groups.append(current_group)
            current_group = [data]
            
    if current_group:
        groups.append(current_group)
        
    # Final pass: Fill missing start times for the first video in a group if possible?
    # If we have a group, and the 2nd video has a time, we can back-calculate the 1st video time.
    for group in groups:
        # Forward fill
        for i in range(1, len(group)):
            prev = group[i-1]
            curr = group[i]
            if prev['end_time'] and not curr['start_time']:
                curr['start_time'] = prev['end_time']
            if curr['start_time'] and not curr['end_time']:
                curr['end_time'] = curr['start_time'] + timedelta(seconds=curr['duration'])
                
        # Backward fill
        for i in range(len(group)-2, -1, -1):
            curr = group[i]
            next_v = group[i+1]
            if next_v['start_time'] and not curr['end_time']:
                curr['end_time'] = next_v['start_time']
            if curr['end_time'] and not curr['start_time']:
                curr['start_time'] = curr['end_time'] - timedelta(seconds=curr['duration'])

    # Process groups
    console.print(f"\nIdentified {len(groups)} video groups:")
    
    for idx, group in enumerate(groups):
        first = group[0]
        start_t = first['start_time']
        
        if not start_t:
            console.print(f"[red]Group {idx+1}: Could not determine start time for first video {first['file'].name}.[/red]")
            out_name = f"unknown_{first['file'].stem}.mp4"
        else:
            out_name = start_t.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
            
        console.print(f"  Group {idx+1} -> {out_name} ({len(group)} files)")
        for v in group:
            st = v['start_time'].strftime("%H:%M:%S") if v['start_time'] else "?"
            et = v['end_time'].strftime("%H:%M:%S") if v['end_time'] else "?"
            console.print(f"    - {v['file'].name} ({v['duration']:.2f}s) [{st} - {et}]")

    if dry_run:
        console.print("[yellow]Dry run complete. No files created.[/yellow]")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    for group in groups:
        first = group[0]
        start_t = first['start_time']
        if start_t:
            out_name = start_t.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
        else:
            out_name = f"unknown_{first['file'].stem}.mp4"
            
        out_path = output_folder / out_name
        
        if len(group) == 1:
            # Copy
            console.print(f"Copying {first['file'].name} to {out_path}...")
            shutil.copy2(first['file'], out_path)
        else:
            # Concat
            console.print(f"Concatenating group to {out_path}...")
            
            # Create list file for ffmpeg
            list_file = output_folder / "concat_list.txt"
            with open(list_file, "w") as f:
                for v in group:
                    # Escape path for ffmpeg
                    path_str = str(v['file'].absolute()).replace("'", "'\\''")
                    f.write(f"file '{path_str}'\n")
            
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                "-y", # Overwrite
                str(out_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                console.print(f"[green]Successfully created {out_path}[/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error concatenating: {e}[/red]")
            finally:
                if list_file.exists():
                    list_file.unlink()

def main():
    parser = argparse.ArgumentParser(description="Concatenate racing videos based on OCR timestamps.")
    parser.add_argument("input_folder", type=Path, help="Folder containing video files")
    parser.add_argument("--output", "-o", type=Path, help="Output folder (default: same as input)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, do not process")
    parser.add_argument("--debug", action="store_true", help="Save debug images for OCR")
    
    args = parser.parse_args()
    
    if not args.input_folder.exists():
        console.print(f"[red]Input folder {args.input_folder} does not exist.[/red]")
        sys.exit(1)
        
    output_folder = args.output if args.output else args.input_folder
        
    process_videos(args.input_folder, output_folder, args.dry_run, args.debug)

if __name__ == "__main__":
    main()
