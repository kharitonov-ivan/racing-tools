#!/usr/bin/env python3
import argparse
import sys
import ffmpeg
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Any

# Add the current directory to sys.path to allow importing sync_ui
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "render"))

from racing_tools.sync_ui import run_trim_selection


@dataclass
class VideoSidecar:
    """
    Universal sidecar file storage for video-related cached data.
    
    Usage:
        sidecar = VideoSidecar.load(video_path, "sync")  # .sync-{name}.json
        sidecar.data  # {"offset": 1.234}
        sidecar.save({"offset": 1.234})
    """
    video_path: Path
    key: str  # "sync", "trim", "crossings"
    data: dict = field(default_factory=dict)
    exists: bool = False

    @property
    def info_path(self) -> Path:
        return self.video_path.parent / f".{self.key}-{self.video_path.name}.json"

    @classmethod
    def load(cls, video_path: Path, key: str) -> "VideoSidecar":
        instance = cls(Path(video_path), key)
        if instance.info_path.exists():
            try:
                instance.data = json.loads(instance.info_path.read_text())
                instance.exists = True
            except Exception as e:
                print(f"Warning: Failed to load {instance.info_path}: {e}")
        return instance

    def save(self, data: dict):
        self.data, self.exists = data, True
        try:
            self.info_path.write_text(json.dumps(data, indent=2))
            print(f"Saved {self.key} data to {self.info_path}")
        except Exception as e:
            print(f"Warning: Failed to save {self.key}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass
class TrimInfo:
    video_path: Path
    start: float = 0.0
    end: Optional[float] = None
    exists: bool = False

    @property
    def info_path(self): 
        return self.video_path.with_suffix(".trim.json")

    @classmethod
    def load(cls, video_path):
        instance = cls(video_path)
        if instance.info_path.exists():
            try:
                with open(instance.info_path, "r") as f:
                    data = json.load(f)
                    instance.start = data.get("start", 0.0)
                    instance.end = data.get("end")
                    instance.exists = True
            except Exception as e:
                print(f"Warning: Failed to load {instance.info_path}: {e}")
        return instance

    def save(self, start, end):
        self.start, self.end, self.exists = start, end, True
        try:
            with open(self.info_path, "w") as f:
                json.dump({"start": start, "end": end}, f, indent=4)
            print(f"Saved trim info to {self.info_path}")
        except Exception as e:
            print(f"Warning: Failed to save trim info: {e}")


@dataclass
class CrossingsInfo:
    """Stores lap crossing times for a video."""
    video_path: Path
    times: list[float] = None
    exists: bool = False

    def __post_init__(self):
        if self.times is None:
            self.times = []

    @property
    def info_path(self):
        return self.video_path.parent / f".crossings-{self.video_path.name}.json"

    @classmethod
    def load(cls, video_path: Path) -> "CrossingsInfo":
        instance = cls(Path(video_path))
        if instance.info_path.exists():
            try:
                data = json.loads(instance.info_path.read_text())
                instance.times = data.get("times", [])
                instance.exists = bool(instance.times)
            except Exception as e:
                print(f"Warning: Failed to load {instance.info_path}: {e}")
        return instance

    def save(self, times: list[float]):
        self.times, self.exists = times, True
        try:
            self.info_path.write_text(json.dumps({"times": times}, indent=2))
            print(f"Saved {len(times)} crossing times to {self.info_path}")
        except Exception as e:
            print(f"Warning: Failed to save crossings: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Interactively trim a video.")
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive mode")
    return parser.parse_args()


def get_crossings_info(video_path: Path, start_time: float = 0.0, no_interactive: bool = False) -> CrossingsInfo:
    """
    Load or interactively create lap crossing times.
    
    Args:
        video_path: Path to video file.
        start_time: Initial seek position for interactive UI.
        no_interactive: If True, skip interactive UI if no saved data.
        
    Returns:
        CrossingsInfo with times list (may be empty if no data).
    """
    from racing_tools.sync_ui import run_manual_lap_marking
    
    info = CrossingsInfo.load(video_path)
    
    if info.exists:
        print(f"Found saved crossing times: {len(info.times)} laps at {info.times}")
        if not no_interactive:
            if input("Regenerate lap markings? [y/N]: ").strip().lower() == "y":
                info.exists = False  # Force regeneration
    
    if not info.exists and not no_interactive:
        times = run_manual_lap_marking(video_path, start_time=start_time)
        if times:
            info.save(times)
        else:
            print("No lap crossings marked")
    
    return info



def get_trim_info(video_path, no_interactive):
    info = TrimInfo.load(video_path)
    
    if info.exists:
        print(f"Found saved trim info: Start {info.start:.3f}s, End {info.end}")

    should_interact = False

    if info.exists:
        if no_interactive:
            should_interact = False
        else:
            # Ask user
            try:
                if input("Run interactive selection? [y/N]: ").strip().lower() == 'y':
                    should_interact = True
            except KeyboardInterrupt:
                sys.exit(0)
    else:
        # No info
        if no_interactive:
            print("No saved trim info and --no-interactive specified. Skipping trim (copying full).")
            print("Warning: Could not parse timestamp from filename. Appending -trimmed.")
            should_interact = False
        else:
            should_interact = True

    if should_interact:
        if run_trim_selection is None:
             sys.exit("Error: sync_ui module required for interactive mode.")
        
        print(f"Opening {video_path.name} for trim selection...")
        res = run_trim_selection(video_path)
        if hasattr(res, '__len__') and len(res) == 2 and all(x is not None for x in res):
             s, e = res
             print(f"Selected Trim: Start {s:.3f}s, End {e:.3f}s")
             info.save(s, e)
        else:
             sys.exit("Trim selection cancelled. Exiting.")

    return info


def main():
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    
    if not video_path.exists():
        sys.exit(f"Error: Video file {video_path} not found.")

    info = get_trim_info(video_path, args.no_interactive)

    if info.end is None:
        try:
            info.end = float(ffmpeg.probe(str(video_path))['format']['duration'])
        except Exception as e:
            sys.exit(f"Error probing duration: {e}")

    duration = info.end - info.start
    if duration <= 0:
        sys.exit("Error: Invalid duration (End <= Start).")

    # Output filename generation
    new_name = f"{video_path.stem}-trimmed{video_path.suffix}"
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", video_path.name)
    if match:
        ts = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S") + timedelta(seconds=info.start)
        new_name = video_path.name.replace(match.group(1), ts.strftime("%Y-%m-%d_%H-%M-%S"))
        if "-trimmed" not in new_name and "-trimmed" not in video_path.stem:
            new_name = Path(new_name).with_name(f"{Path(new_name).stem}-trimmed{Path(new_name).suffix}").name

    output_path = video_path.with_name(new_name)
    print(f"Output file: {output_path}")
    print(f"Running ffmpeg from {info.start:.3f} with duration {duration:.3f}")

    try:
        (
            ffmpeg
            .input(str(video_path), ss=info.start, t=duration)
            .output(str(output_path), c="copy", map="0")
            .overwrite_output()
            .run()
        )
        print(f"Trim successful!\n{output_path}")
    except ffmpeg.Error as e:
        sys.exit(f"Error running ffmpeg: {e}")

if __name__ == "__main__":
    main()
