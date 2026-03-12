from dataclasses import dataclass
from pathlib import Path
import ffmpeg
from typing import Optional
from racing_tools.camera.model import CameraModel as CameraIntrinsics

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int
    intrinsics: Optional[CameraIntrinsics] = None


def to_float(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def probe_video(path: Path) -> VideoInfo:
    """Use ffprobe to collect geometry/fps info."""
    try:
        data = ffmpeg.probe(
            str(path),
            select_streams="v:0",
            show_entries="stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr.decode() if e.stderr else str(e)}") from e

    stream = data.get("streams", [{}])[0]
    fmt = data.get("format", {})
    rate = stream.get("r_frame_rate", "0/1")
    num, _, den = rate.partition("/")
    fps = to_float(num) / max(to_float(den), 1.0)
    if fps == 0.0:
        fps = 30.0

    duration = to_float(fmt.get("duration", stream.get("duration", 0.0)))
    
    nb_frames_str = stream.get("nb_frames")
    if nb_frames_str:
        nb_frames = int(nb_frames_str)
    else:
        nb_frames = int(duration * fps)

    return VideoInfo(
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=fps,
        duration=duration,
        nb_frames=nb_frames,
    )
