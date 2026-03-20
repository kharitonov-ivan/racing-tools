"""Telemetry synchronization utilities for aligning video and telemetry data."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from racing_tools.session.video_info import VideoInfo

from racing_tools.session.session import Session


def _align_crossings(video_cx: list[float], telem_cx: list[float]) -> tuple[list[tuple[float, float]], int]:
    """Auto-align video and telemetry crossings by matching interval patterns.

    Tries every possible offset between the two lists and picks the one
    with lowest RMSE of interval durations. Returns (anchors, best_offset)
    where best_offset is the index in video_cx that maps to telem_cx[0].
    """
    if not video_cx or not telem_cx:
        return [], 0

    v_intervals = [video_cx[i + 1] - video_cx[i] for i in range(len(video_cx) - 1)]
    t_intervals = [telem_cx[i + 1] - telem_cx[i] for i in range(len(telem_cx) - 1)]

    # Filter out abnormal intervals (> 2x median) for matching
    t_median = sorted(t_intervals)[len(t_intervals) // 2]
    v_median = sorted(v_intervals)[len(v_intervals) // 2]
    max_valid_interval = max(t_median, v_median) * 1.5

    print(f"[Sync] Video intervals: {[f'{v:.1f}' for v in v_intervals]}")
    print(f"[Sync] Telem intervals: {[f'{t:.1f}' for t in t_intervals]}")
    print(f"[Sync] Median video={v_median:.1f}s, telem={t_median:.1f}s, max_valid={max_valid_interval:.1f}s")

    best_rmse = float("inf")
    best_offset = 0

    # Try each possible shift of video relative to telem
    for shift in range(len(video_cx)):
        n = 0
        sse = 0.0
        for j in range(len(t_intervals)):
            vi = shift + j
            if vi >= len(v_intervals):
                break
            # Skip abnormal intervals in matching
            if v_intervals[vi] > max_valid_interval or t_intervals[j] > max_valid_interval:
                continue
            diff = v_intervals[vi] - t_intervals[j]
            sse += diff * diff
            n += 1
        if n >= 2:
            rmse = (sse / n) ** 0.5
            if rmse < best_rmse:
                best_rmse = rmse
                best_offset = shift

    # Build anchors with best alignment
    anchors: list[tuple[float, float]] = []
    for j in range(len(telem_cx)):
        vi = best_offset + j
        if 0 <= vi < len(video_cx):
            anchors.append((video_cx[vi], telem_cx[j]))

    print(f"[Sync] Auto-align: best_offset={best_offset}, RMSE={best_rmse:.4f}s, {len(anchors)} pairs")
    for i, (v, t) in enumerate(anchors[:5]):
        print(f"  Anchor {i}: video={v:.2f}s, telem={t:.2f}s, offset={v - t:.2f}s")
    if len(anchors) > 5:
        print(f"  ... ({len(anchors)} total)")
    return anchors, best_offset


def create_session_from_crossings(video_info: "VideoInfo", crossing_times: list[float]) -> Session:
    """
    Create a Session object from video info and manual crossing times (no telemetry).

    Creates one row per video frame with:
    - Time: frame_number / fps
    - Duration: cumulative time
    - LapNumber: assigned from crossing_times (before first crossing = Lap 0)

    Args:
        video_info: VideoInfo with fps, nb_frames, etc.
        crossing_times: Sorted list of lap crossing times in seconds.

    Returns:
        Session object with minimal frame-based data.
    """
    from bisect import bisect_right

    fps = video_info.fps
    nb_frames = video_info.nb_frames

    # Create time array for each frame
    times = np.arange(nb_frames) / fps

    # Assign lap numbers based on crossing times
    # Everything before first crossing is Lap 0
    # crossing_times[0] -> start of Lap 1, etc.
    sorted_crossings = sorted(crossing_times) if crossing_times else []

    def get_lap_number(t: float) -> int:
        if not sorted_crossings:
            return 0
        return bisect_right(sorted_crossings, t)

    lap_numbers = np.array([get_lap_number(t) for t in times])

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Time": times,
            "Duration": times,
            "LapNumber": lap_numbers,
        }
    )

    return Session(table=df)
