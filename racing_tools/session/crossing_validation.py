"""Validation for video and GPS crossing alignment."""

import numpy as np

from racing_tools.track.constants import MIN_VALID_LAP_TIME


def validate_crossings(
    crossings_video: list[float],
    crossings_telem: list[float],
    max_lap_delta: float = 0.3,
    min_valid_lap_time: float = MIN_VALID_LAP_TIME,
) -> None:
    """Validate video and GPS crossing alignment.

    Checks:
        - Both lists are strictly monotonically increasing
        - At least 2 crossings in each list (1 lap minimum)
        - Lap counts match between video and telemetry
        - Lap time deltas between video and GPS are within max_lap_delta
        - Warns about pit-in/pit-out laps (abnormally long)

    Args:
        crossings_video: Video crossing times in seconds
        crossings_telem: Telemetry (GPS) crossing times in seconds
        max_lap_delta: Max allowed difference between video and GPS lap times (seconds)
        min_valid_lap_time: Minimum lap time to be considered valid (seconds)

    Raises:
        ValueError: If any hard validation check fails
    """
    _assert_monotonic(crossings_video, "video")
    _assert_monotonic(crossings_telem, "telemetry")
    _assert_minimum_crossings(crossings_video, "video")
    _assert_minimum_crossings(crossings_telem, "telemetry")
    _assert_lap_count_match(crossings_video, crossings_telem)

    video_laps = _compute_lap_times(crossings_video)
    telem_laps = _compute_lap_times(crossings_telem)

    _warn_pit_laps(video_laps, min_valid_lap_time, source="video")
    _warn_pit_laps(telem_laps, min_valid_lap_time, source="telemetry")
    _assert_lap_time_deltas(video_laps, telem_laps, max_lap_delta)

    print(f"[Validation] All {len(video_laps)} laps passed (max_delta={max_lap_delta}s)")


def _assert_monotonic(crossings: list[float], source: str) -> None:
    """Assert crossings are strictly increasing."""
    for i in range(1, len(crossings)):
        assert crossings[i] > crossings[i - 1], (
            f"{source} crossings not monotonic at index {i}: "
            f"{crossings[i - 1]:.3f}s >= {crossings[i]:.3f}s"
        )


def _assert_minimum_crossings(crossings: list[float], source: str) -> None:
    """Assert at least 2 crossings exist (1 lap minimum)."""
    assert len(crossings) >= 2, (
        f"{source} has only {len(crossings)} crossing(s), need at least 2 for 1 lap"
    )


def _assert_lap_count_match(
    crossings_video: list[float], crossings_telem: list[float]
) -> None:
    """Assert video and telemetry have the same number of crossings."""
    n_video = len(crossings_video)
    n_telem = len(crossings_telem)
    assert n_video == n_telem, (
        f"Crossing count mismatch: video={n_video}, telemetry={n_telem}. "
        f"Check for missing or extra pit-in/pit-out laps."
    )


def _compute_lap_times(crossings: list[float]) -> list[float]:
    """Compute lap times from consecutive crossings. Returns list of shape (N-1,)."""
    arr = np.array(crossings)
    return list(np.diff(arr))


def _warn_pit_laps(
    lap_times: list[float], min_valid_lap_time: float, source: str
) -> None:
    """Warn about pit-in/pit-out laps (abnormally long or short)."""
    valid_laps = [t for t in lap_times if t >= min_valid_lap_time]
    if not valid_laps:
        return

    median_time = float(np.median(valid_laps))
    pit_threshold = median_time * 2.0

    for i, lap_time in enumerate(lap_times):
        if lap_time < min_valid_lap_time:
            print(
                f"[Validation] WARNING: {source} lap {i + 1} is too short "
                f"({lap_time:.1f}s < {min_valid_lap_time:.1f}s) — likely pit-out"
            )
        elif lap_time > pit_threshold:
            print(
                f"[Validation] WARNING: {source} lap {i + 1} is abnormally long "
                f"({lap_time:.1f}s > {pit_threshold:.1f}s median×2) — likely pit-in"
            )


def _assert_lap_time_deltas(
    video_laps: list[float], telem_laps: list[float], max_delta: float
) -> None:
    """Assert lap time differences between video and GPS are within threshold."""
    for i, (v_lap, t_lap) in enumerate(zip(video_laps, telem_laps)):
        delta = abs(v_lap - t_lap)
        assert delta <= max_delta, (
            f"Lap {i + 1} time delta too large: "
            f"video={v_lap:.3f}s, telem={t_lap:.3f}s, delta={delta:.3f}s > {max_delta}s"
        )
