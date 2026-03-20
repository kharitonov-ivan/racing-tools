"""
Track segmentation into straights, turns, and sectors.

Functions:
- segment_track: Split track into straights and turns based on curvature
- create_sectors_from_distances: Create sectors from distance breakpoints
- load_sectors_json: Load sector distances from JSON file
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pyproj import Transformer

from .constants import (
    SMOOTHING_WINDOW,
    MIN_SEGMENT_POINTS,
    DEFAULT_TURN_THRESHOLD,
    WGS84_CRS,
    WEBMERCATOR_CRS,
)
from .utils import normalize_angle, calculate_heading, get_transformer


def segment_track(
    polylines: List[List[Tuple[float, float]]],
    turn_threshold: float = DEFAULT_TURN_THRESHOLD,
) -> List[Dict]:
    """Split track into segments based on curvature.

    Args:
        polylines: List of polylines representing track
                   Each polyline: List of (x, y) tuples in meters (UTM)
        turn_threshold: Degrees of heading change per point to consider a turn

    Returns:
        List of segment dicts with keys:
            - 'type': 'straight' or 'turn'
            - 'points': List of (x, y) tuples

    Algorithm:
        1. Calculate heading at each point
        2. Calculate curvature (change in heading)
        3. Smooth curvature with sliding window
        4. Classify as turn/straight based on threshold
        5. Merge small segments to prevent noise
    """
    # Flatten polylines into a single list of points (assuming single loop)
    points = []
    for poly in polylines:
        points.extend(poly)

    if not points:
        return []

    # Convert to numpy for easier handling
    points_arr = np.array(points)

    # Calculate headings
    headings = []
    for i in range(len(points_arr) - 1):
        h = calculate_heading(points_arr[i], points_arr[i + 1])
        headings.append(h)

    # Calculate curvature (change in heading)
    curvatures = []
    for i in range(len(headings) - 1):
        diff = normalize_angle(headings[i + 1] - headings[i])
        curvatures.append(abs(diff))

    curvatures = [0] + curvatures + [0]

    segments = []
    current_type = None  # 'straight' or 'turn'
    current_points = []

    # Window size for smoothing
    window = SMOOTHING_WINDOW

    for i in range(len(points_arr) - 1):
        # Simple smoothing
        start = max(0, i - window)
        end = min(len(curvatures), i + window + 1)
        avg_curv = np.mean(curvatures[start:end])

        segment_type = 'turn' if avg_curv > turn_threshold else 'straight'

        if segment_type != current_type:
            if current_points:
                segments.append({"type": current_type, "points": current_points})
            current_type = segment_type
            current_points = [points_arr[i]]
        else:
            current_points.append(points_arr[i])

    # Add last segment
    if current_points:
        current_points.append(points_arr[-1])
        segments.append({"type": current_type, "points": current_points})

    # Merge small segments
    min_points = MIN_SEGMENT_POINTS

    if len(segments) > 1:
        cleaned_segments = []
        cleaned_segments.append(segments[0])

        for i in range(1, len(segments)):
            seg = segments[i]
            last = cleaned_segments[-1]

            if len(seg["points"]) < min_points:
                # Too small, merge into last
                last["points"].extend(seg["points"])
            else:
                # If type matches last (because we absorbed something), merge
                if seg["type"] == last["type"]:
                    last["points"].extend(seg["points"])
                else:
                    cleaned_segments.append(seg)

        segments = cleaned_segments

    return segments


def create_sectors_from_distances(
    polylines: List[List[Tuple[float, float]]],
    distances: List[float],
) -> List[Dict]:
    """Create sector segments based on distance breakpoints.

    Args:
        polylines: Track polylines - List of (lon, lat) tuples in WGS84 or (x, y) in WebMercator
        distances: List of cumulative distances defining sector boundaries [0, 500, 1200, ...]
                   in meters

    Returns:
        List of segment dicts with keys:
            - 'type': 'sector'
            - 'points': List of coordinate tuples
            - 'start_dist': Start distance in meters
            - 'end_dist': End distance in meters
            - 'index': Sector index (0-based)

    Example:
        >>> distances = [0, 500, 1200, 2000, 2500]  # 4 sectors
        >>> sectors = create_sectors_from_distances(polylines, distances)
    """
    # Flatten polylines
    all_points = []
    for poly in polylines:
        all_points.extend(poly)

    if not all_points:
        return []

    pts = np.array(all_points)

    # Project to meters if WGS84
    if np.max(np.abs(pts[:, 0])) <= 180 and np.max(np.abs(pts[:, 1])) <= 90:
        # Assume WGS84
        transformer = get_transformer(WGS84_CRS, WEBMERCATOR_CRS)
        xs, ys = transformer.transform(pts[:, 0], pts[:, 1])
        pts_m = np.column_stack((xs, ys))
    else:
        pts_m = pts

    # Calculate cumulative distance along track
    diffs = np.linalg.norm(pts_m[1:] - pts_m[:-1], axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(diffs)))

    # Create sectors
    sectors = []
    for i in range(len(distances) - 1):
        start_d = distances[i]
        end_d = distances[i + 1]

        # Find points in this distance range
        mask = (cum_dists >= start_d) & (cum_dists <= end_d)
        indices = np.where(mask)[0]

        if len(indices) > 0:
            sector_points = [tuple(all_points[j]) for j in indices]
            sectors.append({
                "type": "sector",
                "points": sector_points,
                "start_dist": start_d,
                "end_dist": end_d,
                "index": i,
            })

    return sectors


def load_sectors_json(path: str) -> Optional[List[float]]:
    """Load sector distances from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Sorted list of sector boundary distances [0, 500, 1200, ...],
        or None if file invalid/missing

    Expected format:
        {"distances": [0, 500, 1200, 2000, ...]}
    """
    path = path if isinstance(path, Path) else Path(path)

    if not path.is_file():
        return None

    with open(path, "r") as f:
        data = json.load(f)

    distances = data.get("distances", [])
    if not distances or len(distances) < 2:
        return None

    return sorted(distances)


def find_segment_by_distance(
    segments: List[Dict],
    distance: float,
    total_length: float,
) -> Optional[Dict]:
    """Find which segment contains a given distance along track.

    Args:
        segments: List of segments from segment_track()
        distance: Distance along track in meters
        total_length: Total track length in meters

    Returns:
        Segment dict if found, None otherwise
    """
    # Calculate cumulative segment lengths
    cum_length = 0.0
    for seg in segments:
        seg_points = np.array(seg["points"])
        seg_length = np.sum(np.linalg.norm(seg_points[1:] - seg_points[:-1], axis=1))

        if cum_length <= distance <= cum_length + seg_length:
            return seg

        cum_length += seg_length

    return None
