"""
Utility functions for track geometry processing.

Functions:
- normalize_angle: Normalize angle to -180 to 180
- calculate_heading: Calculate heading between two points
- load_polyline_geojson: Load polyline from GeoJSON file
- load_track_config: Load track configuration from JSON
- get_transformer: Get coordinate system transformer
- transform_coordinates: Transform coordinates between CRS
- resample_linestring: Uniform arc-length resampling
- compute_centerline: Generate centerline from boundaries
"""

import json
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from pyproj import Transformer
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from .constants import (
    CONTINUITY_WEIGHT,
    DEFAULT_CENTERLINE_SAMPLES,
    K_NEIGHBORS,
    DUPLICATE_TOLERANCE,
    DEFAULT_UTM_ZONE,
    WGS84_CRS,
    WEBMERCATOR_CRS,
)


def normalize_angle(angle: float) -> float:
    """Normalize angle to -180 to 180.

    Args:
        angle: Angle in degrees

    Returns:
        Normalized angle in range [-180, 180]
    """
    return ((angle + 180) % 360) - 180


def calculate_heading(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate heading between two points in degrees.

    Args:
        p1: First point [x, y], shape (2,) - in meters (UTM)
        p2: Second point [x, y], shape (2,) - in meters (UTM)

    Returns:
        Heading in degrees [0, 360) measured from East counterclockwise
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = np.arctan2(dy, dx)
    deg = np.degrees(rads)
    return deg


def load_polyline_geojson(path: Path) -> Optional[List[Tuple[float, float]]]:
    """Load polyline from GeoJSON file.

    Args:
        path: Path to GeoJSON file

    Returns:
        List of (lon, lat) tuples in WGS84, or None if file invalid/empty

    Supports:
        - LineString geometries
        - MultiLineString (takes first part)
        - Skips null geometries
    """
    if not path.is_file():
        return None

    gdf = gpd.read_file(path)
    if gdf.empty:
        return None

    # Find first feature with a non-null geometry
    geom = None
    for g in gdf.geometry:
        if g is not None:
            geom = g
            break

    if geom is None:
        return None

    # Handle MultiLineString by taking first part
    if geom.geom_type == 'MultiLineString':
        geom = geom.geoms[0]

    # Extract coordinates (ignore Z if present)
    coords = list(geom.coords)
    return [(c[0], c[1]) for c in coords]


def load_track_config(track_dir: Path) -> Dict:
    """Load track configuration from track_config.json.

    Args:
        track_dir: Track directory path

    Returns:
        Config dict with keys:
            - 'utm_zone': EPSG code for UTM projection (default: EPSG:32638)
            - 'name': Track name (optional)
            - 'version': Config format version (optional, default: 1)
            - Other metadata fields

    Example track_config.json:
        {
            "utm_zone": "EPSG:32638",
            "name": "RIM Sport Karting",
            "version": 1,
            "comment": "UTM Zone 38N for accurate scaling"
        }
    """
    config_path = Path(track_dir) / "track_config.json"

    default_config = {
        "utm_zone": DEFAULT_UTM_ZONE,
        "name": None,
        "version": 1,
    }

    if not config_path.is_file():
        return default_config

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return {**default_config, **config}
    except (json.JSONDecodeError, IOError):
        return default_config


def get_transformer(
    from_crs: str = WGS84_CRS,
    to_crs: str = DEFAULT_UTM_ZONE,
) -> Transformer:
    """Get coordinate system transformer.

    Args:
        from_crs: Source CRS (default: WGS84)
        to_crs: Target CRS (default: UTM Zone 38N)

    Returns:
        PyProj Transformer object

    Common CRS values:
        - EPSG:4326: WGS84 (GPS coordinates in degrees)
        - EPSG:3857: Web Mercator (web mapping, distorts distances)
        - EPSG:32638: UTM Zone 38N (accurate for Georgia)

    Example:
        >>> transformer = get_transformer("EPSG:4326", "EPSG:32638")
        >>> x_m, y_m = transformer.transform(lons, lats)
    """
    return Transformer.from_crs(from_crs, to_crs, always_xy=True)


def transform_coordinates(
    coords: np.ndarray,
    from_crs: str = WGS84_CRS,
    to_crs: str = DEFAULT_UTM_ZONE,
) -> np.ndarray:
    """Transform coordinates from one CRS to another.

    Args:
        coords: Input coordinates, shape (N, 2) - [x, y] or [lon, lat]
        from_crs: Source CRS
        to_crs: Target CRS

    Returns:
        Transformed coordinates, shape (N, 2) - [x, y] in target CRS

    Example:
        >>> # Convert WGS84 to UTM
        >>> wgs84_coords = np.array([[44.5, 41.7], [44.6, 41.8]])  # [lon, lat]
        >>> utm_coords = transform_coordinates(wgs84_coords, "EPSG:4326", "EPSG:32638")
    """
    transformer = get_transformer(from_crs, to_crs)
    xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
    return np.column_stack((xs, ys))


def resample_linestring(
    coords: np.ndarray,
    n_samples: int,
    tolerance: float = DUPLICATE_TOLERANCE,
) -> np.ndarray:
    """Resample closed loop uniformly by arc length.

    Removes duplicate points, enforces a closed loop, samples equally spaced points.

    Args:
        coords: Input coordinates, shape (N, 2) - [x, y] in meters (UTM)
        n_samples: Number of output samples (recommended: 512-1024)
        tolerance: Minimum distance (m) to consider points distinct

    Returns:
        Resampled coordinates, shape (n_samples, 2)

    Algorithm:
        1. Remove duplicate points within tolerance
        2. Ensure closed loop (last point ≈ first point)
        3. Calculate cumulative arc length
        4. Interpolate at n_samples equally spaced positions
    """
    # Remove duplicates (vectorized approach is more efficient)
    if len(coords) <= 1:
        return coords

    unique_coords = [coords[0]]
    for i in range(1, len(coords)):
        dist = np.linalg.norm(coords[i] - unique_coords[-1])
        if dist > tolerance:
            unique_coords.append(coords[i])

    # Check if closed loop
    if len(unique_coords) > 1:
        dist_to_first = np.linalg.norm(unique_coords[-1] - unique_coords[0])
        if dist_to_first < tolerance:
            unique_coords = unique_coords[:-1]

    coords = np.array(unique_coords)

    if len(coords) < 3:
        return coords

    # Calculate segments
    segments = []
    total = 0.0
    for i in range(len(coords)):
        p0 = coords[i]
        p1 = coords[(i + 1) % len(coords)]
        d = np.linalg.norm(p1 - p0)
        segments.append((p0, p1, d))
        total += d

    if total < 1e-6:
        return coords

    # Resample
    result = []
    step = total / n_samples
    acc = 0.0
    si = 0

    for k in range(n_samples):
        target = k * step
        while target > acc + segments[si][2] and segments[si][2] > 0.0:
            acc += segments[si][2]
            si = (si + 1) % len(segments)

        p0, p1, seg_len = segments[si]
        t = 0.0 if seg_len == 0.0 else (target - acc) / seg_len
        result.append(p0 * (1 - t) + p1 * t)

    return np.array(result)


def compute_centerline(
    inner_coords: np.ndarray,
    outer_coords: np.ndarray,
    n_samples: int = DEFAULT_CENTERLINE_SAMPLES,
    k_neigh: int = K_NEIGHBORS,
) -> np.ndarray:
    """Compute centerline by pairing resampled outer points with nearest inner points.

    Uses dynamic programming approach to maintain continuity while minimizing distance.

    Args:
        inner_coords: Inner boundary coordinates, shape (N, 2) - [x, y] in meters (UTM)
        outer_coords: Outer boundary coordinates, shape (M, 2) - [x, y] in meters (UTM)
        n_samples: Number of samples for resampling (recommended: 512-1024)
        k_neigh: Number of neighbors to consider for pairing (recommended: 8)

    Returns:
        Centerline coordinates, shape (n_samples, 2) - [x, y] in meters (UTM)

    Algorithm:
        1. Resample both boundaries to n_samples equally spaced points
        2. Build KD-tree on inner boundary for fast nearest-neighbor queries
        3. Greedily match outer points to inner points with continuity constraint
        4. Compute midpoints of matched pairs as centerline
        5. Smooth centerline with 2-pass averaging

    Note:
        Continuity weight (CONTINUITY_WEIGHT) balances smoothness vs. minimum distance.
        Higher values produce smoother centerlines but may deviate from geometric center.
    """
    # Resample both boundaries uniformly
    inner_resampled = resample_linestring(inner_coords, n_samples)
    outer_resampled = resample_linestring(outer_coords, n_samples)

    # Build KD-tree for fast nearest neighbor queries
    tree = cKDTree(inner_resampled)

    # Start with closest match for first outer point
    _, j0 = tree.query(outer_resampled[0])
    prev_j = j0
    pairs = [(0, prev_j)]

    # Greedy matching with continuity constraint
    for i in range(1, n_samples):
        # Query k nearest neighbors
        _, candidates = tree.query(outer_resampled[i], k=k_neigh)
        if np.isscalar(candidates):
            candidates = [int(candidates)]

        # Select candidate that minimizes: distance + continuity_cost
        best_j = None
        best_cost = float("inf")
        for cand in candidates:
            # Forward and wraparound distances (for closed loop)
            diff = (cand - prev_j) % n_samples
            wrap_diff = (prev_j - cand) % n_samples
            continuity_cost = min(diff, wrap_diff)

            # Total cost = distance + continuity_weight * continuity_penalty
            dist_cost = np.linalg.norm(inner_resampled[cand] - outer_resampled[i])
            cost = dist_cost + CONTINUITY_WEIGHT * continuity_cost

            if cost < best_cost:
                best_cost = cost
                best_j = cand

        prev_j = best_j
        pairs.append((i, best_j))

    # Compute midpoints as centerline
    centerline = []
    for i, j in pairs:
        mid = (outer_resampled[i] + inner_resampled[j]) / 2.0
        centerline.append(mid)

    centerline = np.array(centerline)

    # Smooth centerline (2-pass averaging)
    for _ in range(2):
        smoothed = centerline.copy()
        for i in range(n_samples):
            prev_pt = centerline[(i - 1) % n_samples]
            next_pt = centerline[(i + 1) % n_samples]
            avg = (prev_pt + next_pt) / 2.0
            smoothed[i] = centerline[i] * 0.5 + avg * 0.5
        centerline = smoothed

    return centerline
