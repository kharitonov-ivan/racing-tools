"""
GeoJSON validation utilities for track data.

Functions:
- validate_geojson_polyline: Validate GeoJSON polyline structure
- validate_geojson_crs: Check if coordinates are in expected CRS
- validate_track_directory: Validate complete track directory structure
"""

import json
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Optional, Tuple, List

from .constants import WGS84_CRS


def validate_geojson_polyline(path: Path) -> Tuple[bool, Optional[str]]:
    """Validate GeoJSON file contains a valid polyline.

    Args:
        path: Path to GeoJSON file

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if file is valid
        - error_message: None if valid, error description if invalid

    Checks:
        - File exists
        - Valid GeoJSON format
        - Contains at least one feature
        - Feature has valid geometry (LineString or MultiLineString)
        - Has at least 2 coordinates
        - Coordinates are numeric
    """
    # Check file exists
    if not path.is_file():
        return False, f"File not found: {path}"

    # Try to load and parse
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        return False, f"Failed to read GeoJSON: {e}"

    # Check not empty
    if gdf.empty:
        return False, "GeoJSON contains no features"

    # Find first valid geometry
    geom = None
    for g in gdf.geometry:
        if g is not None:
            geom = g
            break

    if geom is None:
        return False, "GeoJSON contains no valid geometries"

    # Check geometry type
    if geom.geom_type not in ['LineString', 'MultiLineString']:
        return False, f"Unsupported geometry type: {geom.geom_type}. Expected LineString or MultiLineString"

    # Extract coordinates
    if geom.geom_type == 'MultiLineString':
        # Use first part of MultiLineString
        geom = geom.geoms[0]

    coords = list(geom.coords)

    # Check minimum points
    if len(coords) < 2:
        return False, f"Polyline has only {len(coords)} points. Minimum 2 required"

    # Check coordinates are numeric
    for i, coord in enumerate(coords):
        if len(coord) < 2:
            return False, f"Coordinate {i} has insufficient dimensions: {coord}"
        try:
            float(coord[0])
            float(coord[1])
        except (ValueError, TypeError):
            return False, f"Coordinate {i} contains non-numeric values: {coord}"

    return True, None


def validate_geojson_crs(coords: List[Tuple[float, float]], expected_crs: str = WGS84_CRS) -> Tuple[bool, Optional[str]]:
    """Validate coordinates are in expected coordinate reference system.

    Args:
        coords: List of (x, y) or (lon, lat) tuples
        expected_crs: Expected CRS identifier (for error message)

    Returns:
        Tuple of (is_valid, error_message)

    Heuristics:
        - WGS84 (EPSG:4326): lon in [-180, 180], lat in [-90, 90]
        - Web Mercator (EPSG:3857): x, y in ~[-20M, 20M]
        - UTM: x in [0M, 1M], y varies by zone
    """
    if not coords:
        return False, "Empty coordinate list"

    coords_arr = np.array(coords)
    x_vals = coords_arr[:, 0]
    y_vals = coords_arr[:, 1]

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # Check if WGS84 (most common for GPS data)
    if expected_crs == WGS84_CRS:
        # WGS84: longitude [-180, 180], latitude [-90, 90]
        if x_max > 180 or x_min < -180:
            return False, f"X values [{x_min:.2f}, {x_max:.2f}] outside WGS84 longitude range [-180, 180]"
        if y_max > 90 or y_min < -90:
            return False, f"Y values [{y_min:.2f}, {y_max:.2f}] outside WGS84 latitude range [-90, 90]"

    return True, None


def validate_track_directory(track_dir: Path) -> Tuple[bool, List[str]]:
    """Validate track directory has required files and valid structure.

    Args:
        track_dir: Path to track directory

    Returns:
        Tuple of (is_valid, error_messages)

    Required files:
        - track-inner.geojson: Inner track boundary
        - track-outer.geojson: Outer track boundary

    Optional files:
        - start-finish.geojson: Start/finish line
        - bestline.geojson: Optimal racing line
        - track_config.json: Track metadata and UTM zone
    """
    track_dir = Path(track_dir)
    errors = []

    # Check directory exists
    if not track_dir.is_dir():
        return False, [f"Not a directory: {track_dir}"]

    # Check required files
    geometry_dir = track_dir / "geometry"
    required_files = {
        "geometry/track-inner.geojson": "Inner track boundary",
        "geometry/track-outer.geojson": "Outer track boundary",
    }

    for filename, description in required_files.items():
        filepath = track_dir / filename
        is_valid, error_msg = validate_geojson_polyline(filepath)
        if not is_valid:
            errors.append(f"{filename} ({description}): {error_msg}")

    # Validate optional files if present
    optional_files = {
        "geometry/start-finish.geojson": "Start/finish line",
        "geometry/bestline.geojson": "Optimal racing line",
    }

    for filename, description in optional_files.items():
        filepath = track_dir / filename
        if filepath.exists():
            is_valid, error_msg = validate_geojson_polyline(filepath)
            if not is_valid:
                errors.append(f"{filename} ({description}): {error_msg}")

    # Validate track_config.json if present
    config_path = track_dir / "track_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            # Check for version field
            if "version" not in config:
                errors.append("track_config.json: Missing 'version' field (recommended for future compatibility)")
            # Check for utm_zone
            if "utm_zone" not in config:
                errors.append("track_config.json: Missing 'utm_zone' field (will use default EPSG:32638)")
        except json.JSONDecodeError as e:
            errors.append(f"track_config.json: Invalid JSON: {e}")
        except Exception as e:
            errors.append(f"track_config.json: Error reading file: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors
