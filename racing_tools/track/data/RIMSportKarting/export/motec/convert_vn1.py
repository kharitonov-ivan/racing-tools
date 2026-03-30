#!/usr/bin/env python3
"""
VN1 Conversion Utilities

Convert between MoTeC VN1 format and other formats (GeoJSON, CSV, etc.)
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple
import struct

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vn1_codec import decode_vn1, VN1Venue, VN1Metadata


def geojson_to_vn1(geojson_file: str, vn1_output: str, track_name: str = "Track", reference_vn1: str = None) -> None:
    """
    Convert GeoJSON to VN1 file

    Args:
        geojson_file: Input GeoJSON file path
        vn1_output: Output VN1 file path
        track_name: Name for the track
        reference_vn1: Optional reference VN1 file to copy metadata from
    """
    with open(geojson_file, "r") as f:
        gj = json.load(f)

    # Extract coordinates from GeoJSON
    coordinates = []
    features = gj.get("features", [])

    for feature in features:
        geom = feature.get("geometry", {})
        geom_type = geom.get("type", "")

        if geom_type == "LineString":
            coords = geom.get("coordinates", [])
            for lon, lat, *rest in coords:
                coordinates.append((lon, lat))

        elif geom_type == "Point":
            coords = geom.get("coordinates", [])
            if coords:
                lon, lat, *rest = coords
                coordinates.append((lon, lat))

        elif geom_type == "MultiLineString":
            for line in geom.get("coordinates", []):
                for lon, lat, *rest in line:
                    coordinates.append((lon, lat))

    if not coordinates:
        raise ValueError("No coordinates found in GeoJSON")

    # Load reference if provided
    ref_data = None
    metadata = None
    if reference_vn1:
        with open(reference_vn1, "rb") as f:
            ref_data = f.read()
        metadata_bytes = ref_data[0x950:0x980]
        metadata = VN1Metadata(raw_bytes=metadata_bytes)

    # Create venue
    venue = VN1Venue(name=track_name, source_path="Generated from GeoJSON", coordinates=coordinates, metadata=metadata)

    # Encode
    from vn1_codec import VN1Encoder

    encoder = VN1Encoder()
    encoder.encode(venue, vn1_output, reference=ref_data)
    print(f"Converted {len(coordinates)} coordinates to {vn1_output}")


def vn1_to_geojson(vn1_file: str, geojson_output: str, feature_name: str = "track") -> None:
    """
    Convert VN1 file to GeoJSON

    Args:
        vn1_file: Input VN1 file path
        geojson_output: Output GeoJSON file path
        feature_name: Name for the feature
    """
    venue = decode_vn1(vn1_file)

    # Create GeoJSON structure
    feature = {
        "type": "Feature",
        "properties": {"name": venue.name, "source": venue.source_path, "coordinate_count": len(venue.coordinates)},
        "geometry": {"type": "LineString", "coordinates": [[lon, lat, 0.0] for lon, lat in venue.coordinates]},
    }

    geojson = {"type": "FeatureCollection", "features": [feature]}

    with open(geojson_output, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Converted {len(venue.coordinates)} coordinates to {geojson_output}")


def vn1_to_csv(vn1_file: str, csv_output: str) -> None:
    """
    Convert VN1 file to CSV

    Args:
        vn1_file: Input VN1 file path
        csv_output: Output CSV file path
    """
    venue = decode_vn1(vn1_file)

    with open(csv_output, "w") as f:
        f.write("index,latitude,longitude\n")
        for i, (lon, lat) in enumerate(venue.coordinates):
            f.write(f"{i},{lat:.10f},{lon:.10f}\n")

    print(f"Converted {len(venue.coordinates)} coordinates to {csv_output}")


def csv_to_vn1(csv_file: str, vn1_output: str, track_name: str = "Track", reference_vn1: str = None) -> None:
    """
    Convert CSV to VN1 file

    CSV format: index,latitude,longitude

    Args:
        csv_file: Input CSV file path
        vn1_output: Output VN1 file path
        track_name: Name for the track
        reference_vn1: Optional reference VN1 file to copy metadata from
    """
    coordinates = []

    with open(csv_file, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                idx, lat, lon = parts[0], float(parts[1]), float(parts[2])
                coordinates.append((lon, lat))

    if not coordinates:
        raise ValueError("No coordinates found in CSV")

    # Load reference if provided
    ref_data = None
    metadata = None
    if reference_vn1:
        with open(reference_vn1, "rb") as f:
            ref_data = f.read()
        metadata_bytes = ref_data[0x950:0x980]
        metadata = VN1Metadata(raw_bytes=metadata_bytes)

    # Create venue
    venue = VN1Venue(name=track_name, source_path="Generated from CSV", coordinates=coordinates, metadata=metadata)

    # Encode
    from vn1_codec import VN1Encoder

    encoder = VN1Encoder()
    encoder.encode(venue, vn1_output, reference=ref_data)
    print(f"Converted {len(coordinates)} coordinates to {vn1_output}")


def simplify_coordinates(coordinates: List[Tuple[float, float]], tolerance: float = 0.00001) -> List[Tuple[float, float]]:
    """
    Simplify coordinates using Douglas-Peucker algorithm (simplified)

    Args:
        coordinates: List of (lon, lat) tuples
        tolerance: Simplification tolerance in degrees

    Returns:
        Simplified list of coordinates
    """
    if len(coordinates) <= 2:
        return coordinates

    # Find point with maximum distance from line
    max_dist = 0
    max_idx = 0
    start = coordinates[0]
    end = coordinates[-1]

    for i in range(1, len(coordinates) - 1):
        dist = point_line_distance(coordinates[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # Recursively simplify if max distance > tolerance
    if max_dist > tolerance:
        left = simplify_coordinates(coordinates[: max_idx + 1], tolerance)
        right = simplify_coordinates(coordinates[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [start, end]


def point_line_distance(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
    """Calculate perpendicular distance from point to line"""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Line length
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2) ** 0.5

    if length == 0:
        return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

    # Perpendicular distance
    return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / length


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert VN1 files")
    subparsers = parser.add_subparsers(dest="command", help="Conversion command")

    # GeoJSON to VN1
    geojson_to_vn1_parser = subparsers.add_parser("geojson-to-vn1", help="Convert GeoJSON to VN1")
    geojson_to_vn1_parser.add_argument("input", help="Input GeoJSON file")
    geojson_to_vn1_parser.add_argument("output", help="Output VN1 file")
    geojson_to_vn1_parser.add_argument("--name", default="Track", help="Track name")
    geojson_to_vn1_parser.add_argument("--reference", help="Reference VN1 file")

    # VN1 to GeoJSON
    vn1_to_geojson_parser = subparsers.add_parser("vn1-to-geojson", help="Convert VN1 to GeoJSON")
    vn1_to_geojson_parser.add_argument("input", help="Input VN1 file")
    vn1_to_geojson_parser.add_argument("output", help="Output GeoJSON file")
    vn1_to_geojson_parser.add_argument("--feature-name", default="track", help="Feature name in GeoJSON")

    # VN1 to CSV
    vn1_to_csv_parser = subparsers.add_parser("vn1-to-csv", help="Convert VN1 to CSV")
    vn1_to_csv_parser.add_argument("input", help="Input VN1 file")
    vn1_to_csv_parser.add_argument("output", help="Output CSV file")

    # CSV to VN1
    csv_to_vn1_parser = subparsers.add_parser("csv-to-vn1", help="Convert CSV to VN1")
    csv_to_vn1_parser.add_argument("input", help="Input CSV file")
    csv_to_vn1_parser.add_argument("output", help="Output VN1 file")
    csv_to_vn1_parser.add_argument("--name", default="Track", help="Track name")
    csv_to_vn1_parser.add_argument("--reference", help="Reference VN1 file")

    args = parser.parse_args()

    if args.command == "geojson-to-vn1":
        geojson_to_vn1(args.input, args.output, args.name, args.reference)
    elif args.command == "vn1-to-geojson":
        vn1_to_geojson(args.input, args.output, args.feature_name)
    elif args.command == "vn1-to-csv":
        vn1_to_csv(args.input, args.output)
    elif args.command == "csv-to-vn1":
        csv_to_vn1(args.input, args.output, args.name, args.reference)
    else:
        parser.print_help()
