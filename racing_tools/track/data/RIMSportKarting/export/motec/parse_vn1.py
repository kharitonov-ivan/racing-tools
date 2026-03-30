#!/usr/bin/env python3
"""
MoTeC Venue (.VN1) file parser

Format structure:
- 0x00-0x40: Track name (padded with nulls)
- 0x40-0x80: Source file path
- 0x80-0x140: Reserved/padding
- 0x148-0x150: Format markers (big-endian count, etc)
- 0x950-0x980: Metadata (possibly sector/finish info)
- 0x980-0x9d4: Coordinate data (lat*1e7, lon*1e7 as int32 pairs)

Coordinates: lat/lon stored as signed 32-bit integers scaled by 10^7
"""

import struct
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class VN1Venue:
    name: str
    source_path: str
    coordinates: List[Tuple[float, float]]  # (lon, lat) in decimal degrees
    metadata: bytes


def parse_vn1(filepath: str) -> VN1Venue:
    """Parse MoTeC VN1 venue file"""
    with open(filepath, 'rb') as f:
        data = f.read()

    # Parse header
    name = data[0x00:0x40].decode('ascii').rstrip('\x00')
    source_path = data[0x40:0x80].decode('ascii').rstrip('\x00')

    # Parse format markers
    count_marker = struct.unpack('>I', data[0x148:0x14C])[0]  # big-endian
    format_marker = struct.unpack('<I', data[0x14C:0x150])[0]

    # Parse metadata section (0x0950-0x0980)
    # Contains sector/finish line information
    metadata = data[0x950:0x980]

    # Parse coordinates
    # 0x0980-0x0988: Special marker (6342, longitude_ref)
    # 0x0988 onwards: Coordinate pairs (lat*1e7, lon*1e7)
    coordinates = []
    offset = 0x0988  # Start of actual coordinate data

    while offset < len(data) - 8:
        lat_raw, lon_raw = struct.unpack('<II', data[offset:offset + 8])

        # Check for end of data (both zero or invalid values)
        if lat_raw == 0 and lon_raw == 0:
            break

        # Check for valid coordinate range (roughly Georgia/Rustavi area)
        # Lat: 40-45 * 1e7 = 400000000-450000000
        # Lon: 40-50 * 1e7 = 400000000-500000000
        if not (400000000 < lat_raw < 500000000 and 400000000 < lon_raw < 500000000):
            break

        lat = lat_raw / 1e7
        lon = lon_raw / 1e7
        coordinates.append((lon, lat))
        offset += 8

    return VN1Venue(
        name=name,
        source_path=source_path,
        coordinates=coordinates,
        metadata=metadata
    )


def print_summary(venue: VN1Venue) -> None:
    """Print parsed venue information"""
    print(f"=== MoTeC VN1 Venue File ===")
    print(f"Track: {venue.name}")
    print(f"Source: {venue.source_path}")
    print(f"\nCoordinates ({len(venue.coordinates)} points):")

    for i, (lon, lat) in enumerate(venue.coordinates):
        print(f"  {i:2d}: [{lon:.10f}, {lat:.10f}]")

    if len(venue.coordinates) > 0:
        print(f"\nBounds:")
        lons = [c[0] for c in venue.coordinates]
        lats = [c[1] for c in venue.coordinates]
        print(f"  Min: [{min(lons):.10f}, {min(lats):.10f}]")
        print(f"  Max: [{max(lons):.10f}, {max(lats):.10f}]")
        print(f"  Center: [{(min(lons)+max(lons))/2:.10f}, {(min(lats)+max(lats))/2:.10f}]")


def analyze_metadata(venue: VN1Venue) -> None:
    """Analyze metadata section for sector/finish line info"""
    print(f"\n=== Metadata Analysis (0x0950-0x0980) ===")
    print(f"Raw bytes (hex): {venue.metadata.hex()}")
    print()

    # Parse metadata structure
    print("Possible interpretation:")
    print("  - May contain finish line coordinates or sector split points")
    print("  - Values 6342 (0x18c6) and 6858 (0x1aca) appear frequently")
    print("  - These could be sector markers or reference IDs")
    print()

    print("As int16 quads (more meaningful):")
    for i in range(0, len(venue.metadata), 8):
        if i + 8 <= len(venue.metadata):
            vals = struct.unpack('<hhhh', venue.metadata[i:i + 8])
            print(f"  {i:02x}: {vals}")

    print("\nAs unsigned int16 quads:")
    for i in range(0, len(venue.metadata), 8):
        if i + 8 <= len(venue.metadata):
            vals = struct.unpack('<HHHH', venue.metadata[i:i + 8])
            # Check if any could be coordinates scaled differently
            print(f"  {i:02x}: {vals}")
            # Try to detect coordinate-like patterns
            for j, v in enumerate(vals):
                if 40000 < v < 50000:  # Could be lat/lon * 1000
                    coord = v / 1000
                    print(f"     -> vals[{j}] = {v} -> {coord:.8f}")


if __name__ == '__main__':
    import sys

    vn1_file = 'track/data/RIMSportKarting/motec-venue/RIM.VN1'
    if len(sys.argv) > 1:
        vn1_file = sys.argv[1]

    venue = parse_vn1(vn1_file)
    print_summary(venue)
    analyze_metadata(venue)
