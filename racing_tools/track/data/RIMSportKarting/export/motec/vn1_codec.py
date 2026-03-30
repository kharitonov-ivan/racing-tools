#!/usr/bin/env python3
"""
MoTeC Venue (.VN1) Codec - Encoder and Decoder

Format structure:
- 0x00-0x40 (64 bytes): Track name (padded with nulls)
- 0x40-0x80 (64 bytes): Source file path (padded with nulls)
- 0x80-0x140 (96 bytes): Reserved/padding
- 0x140-0x148 (8 bytes): Unknown/reserved
- 0x148-0x14C (4 bytes): Coordinate count (big-endian)
- 0x14C-0x150 (4 bytes): Format marker 0x00020002
- 0x150-0x950 (2048 bytes): Reserved/padding
- 0x950-0x980 (48 bytes): Metadata (sector/finish info)
- 0x980-0x988 (8 bytes): Special marker + reference longitude
- 0x988-0x9D0+: Coordinate data (lat*1e7, lon*1e7 as int32 pairs)

Coordinates: lat/lon stored as signed 32-bit integers scaled by 10^7
"""

import struct
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VN1Metadata:
    """Metadata section (0x950-0x980) - contains sector/finish line info"""
    raw_bytes: bytes
    # Common patterns observed:
    # - 6342 (0x18c6) appears frequently - sector marker?
    # - 6858 (0x1aca) appears frequently - reference ID?


@dataclass
class VN1Venue:
    """Complete MoTeC venue file structure"""
    name: str
    source_path: str
    coordinates: List[Tuple[float, float]]  # (lon, lat) in decimal degrees
    metadata: Optional[VN1Metadata] = None
    coord_count: Optional[int] = None  # Stored at 0x148


class VN1Decoder:
    """Decode MoTeC VN1 venue files"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.data = self._read_file()

    def _read_file(self) -> bytes:
        """Read VN1 file"""
        with open(self.filepath, 'rb') as f:
            return f.read()

    def decode(self) -> VN1Venue:
        """Decode VN1 file to VN1Venue object"""
        if len(self.data) < 0x988:
            raise ValueError(f"File too small: {len(self.data)} bytes")

        # Parse header
        name = self._parse_string(0x00, 64)
        source_path = self._parse_string(0x40, 64)

        # Parse coordinate count
        coord_count = struct.unpack('>I', self.data[0x148:0x14C])[0]

        # Parse format marker
        format_marker = struct.unpack('<I', self.data[0x14C:0x150])[0]
        if format_marker != 0x00020002:
            print(f"Warning: Unexpected format marker 0x{format_marker:08x}")

        # Parse metadata
        metadata_bytes = self.data[0x950:0x980]
        metadata = VN1Metadata(raw_bytes=metadata_bytes)

        # Parse special marker section
        special_marker = struct.unpack('<I', self.data[0x980:0x984])[0]
        ref_lon_raw = struct.unpack('<I', self.data[0x984:0x988])[0]
        ref_lon = ref_lon_raw / 1e7 if ref_lon_raw > 0 else None

        # Parse coordinates
        coordinates = self._parse_coordinates()

        return VN1Venue(
            name=name,
            source_path=source_path,
            coordinates=coordinates,
            metadata=metadata,
            coord_count=coord_count
        )

    def _parse_string(self, offset: int, max_len: int) -> str:
        """Parse null-terminated string"""
        raw = self.data[offset:offset + max_len]
        null_idx = raw.find(b'\x00')
        if null_idx >= 0:
            raw = raw[:null_idx]
        return raw.decode('ascii', errors='ignore').strip()

    def _parse_coordinates(self) -> List[Tuple[float, float]]:
        """Parse coordinate section starting at 0x988"""
        coordinates = []
        offset = 0x988

        while offset < len(self.data) - 8:
            lat_raw, lon_raw = struct.unpack('<II', self.data[offset:offset + 8])

            # Check for end of data
            if lat_raw == 0 and lon_raw == 0:
                break

            # Validate coordinate range (roughly Georgia/Rustavi area)
            # Lat: 40-45 * 1e7 = 400000000-450000000
            # Lon: 40-50 * 1e7 = 400000000-500000000
            if not (400000000 < lat_raw < 500000000 and 400000000 < lon_raw < 500000000):
                if offset > 0x988:  # Allow some margin
                    break
                offset += 8
                continue

            lat = lat_raw / 1e7
            lon = lon_raw / 1e7
            coordinates.append((lon, lat))
            offset += 8

        return coordinates


class VN1Encoder:
    """Encode MoTeC VN1 venue files"""

    def __init__(self):
        self.file_size = 4096  # Standard VN1 file size

    def encode(self, venue: VN1Venue, filepath: str, reference: Optional[bytes] = None) -> None:
        """
        Encode VN1Venue object to VN1 file

        Args:
            venue: VN1Venue object to encode
            filepath: Output file path
            reference: Optional reference VN1 file to copy metadata from
        """
        data = bytearray(self.file_size)

        # If reference file provided, copy its structure
        if reference is not None:
            ref_data = reference
            # Copy sections from reference
            data[0x00:0x100] = ref_data[0x00:0x100]  # Header and count
            data[0x950:0x988] = ref_data[0x950:0x988]  # Metadata and special marker
        else:
            # Write header from scratch
            self._write_string(data, 0x00, 64, venue.name)
            self._write_string(data, 0x40, 64, venue.source_path)

            # Write coordinate count (big-endian)
            coord_count = len(venue.coordinates) if venue.coord_count is None else venue.coord_count
            struct.pack_into('>I', data, 0x148, coord_count)

            # Write format marker
            struct.pack_into('<I', data, 0x14C, 0x00020002)

            # Write metadata if provided
            if venue.metadata is not None:
                data[0x950:0x980] = venue.metadata.raw_bytes

            # Write special marker (6342 seems to be standard)
            struct.pack_into('<I', data, 0x980, 6342)

            # Write reference longitude (slightly less than first coord's lon)
            # This seems to be a reference value, not exactly the first coordinate
            if venue.coordinates:
                ref_lon_raw = int(venue.coordinates[0][0] * 1e7) - 2573  # Offset observed in original
                struct.pack_into('<I', data, 0x984, ref_lon_raw)

        # Write coordinates
        self._write_coordinates(data, venue.coordinates)

        # Write terminating coordinate if present in reference
        if reference is not None:
            # Check if there's data after coordinates
            coord_end = 0x988 + len(venue.coordinates) * 8
            if coord_end + 8 <= len(ref_data):
                # Copy any remaining data
                data[coord_end:] = ref_data[coord_end:]

        # Write to file
        with open(filepath, 'wb') as f:
            f.write(data)

    def _write_string(self, data: bytearray, offset: int, max_len: int, value: str) -> None:
        """Write null-terminated string"""
        encoded = value.encode('ascii')[:max_len - 1]
        data[offset:offset + len(encoded)] = encoded
        data[offset + len(encoded)] = 0  # Null terminator

    def _write_coordinates(self, data: bytearray, coordinates: List[Tuple[float, float]]) -> None:
        """Write coordinates starting at 0x988"""
        offset = 0x988

        for lon, lat in coordinates:
            if offset + 8 > len(data):
                break

            lat_raw = int(lat * 1e7)
            lon_raw = int(lon * 1e7)

            struct.pack_into('<II', data, offset, lat_raw, lon_raw)
            offset += 8


def print_venue_info(venue: VN1Venue) -> None:
    """Print venue information"""
    print(f"=== MoTeC VN1 Venue ===")
    print(f"Name: {venue.name}")
    print(f"Source: {venue.source_path}")
    print(f"Coordinates: {len(venue.coordinates)} points")

    if venue.coordinates:
        lons = [c[0] for c in venue.coordinates]
        lats = [c[1] for c in venue.coordinates]
        print(f"  Bounds:")
        print(f"    Min: [{min(lons):.10f}, {min(lats):.10f}]")
        print(f"    Max: [{max(lons):.10f}, {max(lats):.10f}]")
        print(f"    Center: [{(min(lons)+max(lons))/2:.10f}, {(min(lats)+max(lats))/2:.10f}]")
        print(f"\n  First 3 points:")
        for i, (lon, lat) in enumerate(venue.coordinates[:3]):
            print(f"    {i}: [{lon:.10f}, {lat:.10f}]")

    if venue.metadata:
        print(f"\nMetadata: {venue.metadata.raw_bytes.hex()}")


def decode_vn1(filepath: str) -> VN1Venue:
    """Convenience function to decode VN1 file"""
    decoder = VN1Decoder(filepath)
    return decoder.decode()


def encode_vn1(venue: VN1Venue, filepath: str) -> None:
    """Convenience function to encode VN1 file"""
    encoder = VN1Encoder()
    encoder.encode(venue, filepath)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vn1_codec.py <vn1_file>")
        print("       python vn1_codec.py decode <vn1_file>")
        print("       python vn1_codec.py encode <geojson_file> <vn1_output>")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'decode':
        if len(sys.argv) < 3:
            print("Error: Please specify VN1 file to decode")
            sys.exit(1)

        vn1_file = sys.argv[2]
        venue = decode_vn1(vn1_file)
        print_venue_info(venue)

    elif command == 'encode':
        if len(sys.argv) < 4:
            print("Error: Please specify input GeoJSON and output VN1 file")
            sys.exit(1)

        # TODO: Implement GeoJSON to VN1 conversion
        print("GeoJSON to VN1 conversion - coming soon!")

    else:
        # Try to decode as VN1 file
        try:
            venue = decode_vn1(command)
            print_venue_info(venue)
        except Exception as e:
            print(f"Error decoding file: {e}")
            sys.exit(1)
