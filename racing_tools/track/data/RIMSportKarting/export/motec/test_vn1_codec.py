#!/usr/bin/env python3
"""
Test suite for VN1 codec
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vn1_codec import decode_vn1, VN1Venue, VN1Encoder
from convert_vn1 import vn1_to_geojson, geojson_to_vn1, vn1_to_csv


def test_decode():
    """Test decoding VN1 file"""
    print("=== Test: Decode VN1 ===")

    script_dir = Path(__file__).parent
    vn1_file = script_dir / "RIM.VN1"
    venue = decode_vn1(str(vn1_file))

    assert venue.name == "RIMSportKarting"
    assert len(venue.coordinates) == 9
    assert venue.coord_count == 438

    # Check first coordinate
    lon, lat = venue.coordinates[0]
    assert abs(lon - 44.9501264) < 1e-10
    assert abs(lat - 41.5666153) < 1e-10

    print("✓ Decode test passed")
    return venue


def test_encode(venue):
    """Test encoding VN1 file"""
    print("\n=== Test: Encode VN1 ===")

    with tempfile.NamedTemporaryFile(suffix=".VN1", delete=False) as f:
        output_file = f.name

    try:
        encoder = VN1Encoder()
        encoder.encode(venue, output_file)

        # Verify encoded file can be decoded
        venue_decoded = decode_vn1(output_file)
        assert venue_decoded.name == venue.name
        assert len(venue_decoded.coordinates) == len(venue.coordinates)

        print("✓ Encode test passed")
        return output_file
    finally:
        Path(output_file).unlink(missing_ok=True)


def test_roundtrip():
    """Test encode/decode roundtrip"""
    print("\n=== Test: Roundtrip ===")

    script_dir = Path(__file__).parent
    original_file = script_dir / "RIM.VN1"
    original = decode_vn1(str(original_file))

    with tempfile.NamedTemporaryFile(suffix=".VN1", delete=False) as f:
        roundtrip_file = f.name

    try:
        # Encode and decode
        encoder = VN1Encoder()
        encoder.encode(original, roundtrip_file)
        roundtrip = decode_vn1(roundtrip_file)

        # Compare
        assert len(original.coordinates) == len(roundtrip.coordinates)
        for (o_lon, o_lat), (r_lon, r_lat) in zip(original.coordinates, roundtrip.coordinates):
            assert abs(o_lon - r_lon) < 1e-10, f"Longitude mismatch: {o_lon} vs {r_lon}"
            assert abs(o_lat - r_lat) < 1e-10, f"Latitude mismatch: {o_lat} vs {r_lat}"

        print("✓ Roundtrip test passed")
    finally:
        Path(roundtrip_file).unlink(missing_ok=True)


def test_geojson_conversion():
    """Test VN1 to GeoJSON conversion"""
    print("\n=== Test: GeoJSON Conversion ===")

    script_dir = Path(__file__).parent
    vn1_file = script_dir / "RIM.VN1"

    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
        geojson_file = f.name

    try:
        # Convert to GeoJSON
        vn1_to_geojson(vn1_file, geojson_file)

        # Verify GeoJSON was created
        assert Path(geojson_file).exists()

        # Read and verify structure
        import json

        with open(geojson_file, "r") as f:
            gj = json.load(f)

        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) > 0
        assert gj["features"][0]["geometry"]["type"] == "LineString"

        print("✓ GeoJSON conversion test passed")
    finally:
        Path(geojson_file).unlink(missing_ok=True)


def test_csv_conversion():
    """Test VN1 to CSV conversion"""
    print("\n=== Test: CSV Conversion ===")

    script_dir = Path(__file__).parent
    vn1_file = script_dir / "RIM.VN1"

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_file = f.name

    try:
        # Convert to CSV
        vn1_to_csv(vn1_file, csv_file)

        # Verify CSV was created
        assert Path(csv_file).exists()

        # Read and verify
        with open(csv_file, "r") as f:
            lines = f.readlines()

        assert len(lines) > 1  # Header + data
        assert lines[0].strip() == "index,latitude,longitude"

        print("✓ CSV conversion test passed")
    finally:
        Path(csv_file).unlink(missing_ok=True)


def test_bounds():
    """Test coordinate bounds calculation"""
    print("\n=== Test: Coordinate Bounds ===")

    script_dir = Path(__file__).parent
    vn1_file = script_dir / "RIM.VN1"
    venue = decode_vn1(str(vn1_file))

    lons = [c[0] for c in venue.coordinates]
    lats = [c[1] for c in venue.coordinates]

    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Rustavi, Georgia area
    assert 44.9 < min_lon < 45.0
    assert 44.9 < max_lon < 45.0
    assert 41.5 < min_lat < 42.0
    assert 41.5 < max_lat < 42.0

    print(f"  Bounds: [{min_lon:.6f}, {min_lat:.6f}] to [{max_lon:.6f}, {max_lat:.6f}]")
    print("✓ Bounds test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("VN1 Codec Test Suite")
    print("=" * 60)

    try:
        venue = test_decode()
        test_encode(venue)
        test_roundtrip()
        test_geojson_conversion()
        test_csv_conversion()
        test_bounds()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
