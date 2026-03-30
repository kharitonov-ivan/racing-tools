# MoTeC VN1 Codec

Encoder and decoder for MoTeC Venue (.VN1) files.

## File Format Structure

```
Offset  Size    Description
------  ----    -----------
0x00    64      Track name (null-terminated ASCII)
0x40    64      Source file path (null-terminated ASCII)
0x80    96      Reserved/padding
0x140   8       Unknown/reserved
0x148   4       Coordinate count (big-endian uint32)
0x14C   4       Format marker (0x00020002)
0x150   2048    Reserved/padding
0x950   48      Metadata (sector/finish line info)
0x980   4       Special marker (typically 6342)
0x984   4       Reference longitude (int32 * 1e7)
0x988   ~       Coordinate data (lat*1e7, lon*1e7 pairs)
```

### Coordinate Format

Coordinates are stored as signed 32-bit integers scaled by 10^7:
- Latitude: `int32 = lat * 10^7`
- Longitude: `int32 = lon * 10^7`

Example:
- lat=41.5666153 → int32=415666153
- lon=44.9501264 → int32=449501264

## Usage

### Python API

```python
from vn1_codec import decode_vn1, encode_vn1, VN1Venue

# Decode VN1 file
venue = decode_vn1('track.VN1')
print(f"Track: {venue.name}")
print(f"Coordinates: {len(venue.coordinates)}")

# Encode VN1 file
venue = VN1Venue(
    name="My Track",
    source_path="Generated",
    coordinates=[(lon1, lat1), (lon2, lat2), ...]
)
encode_vn1(venue, 'output.VN1')
```

### Command Line

#### Decode VN1 file

```bash
python3 vn1_codec.py decode track.VN1
```

#### Convert to GeoJSON

```bash
python3 convert_vn1.py vn1-to-geojson track.VN1 output.geojson
```

#### Convert from GeoJSON

```bash
python3 convert_vn1.py geojson-to-vn1 input.geojson output.VN1 \
    --name "My Track" \
    --reference reference.VN1
```

The `--reference` option copies metadata from an existing VN1 file,
ensuring maximum compatibility with MoTeC software.

#### Convert to/from CSV

```bash
# To CSV
python3 convert_vn1.py vn1-to-csv track.VN1 output.csv

# From CSV
python3 convert_vn1.py csv-to-vn1 input.csv output.VN1 \
    --name "My Track" \
    --reference reference.VN1
```

CSV format:
```csv
index,latitude,longitude
0,41.5666153000,44.9501264000
1,41.5668062000,44.9493540000
...
```

## Metadata Section

The metadata section (0x950-0x980, 48 bytes) contains information about:
- Sector split points
- Finish line coordinates
- Track-specific markers

Common patterns observed:
- `6342` (0x18c6) - appears frequently, likely sector marker
- `6858` (0x1aca) - appears frequently, likely reference ID

This section is not fully understood. When creating new VN1 files,
use the `--reference` option to copy metadata from an existing file.

## Installation

No external dependencies required - uses Python standard library only.

Tested with Python 3.8+.

## Roundtrip Testing

The codec preserves coordinate data accurately through decode/encode cycles:

```python
# Decode and re-encode
venue = decode_vn1('original.VN1')
encode_vn1(venue, 'roundtrip.VN1')

# Compare
original = decode_vn1('original.VN1')
roundtrip = decode_vn1('roundtrip.VN1')

assert len(original.coordinates) == len(roundtrip.coordinates)
for o, r in zip(original.coordinates, roundtrip.coordinates):
    assert abs(o[0] - r[0]) < 1e-10  # Longitude
    assert abs(o[1] - r[1]) < 1e-10  # Latitude
```

## Limitations

1. **Metadata encoding**: The metadata section format is not fully understood.
   When encoding, use `--reference` to copy from an existing file.

2. **Coordinate count**: The count field at 0x148 may not match actual
   coordinate count in some files.

3. **Binary compatibility**: Roundtrip encoding preserves coordinate data
   but may produce slightly different binary files due to metadata handling.

## Files

- `vn1_codec.py` - Core codec (decoder/encoder)
- `convert_vn1.py` - Conversion utilities (CLI)
- `parse_vn1.py` - Legacy parser (use vn1_codec.py instead)

## License

Part of racing-tools project.
