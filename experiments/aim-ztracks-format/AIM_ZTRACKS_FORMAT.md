# AIM RaceStudio3 `.ztracks` Track Format

Reverse-engineered from `RIMSportKarting.ztracks` exported by RS3.

## Container

ZIP archive containing a single `.tkk` binary file.
The TKK filename is a random 8-char ID (e.g. `gheqlbb1.tkk`).

## TKK Chunk Format

The file is a sequence of chunks. Each chunk has:
- **Header**: `<h` + 4-byte ASCII tag + LE uint32 size + zero-padding + `>`
- **Data**: `size` bytes of payload
- **Footer**: `<` + 4-byte ASCII tag + 2 bytes (footer data) + `>`

Header total: variable (scan for `>` after size field).
Footer total: always 8 bytes.

## Chunks

### Ptkk — File Header (268 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 256 | string | Track name (null-padded, can be empty) |
| 256 | 12 | bytes | File ID (8-char ASCII) + padding |

### Vnfo — Venue Info (476 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 24 | string | Venue short name (null-padded) |
| 24 | 4 | bytes | Padding (zeros) |
| 28 | 2 | ASCII | Country code (ISO 3166-1 alpha-2, e.g. "GE") |
| 30 | 6 | bytes | Padding |
| 36 | 8 | uint64 | Unknown (possibly timestamp, observed non-zero) |
| 44 | 4 | float32 | Track length (meters) |
| 48 | 4 | int32 | Center latitude (× 1e7) |
| 52 | 4 | int32 | Center longitude (× 1e7) |
| 56 | 4 | bytes | Padding |
| 60 | 4 | uint32 | Unknown (observed: 50) |
| 64 | 8 | bytes | Padding |
| 72 | 4 | uint32 | Sector count (NOT counting SF, e.g. 2 for S1+S2) |
| 76 | 4 | uint32 | Flags (observed: 0x00040000) |
| 80 | 288 | bytes | Reserved (zeros) |
| 368 | 16 | sector | SF — Start/Finish point |
| 384 | 16 | sector | S1 — Sector 1 split point |
| 400 | 16 | sector | S2 — Sector 2 split point |
| 416 | 60 | bytes | Remaining (mostly zeros) |

**Sector point format** (16 bytes):
| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | int32 | Latitude (× 1e7) |
| 4 | 4 | int32 | Longitude (× 1e7) |
| 8 | 8 | bytes | Padding (zeros) |

Sector points are single GPS coordinates (not lines). They represent the point
where the bestline crosses the sector boundary.

**Note:** The SF (Start/Finish) position is primarily determined by the first
point of the pts array. The Vnfo SF field is optional — it may be all zeros
(observed in car track exports) while RS3 still correctly identifies SF from pts[0].

### V_sw — Full Venue Name (256 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 256 | string | Full venue/track name (null-padded) |

### Vidx — Index (8 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 8 | bytes | Index data (observed: all zeros) |

### pts — Track Points (N × 12 bytes)

Array of GPS coordinates forming the bestline/racing line.

**Point format** (12 bytes):
| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | int32 | Latitude (× 1e7) |
| 4 | 4 | int32 | Longitude (× 1e7) |
| 8 | 4 | int32 | Altitude in millimeters (e.g. 398000 = 398m ASL) |

The point array forms a closed loop (first point ≈ last point).
Observed: 231 points for a ~1300m karting track.

### zots — Timezone (408 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 408 | string | Timezone info, null-padded. Format observed: `Asia/Tbilisi\0(UTC+04:00) Tbilisi\0@8\0Georgia Time\0Georgia Standard Time\0Georgia Daylight Time\0` |

### srfs — Surface Count (4 bytes)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | uint32 | Number of surfaces (observed: 1) |

### lgo — Logo Image (variable)

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | var | string | Filename (null-terminated, e.g. `gheqlbb1.20260403_152605.logo.jpg`) |
| var | var | bytes | JPEG image data |

### plus — Venue Details XML (variable)

XML document with venue contact information:

```xml
<?xml version="1.0" encoding="utf-8"?>
<DplRoot>
  <a>
    <p n="Cty">Rustavi</p>
    <p n="Adr">21 Rustavi Racing Track, Rustavi</p>
    <p n="Pco">3700</p>
    <p n="Tel">+995551901515</p>
    <p n="Url">https://rim.ge/karting-3/</p>
  </a>
</DplRoot>
```

## Coordinate Encoding

All GPS coordinates are stored as **signed 32-bit integers** representing
degrees multiplied by 10,000,000 (1e7).

Example: latitude 41.5663018° → int32 value 415663018

## Notes

- File ID (`gheqlbb1`) appears in Ptkk, lgo filename, and possibly elsewhere
- The chunk footer 2-byte field may be a checksum or size indicator — needs more samples to confirm
- Maximum observed sector count: 2 (S1, S2), with SF always present
- The track points (pts) appear to be the bestline/racing line, not track boundaries
