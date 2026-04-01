# Alfano7 Raw ZIP Data Format

Reverse-engineered from Alfano7 exports (Windows ADA CLASSIQUE v5.2.3).
Verified against Excel 100Hz export with test scripts in this directory.

## ZIP Contents

| File | Description |
|------|-------------|
| `LAP_N_*.csv` | Per-lap telemetry at 10Hz, comma-delimited |
| `SN*_*.csv` | Session summary: lap times (ms), min/max per signal |
| `projection_orthogonale2023.csv` | Start/finish line definition |
| `temps_moteur.csv` | Engine temperature |
| `A1210.csv` | Voltage calibration (8 entries) |
| `info software.txt` | "Windows ADA CLASSIQUE" + version |
| `EEPROM_PAGE_*.dat` | Binary device config (128 bytes each) |
| `PARAMETRE*.dat` | Binary: track name, driver, serial (~306 bytes) |

## LAP CSV — Base 10Hz Signals

Each row = 0.1s. No explicit Time column (implicit from row index).

| Column | Raw unit | Conversion | Physical |
|--------|----------|------------|----------|
| Partiel | int | — | Lap segment |
| RPM | int | ×1 | RPM (see note below) |
| Speed GPS | int | ÷10 | km/h |
| Orientation | int | ÷100 | degrees |
| Speed rear | int | ÷10 | km/h (SN3476 only) |
| Lat. | int | ÷1e6 | degrees |
| Lon. | int | ÷1e6 | degrees |
| Altitude | int | ÷10 | meters |
| Gf. X | int | (raw−1000)÷100 | G lateral |
| Gf. Y | int | (raw−1000)÷100 | G longitudinal |

## 25Hz GPS Columns — Intermediate Samples (~20Hz effective)

**Key finding:** The "25Hz" value in row N is a measurement taken **between rows N-1 and N** (midpoint of the 0.1s interval). Combined with the 10Hz base, this gives ~20Hz.

| Column | Encoding | Reconstruction |
|--------|----------|----------------|
| Speed GPS 25Hz | Direct value (÷10 → km/h) | Use as-is, place at midpoint between rows |
| Lat. 25Hz | Signed 16-bit delta (microdegrees) | `position = row_Lat + delta` (in raw units, then ÷1e6) |
| Lon. 25Hz | Signed 16-bit delta (microdegrees) | `position = row_Lon + delta` (in raw units, then ÷1e6) |

Signed 16-bit: values >32767 → subtract 65536 (e.g., 65535 = −1).

**Verified:** Speed reconstruction error = 0.000 km/h vs Excel. Position error < 0.000005°.

## RPM Sub-Channels — 50Hz (device-dependent)

Present on SN1061, absent on SN3476. Five additional columns per row:

```
RPM 1 20Hz, RPM 2 50Hz, RPM 3 50Hz, RPM 4 50Hz, RPM 5 50Hz
```

**Key finding:** Sub-channels in row N are 5 measurements taken **between rows N-1 and N**, evenly spaced at 0.02s intervals:

| Offset from prev row | Column (from row N) |
|----------------------|---------------------|
| +0.02s | RPM 1 20Hz |
| +0.04s | RPM 2 50Hz |
| +0.06s | RPM 3 50Hz |
| +0.08s | RPM 4 50Hz |
| +0.10s | RPM 5 50Hz |

**Verified:** Every sub-channel value matches the Excel 100Hz output exactly at corresponding timestamps. Excel linearly interpolates at odd 0.01s steps between adjacent sub-channel anchors.

**Main RPM column** is a separate 10Hz measurement (not equal to RPM5, not an average of RPM1-5). Difference ~±30 RPM typically. Excel ignores main RPM when sub-channels are present.

## Excel CSV Format (100Hz)

Semicolon-delimited, European number format (comma = thousands separator for RPM/Orientation).
All laps in one file, `Time` column resets per lap.

**Interpolation varies by signal:**
- **RPM:** Uses sub-channel anchors at 0.02s + linear interpolation at 0.01s steps
- **Speed, G-forces, Orientation:** Linear interpolation between 10Hz samples
- **Lat/Lon:** Sample-and-hold (staircase) from GPS fixes, NOT smooth interpolation

## Summary CSV

Line 1: session metadata (date, time, serial, track, driver).
Line 2+: per-lap stats — `time lap` in milliseconds, min/max per signal.

Frequency verification: `data_rows / (time_lap_ms / 1000)` consistently gives 10.0x Hz.

## Device Differences

| Feature | SN1061 | SN3476 |
|---------|--------|--------|
| RPM sub-channels | Yes (50Hz) | No |
| Speed rear | No | Yes |
| GPS 25Hz columns | Yes | Yes |

## Effective Signal Rates

| Signal | Raw ZIP | With 25Hz/sub-channels | Excel |
|--------|---------|----------------------|-------|
| RPM | 10Hz | **50Hz** (sub-channels) | 100Hz (interpolated) |
| Speed GPS | 10Hz | **~20Hz** (+25Hz column) | 100Hz (interpolated) |
| Lat/Lon | 10Hz | **~20Hz** (+delta columns) | 100Hz (sample-and-hold) |
| Gf. X/Y | 10Hz | 10Hz | 100Hz (interpolated) |
| Orientation | 10Hz | 10Hz | 100Hz (interpolated) |
