# Converter Guide

## Layout
- `session.py` provides `Session`, `ChannelNormalizer`, distance helpers, and metadata parsing; every converter builds a pandas table plus driver/venue/vehicle/session/device info and lets `Session.to_motec` call the vendored generator.
- `convert.py` is the single CLI with subcommands: `aim DIR`, `alfano DIR`, `alfano-excel DIR`, `batch DIR`, and `mapping`.

## Commands
```bash
uv run python converter/convert.py aim ~/logs/2025-02-01-mykart-session01
uv run python converter/convert.py alfano ~/logs/ALFANO6_LAP_SN999_SESSION
uv run python converter/convert.py alfano-excel ~/logs/ALFANO_EXCEL_EXPORT
uv run python converter/convert.py batch ~/logs
uv run python converter/convert.py mapping
```
Flags shared by the first three modes: `--frequency`, `--output`, `--raw` (keep vendor names/units, skip transforms), `--keep` (leave the `tmp_*.csv`). The batch command auto-detects format per subfolder; it removes `tmp_*.csv` unless they are locked, and `process_all_telemetry.sh` simply clears `.ld/.ldx` then calls `convert.py batch`.

## Metadata
- AIM converters read the CSV header block to capture driver/vehicle/session names plus date/time; these overwrite folder-derived tokens when present.
- Alfano CSV/Excel converters pull date/time from file/folder tokens such as `301025` and `17H02`; Excel exports also read the `UTC time : HHMMSS` field to populate `Session.event_time`.
- `Session.event_date` / `Session.event_time` travel with the pandas table so downstream tools can label logs without guessing.

## Acceleration reference
| Type | Channels | Source | Best for |
| --- | --- | --- | --- |
| GPS | `GPS_AccelLateral`, `GPS_AccelLongitudinal` (`Gf. Y/X`) | Differentiated GPS | Lap deltas, macro grip |
| IMU raw | `IMU_Accel*`, `InlineAcc`, `LateralAcc`, `VerticalAcc` | On-board accelerometer | Driver inputs, kerbs |
| IMU smoothed | `Accel*_Smoothed`, `G Lat/Long Smth` | Low-pass IMU | Clean overlays |
| IMU filtered | `IMU_Accel*_Filtered`, `G Lat/Long imu f` | Heavy filter | Presentation traces |

## Device transforms
| Channel | Operation |
| --- | --- |
| `Speed GPS`, `Speed rear` | value × 0.1 |
| `Altitude` | value × 0.1 |
| `Lat.`, `Lon.` | value × 0.000001 |
| `Gf. X`, `Gf. Y` | value × 0.01 ± offsets, auto-center |
| `Orientation` | value × 0.01 |

The table above covers Alfano6 integer encodings; the rules live in `channel_mapping.json["transformations"]["alfano"]` and run automatically unless you pass `--raw`. AIM data already ships in SI units, so only channel names change.

## Debug flow
1. Run the desired command with `--keep`.
2. Inspect the `tmp_*.csv` saved next to the session.
3. If names or units look off, edit `channel_mapping.json`.
4. Re-run and open the new `.ld` in MoTeC i2 Pro.
