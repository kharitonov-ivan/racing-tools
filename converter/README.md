Repository for light telemetry converters.

## Layout
- `converter/` hosts `session.py` (shared pandas+metadata helpers) and `convert.py` (single CLI with subcommands).
- `Session` instances now include `event_date` and `event_time`, filled automatically from AIM headers or Alfano file names/UTC stamps.
- `process_all_telemetry.sh` removes stale `.ld/.ldx` when possible, then runs the batch converter.
- `third_party/MotecLogGenerator` is vendor code; do not edit it.

## Convert one folder
```bash
uv run python converter/convert.py aim ~/logs/2025-02-01-mykart-session01
uv run python converter/convert.py alfano ~/logs/ALFANO6_LAP_SN999_SESSION
uv run python converter/convert.py alfano-excel ~/logs/ALFANO_EXCEL_EXPORT
```
- Shared flags: `--frequency`, `--output`, `--raw` (keep vendor names/units), `--keep` (leave `tmp_*.csv`).

## Convert many folders
```bash
bash process_all_telemetry.sh @/mnt/data
# or
uv run python converter/convert.py batch /mnt/data
```
`process_all_telemetry.sh` logs any locked files it cannot remove and continues.

## Development
1. `uv sync`
2. edit anything outside `third_party/`
3. run the converter you touched and open the emitted `.ld` in i2 Pro
4. keep docs terse; tables beat paragraphs

## Coding style & metadata
- Target Python 3.12, PEP 8 spacing, f-strings, vectorized pandas ops.
- Always wrap datasets in `Session`; assign driver/venue/vehicle/session/device there.
- Update `channel_mapping.json` when adding aliases or unit tweaks.

## Testing
- Use real telemetry, run with `--keep`, inspect `tmp_*.csv`, then verify the `.ld` in MoTeC i2 Pro.
- When adjusting mappings, mention the sample you exercised in your PR.

## Commits & PRs
- One topic per commit, imperative tense (`add session core`), reference the dataset tested.
- PRs should list commands run, highlight edits to mappings/transforms, and attach MoTeC screenshots when channel layouts change.

## Data hygiene
- Treat `data/` as scratch and avoid committing customer logs or DBC files.
- The batch script deletes `.ld/.ldx`; ensure MoTeC is closed or files are writable before rerunning.
