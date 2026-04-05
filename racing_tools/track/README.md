# Track Tools

## Generate bestline from telemetry

```bash
uv run python -m racing_tools.track.generate_bestline_from_telemetry \
  --track racing_tools/track/data/RIMSportKarting \
  --session /path/to/session.zip \
  --samples 1024
```

This will:
1. Load track and telemetry session
2. Detect lap crossings, find best lap
3. Extract GPS coordinates from best lap
4. Smooth SF junction and apply light filtering
5. Export to all formats:
   - `geometry/bestline.geojson` — bestline coordinates
   - `geometry/export/bestline.gpx` — GPX with altitude
   - `geometry/export/track.kml` — KML for Google Earth
   - `geometry/export/RIMSportKarting.ztracks` — AIM RaceStudio3
   - `track_config.json` — sector distances and metadata

Options:
- `--samples N` — number of resampled points (default: 1024)
- `--smooth-radius M` — SF junction smoothing radius in meters (default: 60)

## Visualize track

```bash
uv run python -m racing_tools.track.visualize_track racing_tools/track/data/RIMSportKarting
```

Saves `track_visualization.png` with satellite imagery overlay, sector lines, and bestline.

## Track directory structure

```
data/RIMSportKarting/
  track_config.json          # metadata, sectors, UTM zone
  geometry/
    track-inner.geojson      # inner boundary
    track-outer.geojson      # outer boundary
    sectors.geojson           # SF, S1, S2 sector lines
    bestline.geojson          # racing line (generated)
    centerline.geojson        # center of track
    export/
      bestline.gpx            # GPX with altitude
      track.kml               # Google Earth
      RIMSportKarting.ztracks  # AIM RS3
```
