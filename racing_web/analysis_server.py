from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

import contextily as ctx
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer

from racing_tools.session.session import Session
from racing_tools.track.constants import WEBMERCATOR_CRS
from racing_tools.track.track import Track

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
COLORS = ["#ff4058", "#6aa7ff", "#8dd17e", "#ffb84d", "#c48bff", "#4dd2ff"]
CHANNELS = {
    "speed": ["Speed", "GPS Speed", "Wheel Speed", "Vitesse"],
    "throttle": ["Throttle", "PedalPosition", "TPS", "Throttle Position", "Throttle Angle"],
    "brake": ["Brake", "BrakePressure", "Brake Pressure"],
    "gear": ["Gear", "Calculated_Gear", "PreCalcGear"],
    "rpm": ["RPM", "Engine RPM", "Régime"],
    "steering": ["Steer", "SteeringAngle", "Steering", "Steering Angle"],
}
TRACK_CACHE: dict[str, Track] = {}
SATELLITE_CACHE: dict[str, dict | None] = {}
WEB_MERCATOR = Transformer.from_crs("EPSG:4326", WEBMERCATOR_CRS, always_xy=True)


class AnalysisHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/analysis/defaults":
            self._send_json({"trackPath": "racing_tools/track/data/RIMSportKarting"})
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        routes = {
            "/api/analysis/load": build_analysis_payload,
            "/api/analysis/pick": pick_local_path,
            "/api/analysis/session-info": build_session_info,
        }
        if path not in routes:
            self.send_error(404, "Unknown API endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = routes[path](payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(result)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_analysis_payload(payload: dict) -> dict:
    track_path = resolve_path(payload.get("trackPath", ""), expect_dir=True)
    entries = normalize_session_entries(payload)
    if not 1 <= len(entries) <= len(COLORS):
        raise ValueError(f"Provide between 1 and {len(COLORS)} session/lap entries.")

    track = load_track(track_path)
    raw_laps = [load_session_lap(track, entry, slot + 1, COLORS[slot]) for slot, entry in enumerate(entries)]
    laps = resample_laps(raw_laps)
    return {
        "title": track.name or track_path.name,
        "trackPath": str(track_path),
        "trackPoints": choose_track_points(track),
        "mapBackground": build_map_background(track, track_path),
        "laps": laps,
    }


def build_session_info(payload: dict) -> dict:
    session_path = resolve_path(payload.get("sessionPath", ""), expect_dir=False)
    track = maybe_load_track(payload.get("trackPath", ""))
    session = Session.load(session_path)
    session.track = track
    prepare_session_laps(session)
    best = pick_best_lap(session)
    laps = [
        {
            "id": int(stat["id"]),
            "time": format_lap_time(float(stat["time"])),
            "seconds": float(stat["time"]),
            "label": lap_option_label(stat),
        }
        for stat in available_lap_stats(session)
    ]
    return {
        "path": str(session_path),
        "driver": session.driver or session_path.stem,
        "bestLapId": best["id"],
        "bestLabel": lap_option_label(best),
        "bestTime": format_lap_time(float(best["time"])),
        "laps": laps,
    }


def normalize_session_entries(payload: dict) -> list[dict]:
    entries = payload.get("sessionEntries")
    if entries:
        normalized = []
        for entry in entries:
            path = str((entry or {}).get("path", "")).strip()
            if not path:
                continue
            normalized.append({"path": resolve_path(path, expect_dir=False), "lapId": (entry or {}).get("lapId")})
        return normalized
    session_paths = [resolve_path(path, expect_dir=False) for path in payload.get("sessionPaths", []) if str(path).strip()]
    return [{"path": path, "lapId": None} for path in session_paths]


def resolve_path(raw: str, *, expect_dir: bool) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if expect_dir and not path.is_dir():
        raise FileNotFoundError(f"Track directory not found: {path}")
    if not expect_dir and not path.exists():
        raise FileNotFoundError(f"Telemetry path not found: {path}")
    return path


def load_track(track_path: Path) -> Track:
    key = str(track_path)
    if key not in TRACK_CACHE:
        TRACK_CACHE[key] = Track.load(track_path)
    return TRACK_CACHE[key]


def maybe_load_track(raw: str) -> Track | None:
    text = str(raw).strip()
    if not text:
        return None
    try:
        path = resolve_path(text, expect_dir=True)
    except FileNotFoundError:
        return None
    return load_track(path)


def pick_local_path(payload: dict) -> dict:
    target = str(payload.get("target", "")).strip()
    mode = str(payload.get("mode", "")).strip()
    if target == "track":
        if mode != "directory":
            raise ValueError("Track picker supports directories only.")
        selected = open_directory_picker("Choose track folder")
    elif target == "session":
        if mode == "directory":
            selected = open_directory_picker("Choose telemetry session folder")
        elif mode == "file":
            selected = open_file_picker("Choose telemetry session file")
        else:
            raise ValueError("Session picker mode must be file or directory.")
    else:
        raise ValueError("Unknown picker target.")
    if not selected:
        raise ValueError("Picker was cancelled.")
    return {"path": normalize_selected_path(selected)}


def open_directory_picker(title: str) -> str:
    if powershell := shutil.which("powershell.exe"):
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f'$dialog.Description = "{title}"; '
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dialog.SelectedPath }"
        )
        return run_picker_command([powershell, "-NoProfile", "-STA", "-Command", script])
    if shutil.which("zenity"):
        return run_picker_command(["zenity", "--file-selection", "--directory", "--title", title])
    if shutil.which("osascript"):
        return run_picker_command(["osascript", "-e", f'POSIX path of (choose folder with prompt "{title}")'])
    raise RuntimeError("No supported native directory picker found. Start analysis_server.py on a desktop environment.")


def open_file_picker(title: str) -> str:
    if powershell := shutil.which("powershell.exe"):
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
            "$dialog.Filter = 'Telemetry files|*.xrk;*.xrs;*.zip;*.gpx;*.csv;*.ld;*.ldx|All files|*.*'; "
            "$dialog.Multiselect = $false; "
            f'$dialog.Title = "{title}"; '
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dialog.FileName }"
        )
        return run_picker_command([powershell, "-NoProfile", "-STA", "-Command", script])
    if shutil.which("zenity"):
        return run_picker_command([
            "zenity", "--file-selection", "--title", title,
            "--file-filter", "Telemetry | *.xrk *.xrs *.zip *.gpx *.csv *.ld *.ldx",
            "--file-filter", "All files | *",
        ])
    if shutil.which("osascript"):
        return run_picker_command(["osascript", "-e", f'POSIX path of (choose file with prompt "{title}")'])
    raise RuntimeError("No supported native file picker found. Start analysis_server.py on a desktop environment.")


def run_picker_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def normalize_selected_path(raw: str) -> str:
    path = raw.strip().strip('"')
    if not path:
        return ""
    posix = Path(path)
    if posix.exists():
        return str(posix)
    windows = PureWindowsPath(path)
    if windows.drive:
        drive = windows.drive[0].lower()
        candidate = Path(f"/mnt/{drive}")
        for part in windows.parts[1:]:
            candidate /= part
        return str(candidate)
    return path


def load_session_lap(track: Track, entry: dict, slot: int, color: str) -> dict:
    path = entry["path"]
    session = Session.load(path)
    session.track = track
    prepare_session_laps(session)
    selected = pick_requested_lap(session, entry.get("lapId"))
    lap_frame = lap_dataframe(session, selected["id"])
    if lap_frame.empty:
        raise ValueError(f"No lap samples found in {path}")

    lap_frame = lap_frame.sort_values("Time").reset_index(drop=True)
    distances = numeric_column(lap_frame, ["Distance", "Distance on GPS Speed"]).to_numpy(dtype=float)
    distances -= distances[0]
    lap_times = lap_time_series(lap_frame)

    utm = Transformer.from_crs("EPSG:4326", track.utm_zone, always_xy=True)
    lons = numeric_column(lap_frame, ["GPS Longitude", "Longitude"]).to_numpy(dtype=float)
    lats = numeric_column(lap_frame, ["GPS Latitude", "Latitude"]).to_numpy(dtype=float)
    xs, ys = utm.transform(lons, lats)
    map_xs, map_ys = WEB_MERCATOR.transform(lons, lats)
    sectors = sector_boundaries(session, selected["id"], distances, lap_times)

    return {
        "slot": slot,
        "id": int(selected["id"]),
        "label": lap_option_label(selected),
        "color": color,
        "driver": session.driver or path.stem,
        "source": str(path),
        "lapSeconds": float(selected["time"]),
        "time": format_lap_time(float(selected["time"])),
        "total": float(distances[-1]),
        "sectors": sectors,
        "arrays": {
            "distance": distances,
            "time": lap_times,
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "mapX": np.asarray(map_xs, dtype=float),
            "mapY": np.asarray(map_ys, dtype=float),
            "speed": channel_values(lap_frame, CHANNELS["speed"]),
            "throttle": channel_values(lap_frame, CHANNELS["throttle"]),
            "brake": channel_values(lap_frame, CHANNELS["brake"]),
            "gear": channel_values(lap_frame, CHANNELS["gear"]),
            "rpm": channel_values(lap_frame, CHANNELS["rpm"]),
            "steering": channel_values(lap_frame, CHANNELS["steering"]),
        },
    }


def prepare_session_laps(session: Session) -> None:
    if "Lap Number" in session.table.columns and "LapNumber" not in session.table.columns:
        session.table["LapNumber"] = pd.to_numeric(session.table["Lap Number"], errors="coerce")
    if "Lap Time" in session.table.columns and "LapTime" not in session.table.columns:
        session.table["LapTime"] = pd.to_numeric(session.table["Lap Time"], errors="coerce")

    lat_col = pick_column(session.table, ["GPS Latitude", "Latitude"])
    lon_col = pick_column(session.table, ["GPS Longitude", "Longitude"])
    if session.track and session.track.start_finish_wgs84 and lat_col and lon_col:
        session.detect_crossings()
        if len(session.crossings) >= 2:
            session.detect_sector_crossings()
            session.add_lap_numbers()

    if "LapNumber" not in session.table.columns:
        times = pd.to_numeric(session.table["Time"], errors="coerce").to_numpy(dtype=float)
        session.table["LapNumber"] = np.ones(len(session.table), dtype=int)
        session.table["LapTime"] = times - times[0]
        return

    if "LapTime" in session.table.columns:
        return

    times = pd.to_numeric(session.table["Time"], errors="coerce").to_numpy(dtype=float)
    lap_numbers = pd.to_numeric(session.table["LapNumber"], errors="coerce").fillna(1).to_numpy(dtype=int)
    lap_times = np.zeros(len(session.table), dtype=float)
    lap_starts: dict[int, float] = {}
    for index, lap_number in enumerate(lap_numbers):
        lap_starts.setdefault(int(lap_number), times[index])
        lap_times[index] = times[index] - lap_starts[int(lap_number)]
    session.table["LapTime"] = lap_times


def available_lap_stats(session: Session) -> list[dict]:
    return [
        lap for lap in session.get_lap_stats()
        if int(lap.get("id", 0)) > 0 and lap.get("time") and float(lap["time"]) > 0.0
    ]


def pick_best_lap(session: Session) -> dict:
    best = session.best_lap
    if best:
        return best
    stats = available_lap_stats(session)
    if stats:
        return min(stats, key=lambda lap: float(lap["time"]))
    lap_time = float(pd.to_numeric(session.table["LapTime"], errors="coerce").max())
    return {"id": 1, "time": lap_time}


def pick_requested_lap(session: Session, lap_id: object) -> dict:
    if lap_id in (None, "", "best"):
        return pick_best_lap(session)
    requested = int(lap_id)
    for lap in available_lap_stats(session):
        if int(lap["id"]) == requested:
            return lap
    raise ValueError(f"Lap {requested} is not available in {session.driver or 'session'}.")


def lap_option_label(lap: dict) -> str:
    label = f"Lap {int(lap['id'])}"
    if lap.get("label"):
        label = f"{label} ({lap['label']})"
    return f"{label} · {format_lap_time(float(lap['time']))}"


def lap_dataframe(session: Session, lap_number: int) -> pd.DataFrame:
    lap_col = "LapNumber" if "LapNumber" in session.table.columns else "Lap Number"
    if lap_col not in session.table.columns:
        return session.table.copy()
    return session.table[pd.to_numeric(session.table[lap_col], errors="coerce") == lap_number].copy()


def lap_time_series(frame: pd.DataFrame) -> np.ndarray:
    if "LapTime" in frame.columns:
        values = pd.to_numeric(frame["LapTime"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any() and np.nanmax(values) > 0:
            return values - values[0]
    values = pd.to_numeric(frame["Time"], errors="coerce").to_numpy(dtype=float)
    return values - values[0]


def sector_boundaries(session: Session, lap_id: int, distances: np.ndarray, lap_times: np.ndarray) -> list[float]:
    splits = session.get_sector_splits().get(lap_id, {}) if hasattr(session, "get_sector_splits") else {}
    boundaries = [0.0]
    elapsed = 0.0
    for split_time in splits.values():
        elapsed += float(split_time)
        boundaries.append(float(np.interp(elapsed, lap_times, distances)))
    boundaries.append(float(distances[-1]))
    deduped = [boundaries[0]]
    for value in boundaries[1:]:
        value = min(value, float(distances[-1]))
        if value > deduped[-1] + 1e-6:
            deduped.append(value)
    if deduped[-1] < float(distances[-1]):
        deduped.append(float(distances[-1]))
    return deduped


def channel_values(frame: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    col = pick_column(frame, candidates)
    if not col:
        return np.zeros(len(frame), dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").interpolate().ffill().bfill().to_numpy(dtype=float)


def resample_laps(raw_laps: list[dict], samples: int = 721) -> list[dict]:
    common_total = min(lap["total"] for lap in raw_laps)
    distance_grid = np.linspace(0.0, common_total, samples)
    laps = []
    for raw in raw_laps:
        arrays = raw["arrays"]
        sectors = [min(value, common_total) for value in raw["sectors"]]
        cleaned = [sectors[0]]
        for value in sectors[1:]:
            if value > cleaned[-1] + 1e-6:
                cleaned.append(value)
        if cleaned[-1] < common_total:
            cleaned.append(common_total)

        points = []
        for index, distance in enumerate(distance_grid):
            point = {
                "distance": float(common_total if index == len(distance_grid) - 1 else distance),
                "time": float(np.interp(distance, arrays["distance"], arrays["time"])),
                "x": float(np.interp(distance, arrays["distance"], arrays["x"])),
                "y": float(np.interp(distance, arrays["distance"], arrays["y"])),
                "mapX": float(np.interp(distance, arrays["distance"], arrays["mapX"])),
                "mapY": float(np.interp(distance, arrays["distance"], arrays["mapY"])),
                "speed": float(np.interp(distance, arrays["distance"], arrays["speed"])),
                "throttle": float(np.interp(distance, arrays["distance"], arrays["throttle"])),
                "brake": float(np.interp(distance, arrays["distance"], arrays["brake"])),
                "gear": int(round(np.interp(distance, arrays["distance"], arrays["gear"]))),
                "rpm": float(np.interp(distance, arrays["distance"], arrays["rpm"])),
                "steering": float(np.interp(distance, arrays["distance"], arrays["steering"])),
            }
            points.append(point)

        laps.append({
            "slot": raw["slot"],
            "id": raw["id"],
            "label": raw["label"],
            "color": raw["color"],
            "driver": raw["driver"],
            "source": raw["source"],
            "lapSeconds": raw["lapSeconds"],
            "time": raw["time"],
            "total": float(common_total),
            "sectors": cleaned,
            "points": points,
        })
    return laps


def choose_track_points(track: Track) -> list[dict[str, float]]:
    points = np.asarray(track.bestline_utm or track.centerline or [], dtype=float)
    if len(points) == 0:
        return []
    projector = Transformer.from_crs(track.utm_zone, WEBMERCATOR_CRS, always_xy=True)
    map_xs, map_ys = projector.transform(points[:, 0], points[:, 1])
    return [
        {"x": float(x), "y": float(y), "mapX": float(map_x), "mapY": float(map_y)}
        for (x, y), map_x, map_y in zip(points, map_xs, map_ys)
    ]


def build_map_background(track: Track, track_path: Path) -> dict | None:
    cache_key = str(track_path)
    if cache_key in SATELLITE_CACHE:
        return SATELLITE_CACHE[cache_key]

    points = np.asarray(track.bestline_utm or track.centerline or [], dtype=float)
    if len(points) == 0:
        SATELLITE_CACHE[cache_key] = None
        return None

    projector = Transformer.from_crs(track.utm_zone, WEBMERCATOR_CRS, always_xy=True)
    xs, ys = projector.transform(points[:, 0], points[:, 1])
    padding = 100.0
    west = float(np.min(xs) - padding)
    east = float(np.max(xs) + padding)
    south = float(np.min(ys) - padding)
    north = float(np.max(ys) + padding)

    try:
        image, extent = ctx.bounds2img(west, south, east, north, zoom="auto", source=ctx.providers.Esri.WorldImagery, ll=False, use_cache=True)
    except Exception:
        SATELLITE_CACHE[cache_key] = None
        return None

    png = io.BytesIO()
    Image.fromarray(image).save(png, format="PNG")
    payload = {
        "image": f"data:image/png;base64,{base64.b64encode(png.getvalue()).decode('ascii')}",
        "bounds": {
            "minX": float(extent[0]),
            "maxX": float(extent[1]),
            "minY": float(extent[2]),
            "maxY": float(extent[3]),
        },
    }
    SATELLITE_CACHE[cache_key] = payload
    return payload


def pick_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def numeric_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    col = pick_column(frame, candidates)
    if not col:
        raise ValueError(f"Missing required column. Tried: {', '.join(candidates)}")
    values = pd.to_numeric(frame[col], errors="coerce").interpolate().ffill().bfill()
    if values.isna().all():
        raise ValueError(f"Column {col} has no numeric data")
    return values


def format_lap_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the telemetry analysis page with local loading API")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), partial(AnalysisHandler, directory=str(WEB_DIR)))
    print(f"Serving analysis UI on http://{args.bind}:{args.port}/analysis.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
