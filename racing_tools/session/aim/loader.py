from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from racing_tools.session.aim.utils import DISTANCE_KEYS, SPEED_KEYS, datetime_from_meta, frame
from racing_tools.session.distance import ensure_distance
from racing_tools.session.normalizer import ChannelNormalizer
from racing_tools.session.utils import infer_datetime_from_path, name_tokens

import subprocess

ROOT = Path(__file__).resolve().parent
TDA_DIR = ROOT.parent.parent / "third_party" / "TrackDataAnalysis"


def _ensure_tda_compiled() -> None:
    """Build TrackDataAnalysis Cython extension if not already compiled."""
    if not TDA_DIR.is_dir():
        return
    data_dir = TDA_DIR / "data"
    so_files = list(data_dir.glob("aim_xrk*.so")) + list(data_dir.glob("aim_xrk*.pyd"))
    if so_files:
        return
    pyx = data_dir / "aim_xrk.pyx"
    if not pyx.exists():
        return
    print("[TDA] Compiling aim_xrk Cython extension (first run)...")
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=TDA_DIR,
        check=True,
    )
    print("[TDA] Compilation done.")



# TODO: replace xrk DLL + TDA Cython with https://github.com/bmc-labs/xdrk (Rust, Apache-2.0)
#   - native XRK parser in Rust, works on Linux and Windows (pre-compiled binaries)
#   - can be wrapped via PyO3/maturin for a single cross-platform Python extension
#   - eliminates both the Windows-only DLL and the Cython compilation step
def load_raw(path: Path, normalize: bool = True) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")

    tags = {}
    table = None
    event_date = ""
    event_time = ""
    driver = ""
    venue = ""
    vehicle = ""

    xrk_dir = ROOT.parent.parent / "third_party" / "xrk"
    xrk_dll = xrk_dir / "MatLabXRK-2017-64-ReleaseU.dll"
    assert xrk_dll.exists() or not xrk_dir.exists(), (
        f"XRK DLL not found at {xrk_dll}. "
        "Did you clone with --recurse-submodules? "
        "Run: git submodule update --init --recursive"
    )

    try:
        from racing_tools.third_party.xrk import xrk

        data = xrk.XRK(str(path))

        tags = {
            "championship": data.championship_name,
            "track": data.track_name,
            "vehicle": data.vehicle_name,
            "racer": data.racer_name,
            "venue_type": data.venue_type,
            "datetime": data.datetime,
            "lapcount": data.lapcount,
        }

        if data.datetime:
            try:
                dt = datetime.strptime(data.datetime, "%Y-%m-%d %H:%M:%S")
                event_date = dt.date().isoformat()
                event_time = dt.strftime("%H:%M")
            except ValueError:
                pass

        driver = data.racer_name
        venue = data.track_name
        vehicle = data.vehicle_name

        series_list = []
        for name, channel in data.channels.items():
            try:
                times, values = channel.samples(xtime=True, xabsolute=True)
            except Exception as e:
                print(f"Warning: failed to load channel {name}: {e}")
                continue

            if not times:
                continue

            s = pd.Series(values, index=pd.Index(times, name="Time"), name=name)
            if s.index.duplicated().any():
                s = s[~s.index.duplicated(keep="first")]
            series_list.append(s)

        if not series_list:
            raise ValueError(f"No valid channels found in {path}")

        table = pd.concat(series_list, axis=1).sort_index()

    except (ImportError, OSError, AttributeError) as e_dll:
        try:
            _ensure_tda_compiled()

            if str(TDA_DIR) not in sys.path:
                sys.path.append(str(TDA_DIR))

            from data import aim_xrk

            def progress(a, b):
                pass

            tda_log = aim_xrk.AIMXRK(str(path), progress)

            tags = tda_log.metadata.copy()
            tags["source"] = "TrackDataAnalysis"

            driver = tags.get("Driver", "")
            venue = tags.get("Venue", "")
            vehicle = tags.get("Vehicle", "")
            event_date = str(tags.get("Log Date", ""))
            event_time = str(tags.get("Log Time", ""))

            series_list = []
            for name, channel in tda_log.channels.items():
                times_ms = np.array(channel.timecodes)
                values = np.array(channel.values)

                if len(times_ms) == 0:
                    continue

                times_sec = times_ms / 1000.0

                s = pd.Series(values, index=pd.Index(times_sec, name="Time"), name=name)
                if s.index.duplicated().any():
                    s = s[~s.index.duplicated(keep="first")]
                series_list.append(s)

            if not series_list:
                raise ValueError("No channels found using TDA parser")

            table = pd.concat(series_list, axis=1).sort_index()

        except Exception as e_tda:
            raise ImportError(
                f"Failed to load XRK file.\n"
                f"  DLL loader: {e_dll}\n"
                f"  TDA parser: {e_tda}\n"
                f"The XRK DLL only works on Windows. On Linux/WSL, the TDA Cython parser is required.\n"
                f"Build it with: cd racing_tools/third_party/TrackDataAnalysis && pip install cython && python setup.py build_ext --inplace"
            )

    table = table.ffill().bfill()

    # Resample to uniform 100 Hz grid
    t = table.index
    uniform_t = np.arange(t.min(), t.max(), 0.01)
    table = table.reindex(table.index.union(uniform_t)).interpolate(method="index").reindex(uniform_t)
    table.index.name = "Time"

    table = table.reset_index()

    if normalize:
        table = ChannelNormalizer(device_type="aim").normalize_dataframe(table)

    table = ensure_distance(table, distance_keys=DISTANCE_KEYS, speed_keys=SPEED_KEYS, frequency=100.0)

    return table, {
        "driver": driver,
        "venue": venue,
        "vehicle": vehicle,
        "event_date": event_date,
        "event_time": event_time,
        "device": "AIM XRK",
        "tags": tags,
    }


def load_csv(path_or_folder: Path, frequency: float = 20.0, normalize: bool = True) -> tuple[pd.DataFrame, dict]:
    path = Path(path_or_folder)
    csv_path = path if path.is_file() else path / "aim.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"AIM CSV not found at {csv_path}")

    df, meta = frame(csv_path)
    df = ensure_distance(df, distance_keys=DISTANCE_KEYS, speed_keys=SPEED_KEYS, frequency=frequency)
    if normalize:
        df = ChannelNormalizer().normalize_dataframe(df)

    driver = ""
    venue = ""
    vehicle = ""
    session = ""
    event_date = ""
    event_time = ""
    tags = {}

    if meta:
        tags.update({"aim_meta": meta})
        if meta.get("Racer"):
            driver = meta["Racer"]
        if meta.get("Vehicle"):
            vehicle = meta["Vehicle"]
        if meta.get("Session"):
            session = meta["Session"]

        event_date, event_time = datetime_from_meta(meta)

    if not event_date or not event_time:
        p = path if path.is_dir() else path.parent
        date_token, time_token = infer_datetime_from_path(p)
        event_date = event_date or date_token
        event_time = event_time or time_token

        tokens = name_tokens(p)
        if len(tokens) > 3 and not venue:
            venue = tokens[3]
        if len(tokens) > 4 and not vehicle:
            vehicle = tokens[4]
        if len(tokens) > 5 and not driver:
            driver = tokens[5]
        if tokens and not session:
            session = tokens[-1]

    return df, {
        "driver": driver,
        "venue": venue,
        "vehicle": vehicle,
        "session": session,
        "event_date": event_date,
        "event_time": event_time,
        "device": "AIM",
        "tags": tags,
    }
