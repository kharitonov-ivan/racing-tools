import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from racing_tools.session.session import Session
from racing_web.analysis_server import build_analysis_payload, build_session_info

TRACK_PATH = "racing_tools/track/data/RIMSportKarting"
SAMPLE_ZIP = "experiments/alfano-log-zip-format/data/ALFANO7_LAP_SN1061_170326_16H32_SG__P__A_13_6309.zip"


def test_load_alfano_zip_keeps_integer_laps_and_realistic_distance() -> None:
    session = Session.load(Path(SAMPLE_ZIP))

    lap_numbers = pd.to_numeric(session.table["Lap Number"], errors="coerce").dropna()
    assert not lap_numbers.empty
    assert lap_numbers.mod(1).eq(0).all()
    assert lap_numbers.astype(int).unique().tolist() == list(range(1, 15))

    distance = pd.to_numeric(session.table["Distance"], errors="coerce")
    assert distance.max() < 20_000.0


def test_build_session_info_returns_unique_lap_choices_for_alfano_zip() -> None:
    info = build_session_info({"trackPath": TRACK_PATH, "sessionPath": SAMPLE_ZIP})

    lap_ids = [lap["id"] for lap in info["laps"]]
    assert lap_ids == list(range(1, 15))
    assert len(lap_ids) == len(set(lap_ids))
    assert info["bestLapId"] == 11
    assert info["bestTime"] == info["laps"][10]["time"]


def test_build_analysis_payload_returns_realistic_lap_geometry_for_alfano_zip() -> None:
    payload = build_analysis_payload(
        {
            "trackPath": TRACK_PATH,
            "sessionEntries": [{"path": SAMPLE_ZIP, "lapId": "best"}],
        }
    )

    lap = payload["laps"][0]
    sectors = lap["sectors"]

    assert 1_000.0 < lap["total"] < 2_000.0
    assert 60.0 < lap["lapSeconds"] < 70.0
    assert len(lap["points"]) == 721
    assert sectors[0] == 0.0
    assert sectors[-1] == lap["total"]
    assert all(current < following for current, following in zip(sectors, sectors[1:]))
    assert all(
        current["distance"] <= following["distance"]
        for current, following in zip(lap["points"], lap["points"][1:])
    )
