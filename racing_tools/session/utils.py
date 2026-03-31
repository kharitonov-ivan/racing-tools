from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

DATE_TOKEN = re.compile(r"\d{6}")
TIME_TOKEN = re.compile(r"\d{2}H\d{2}")


def name_tokens(path: Path) -> list[str]:
    return [p for p in re.split(r"[-_\s]+", path.name) if p]


def decode_compact_date(token: str) -> str:
    if not DATE_TOKEN.fullmatch(token or ""):
        return ""
    day = int(token[:2])
    month = int(token[2:4])
    year = int(token[4:])
    year += 2000 if year < 70 else 1900
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return ""


def decode_time_token(token: str) -> str:
    if not TIME_TOKEN.fullmatch(token or ""):
        return ""
    return f"{token[:2]}:{token[-2:]}"


def decode_utc_clock(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 4:
        return ""
    digits = digits.rjust(6, "0")[:6]
    return f"{digits[:2]}:{digits[2:4]}:{digits[4:]}"


def infer_datetime_from_tokens(tokens: list[str]) -> tuple[str, str]:
    date = ""
    time = ""
    for token in tokens:
        if not date:
            date = decode_compact_date(token)
        if not time:
            time = decode_time_token(token)
    return date, time


def infer_datetime_from_path(path: Path) -> tuple[str, str]:
    return infer_datetime_from_tokens(name_tokens(path))


_SESSION_DIR = Path(__file__).resolve().parent
_MOTEC_SCRIPT = _SESSION_DIR.parent / "third_party" / "MotecLogGenerator" / "motec_log_generator.py"


def motec_script() -> Path:
    if _MOTEC_SCRIPT.is_file():
        return _MOTEC_SCRIPT
    raise FileNotFoundError(
        "motec_log_generator.py not found. "
        "Did you clone with --recurse-submodules? "
        "Run: git submodule update --init --recursive"
    )


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> tuple[bool, float, int]:
    """Return (intersects, t_along_p1p2, crossing_sign).

    crossing_sign is +1 or -1 indicating which direction the segment p1->p2
    crosses the line q1->q2. Use this to filter GPS jitter: only accept
    crossings whose sign matches the first crossing (i.e. forward direction).
    """
    px, py = p1
    rx, ry = p2[0] - px, p2[1] - py
    qx, qy = q1
    sx, sy = q2[0] - qx, q2[1] - qy

    cross = rx * sy - ry * sx
    if abs(cross) < 1e-9:
        return False, 0.0, 0

    t = ((qx - px) * sy - (qy - py) * sx) / cross
    u = ((qx - px) * ry - (qy - py) * rx) / cross

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, t, 1 if cross > 0 else -1
    return False, 0.0, 0
