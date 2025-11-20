#!/usr/bin/env python3
"""Batch converter helper used by process_all_telemetry.sh."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

MAX_CLEAN_DEPTH = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert all telemetry sessions under a folder")
    parser.add_argument(
        "directory",
        nargs="?",
        help="Telemetry root (default: <repo>/data)",
    )
    return parser.parse_args()


def normalize_data_path(arg: str | None, root: Path) -> Path:
    if not arg:
        return (root / "data").resolve()
    raw = arg[1:] if arg.startswith("@") else arg
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def python_runner(root: Path) -> list[str]:
    py = root / ".venv" / "bin" / "python"
    if py.exists() and os.access(py, os.X_OK):
        return [str(py)]
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv missing")
    return [uv, "run", "python"]


def iter_generated_files(data_root: Path, max_depth: int) -> Iterable[Path]:
    base_depth = len(data_root.parts)
    for current, dirs, files in os.walk(data_root):
        current_path = Path(current)
        depth = len(current_path.parts) - base_depth
        if depth >= max_depth:
            dirs[:] = []
        for name in files:
            if name.lower().endswith((".ld", ".ldx")):
                yield current_path / name


def clean_outputs(data_root: Path, max_depth: int = MAX_CLEAN_DEPTH) -> None:
    for file_path in iter_generated_files(data_root, max_depth):
        try:
            file_path.unlink()
        except FileNotFoundError:
            continue
        except PermissionError:
            print(f"skip locked: {file_path}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_root = normalize_data_path(args.directory, root)
    if not data_root.is_dir():
        print(f"missing: {data_root}", file=sys.stderr)
        raise SystemExit(1)

    clean_outputs(data_root)
    runner = python_runner(root)
    cmd = [*runner, "converter/convert.py", "batch", str(data_root)]
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
