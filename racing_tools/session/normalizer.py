from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_mapping() -> dict:
    path = ROOT / "channel_mapping.json"
    return json.loads(path.read_text())


class ChannelNormalizer:
    def __init__(self, mapping_file: str | None = None, device_type: str | None = None):
        config = json.loads(Path(mapping_file).read_text()) if mapping_file else load_mapping()
        self.alias = {}
        for item in config["standard_channels"].values():
            name = item["standard_name"]
            unit = item["unit"]
            for raw in item["aliases"]:
                self.alias[raw.lower()] = (name, unit)

        self.transforms = config.get("transformations", {}).get(device_type or "", {})

    def apply_transformations(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.transforms:
            return frame
        data = frame.copy()
        for key, spec in self.transforms.items():
            if key not in data.columns:
                continue
            series = pd.to_numeric(data[key], errors="coerce")
            scale = spec.get("scale", 1.0)
            offset = spec.get("offset", 0.0)
            if scale != 1.0:
                series = series * scale
            if offset:
                series = series + offset
            if spec.get("auto_center"):
                series = series - series.mean()
            data[key] = series
        return data

    def normalize_dataframe(
        self,
        frame: pd.DataFrame,
        add_units_row: bool = False,
        apply_transforms: bool = True,
    ):
        data = self.apply_transformations(frame) if apply_transforms else frame.copy()
        names: list[str] = []
        units: dict[str, str] = {}
        counts: dict[str, int] = {}

        for col in data.columns:
            base, unit = self.alias.get(col.lower(), (col, ""))
            counts[base] = counts.get(base, 0) + 1
            final = base if counts[base] == 1 else f"{base}_{counts[base]}"
            names.append(final)
            units[final] = unit

        normalized = data.copy()
        normalized.columns = names
        return (normalized, units) if add_units_row else normalized
