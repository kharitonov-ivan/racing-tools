"""Predictive lap time model for telemetry analysis."""

import numpy as np


class PredictiveLapModel:
    """Model for predicting lap times based on distance-time mapping."""

    def __init__(self, distance_time_map: list[tuple[float, float]]) -> None:
        data = np.array(distance_time_map)
        order = np.argsort(data[:, 0])
        self.dists = data[order, 0]
        self.times = data[order, 1]

        unique_indices = np.unique(self.dists, return_index=True)[1]
        self.dists = self.dists[unique_indices]
        self.times = self.times[unique_indices]

    def get_time(self, distance: float) -> float:
        return float(np.interp(distance, self.dists, self.times))
