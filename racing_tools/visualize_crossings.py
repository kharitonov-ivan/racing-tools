#!/usr/bin/env python3
"""Visualize GPS track and crossing detection."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from racing_tools.session.session import Session
from racing_tools.track.track import Track

# Load telemetry
telemetry_path = Path("../data/17-03-2026/16-51/IK_RotaxMax2_ABCD_a_1114.xrk")
session = Session.load(telemetry_path)

# Load track
track = Track.load("track/data/RIMSportKarting")

# Get GPS data
lat_col = "GPS Latitude"
lon_col = "GPS Longitude"
lats = session.table[lat_col].values
lons = session.table[lon_col].values
times = session.table["Time"].values

# Get start-finish line
sf_points = list(dict.fromkeys(track.geometry.start_finish_wgs84))
sf_p1, sf_p2 = sf_points[0], sf_points[-1]

# Extend start-finish line for visualization
sf_dx = sf_p2[0] - sf_p1[0]
sf_dy = sf_p2[1] - sf_p1[1]
sf_len = (sf_dx**2 + sf_dy**2) ** 0.5
extend_deg = 0.001  # ~100 meters
if sf_len > 1e-9:
    extend_dx = sf_dx / sf_len * extend_deg
    extend_dy = sf_dy / sf_len * extend_deg
    sf_p1_ext = (sf_p1[0] - extend_dx, sf_p1[1] - extend_dy)
    sf_p2_ext = (sf_p2[0] + extend_dx, sf_p2[1] + extend_dy)
else:
    sf_p1_ext, sf_p2_ext = sf_p1, sf_p2

# Detect crossings
session.track = track.geometry
crossings = session.detect_crossings()

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Left plot: Full track with start-finish line
ax1 = axes[0]
ax1.plot(lons, lats, 'b-', alpha=0.5, linewidth=0.5, label='GPS track')

# Plot start-finish line (original)
ax1.plot([sf_p1[0], sf_p2[0]], [sf_p1[1], sf_p2[1]], 'r-', linewidth=3, label='Start-Finish (original)')

# Plot start-finish line (extended)
ax1.plot([sf_p1_ext[0], sf_p2_ext[0]], [sf_p1_ext[1], sf_p2_ext[1]], 'g--', linewidth=2, label='Start-Finish (extended)')

# Mark crossing points
for i, t in enumerate(crossings):
    idx = np.argmin(np.abs(times - t))
    ax1.plot(lons[idx], lats[idx], 'go', markersize=8)
    ax1.annotate(f'{i+1}', (lons[idx], lats[idx]), textcoords="offset points", xytext=(5, 5), fontsize=8)

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title(f'GPS Track with {len(crossings)} Crossings Detected')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Right plot: Zoom in on start-finish area
ax2 = axes[1]

# Find GPS points near start-finish line
sf_center_lon = (sf_p1[0] + sf_p2[0]) / 2
sf_center_lat = (sf_p1[1] + sf_p2[1]) / 2
margin = 0.003  # ~300m

mask = (lons > sf_center_lon - margin) & (lons < sf_center_lon + margin) & \
       (lats > sf_center_lat - margin) & (lats < sf_center_lat + margin)

ax2.plot(lons[mask], lats[mask], 'b.', alpha=0.3, markersize=2, label='GPS points')

# Plot start-finish line (original)
ax2.plot([sf_p1[0], sf_p2[0]], [sf_p1[1], sf_p2[1]], 'r-', linewidth=3, label='Start-Finish (original)')

# Plot start-finish line (extended)
ax2.plot([sf_p1_ext[0], sf_p2_ext[0]], [sf_p1_ext[1], sf_p2_ext[1]], 'g--', linewidth=2, label='Start-Finish (extended)')

# Mark crossing points near start-finish
for i, t in enumerate(crossings):
    idx = np.argmin(np.abs(times - t))
    if mask[idx]:
        ax2.plot(lons[idx], lats[idx], 'go', markersize=10)
        ax2.annotate(f'{i+1}: {t:.1f}s', (lons[idx], lats[idx]), textcoords="offset points", xytext=(5, 5), fontsize=9)

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('Zoom on Start-Finish Line')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('crossing_detection.png', dpi=150)
print(f"\nSaved visualization to: crossing_detection.png")

# Print lap durations
print("\n" + "=" * 60)
print("LAP DURATIONS FROM GPS CROSSINGS")
print("=" * 60)
if len(crossings) >= 2:
    for i in range(1, len(crossings)):
        duration = crossings[i] - crossings[i-1]
        print(f"Lap {i}: {duration:.2f}s")
    first_lap = crossings[0] - times[0]
    print(f"Out lap (0): {first_lap:.2f}s")

# Print expected vs actual
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print(f"Total telemetry time: {times[-1]:.1f}s")
print(f"Expected lap count (assuming ~62s laps): {int(times[-1] / 62)}")
print(f"Detected crossings: {len(crossings)}")
print(f"Start-finish line length: {sf_len:.6f}° (≈{sf_len * 111000:.1f}m)")
