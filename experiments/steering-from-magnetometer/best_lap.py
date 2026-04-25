# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scipy"]
# ///
"""Best-lap steering reconstruction with proper bias handling.

Improvements over steering.py:
* Estimate gyro bias from a still segment (lowest |GPS Speed| + low gyro
  magnitude) instead of trusting the raw integral.
* Locate the best lap via XRK lap_info (through the TDA loader metadata
  isn't easy, so we scan for repeating GPS positions and split by best
  guess: take the fastest 60-90 s segment as a proxy here).
* Plot steering vs distance for the chosen lap, alongside speed and LatAcc.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from racing_tools.session.aim.loader import load_raw

XRK = Path(
    "/mnt/c/users/supra/Desktop/racing-data/new/"
    "2026-04-23_17-17_RIMSportKarting_RotaxMax_IvanKharitonov_Training/"
    "IK_RotaxMax2_RIM Kart_a_1147.xrk"
)
OUT = Path(__file__).parent


def detect_laps(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Detect lap boundaries via simple geometric crossing of the start
    line, picked as the location where the kart was when GPS LatAcc≈0
    and Speed is in the upper half of the distribution (start/finish straight).
    Returns list of (t_start, t_end) in seconds.
    """
    t = df["Time"].to_numpy()
    lat = df["GPS Latitude"].to_numpy()
    lon = df["GPS Longitude"].to_numpy()
    sp = df["GPS Speed"].to_numpy()

    moving = sp > max(2.0, 0.3 * np.nanmax(sp))
    if moving.sum() < 100:
        return []
    # pick start as the median high-speed location (very rough)
    cx, cy = np.median(lat[moving]), np.median(lon[moving])
    # use the point closest to centroid as anchor, find re-passings
    d = np.hypot(lat - cx, lon - cy)
    # crossings = local minima of d below a threshold
    thr = np.percentile(d[moving], 20)
    below = d < thr
    # rising edges
    edges = np.where(np.diff(below.astype(int)) == 1)[0]
    # require at least 20 s between crossings
    laps = []
    last_t = -np.inf
    for e in edges:
        if t[e] - last_t > 20:
            laps.append(t[e])
            last_t = t[e]
    return [(laps[i], laps[i + 1]) for i in range(len(laps) - 1)]


def main() -> None:
    df, meta = load_raw(XRK, normalize=False)
    t = df["Time"].to_numpy()
    dt = float(np.median(np.diff(t)))

    M = df[["MagnetomX", "MagnetomY", "MagnetomZ"]].to_numpy(dtype=float)
    G = df[["GyroX", "GyroY", "GyroZ"]].to_numpy(dtype=float)
    speed = df["GPS Speed"].to_numpy()
    latacc = df["GPS LatAcc"].to_numpy()
    heading = df["GPS Heading"].to_numpy()
    distance = df["Distance"].to_numpy() if "Distance" in df.columns else np.cumsum(speed) * dt

    # PCA axis
    M0 = M - M.mean(axis=0)
    cov = np.cov(M0.T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, 0]
    u, v = evecs[:, 1], evecs[:, 2]

    # Mag-based steering
    px, py = M0 @ u, M0 @ v
    cov2 = np.cov(np.column_stack([px, py]).T)
    e2, V2 = np.linalg.eigh(cov2)
    W = V2 @ np.diag(1.0 / np.sqrt(e2)) @ V2.T
    pw = (np.column_stack([px, py]) - np.array([px.mean(), py.mean()])) @ W.T
    mag_angle = np.unwrap(np.arctan2(pw[:, 1], pw[:, 0]))
    yaw_unwrap = np.radians(np.degrees(np.unwrap(np.radians(heading))))

    # both signs already; we know -1 was correct from steering.py
    cand = -mag_angle - (-1) * yaw_unwrap                 # = -steering + const
    straight = (speed > 0.7 * np.nanmax(speed)) & (np.abs(latacc) < 0.2 * np.nanmax(np.abs(latacc)))
    offset = np.median(cand[straight])
    steer_mag = np.degrees((cand - offset) * (-1))         # deg

    # Gyro bias: estimate from segments where the kart is essentially still
    # (very low speed AND tiny gyro magnitude over a window).
    gmag = np.linalg.norm(G, axis=1)
    still = (speed < 0.5) & (gmag < np.percentile(gmag, 10))
    if still.sum() > 50:
        bias = G[still].mean(axis=0)
    else:
        # fallback: assume long-run mean is bias (kart laps cancel out)
        bias = G.mean(axis=0)
    print("gyro bias (deg/s):", bias, "still samples:", int(still.sum()))

    omega_n = (G - bias) @ axis
    yaw_rate = np.gradient(np.degrees(yaw_unwrap), t)
    steer_rate = omega_n - yaw_rate
    steer_rate = savgol_filter(steer_rate, 21, 3)
    steer_int = np.cumsum(steer_rate) * dt
    steer_int -= np.median(steer_int[straight])

    # Complementary filter
    tau = 1.0
    alpha = tau / (tau + dt)
    fused = np.empty_like(steer_mag)
    fused[0] = steer_mag[0]
    for i in range(1, len(fused)):
        fused[i] = alpha * (fused[i - 1] + steer_rate[i] * dt) + (1 - alpha) * steer_mag[i]

    # Detect laps and pick fastest one
    laps = detect_laps(df)
    if not laps:
        print("no laps detected; bailing on best-lap plot")
        return
    durations = [(b - a, a, b) for a, b in laps]
    durations.sort()
    # filter unrealistic short laps
    durations = [d for d in durations if d[0] > 30 and d[0] < 120]
    if not durations:
        print("no plausible laps")
        return
    best_dur, t0, t1 = durations[0]
    print(f"best lap: {best_dur:.2f}s, t∈[{t0:.1f}, {t1:.1f}]")
    mask = (t >= t0) & (t <= t1)
    d_lap = distance[mask] - distance[mask][0]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(d_lap, speed[mask], lw=0.9, color="tab:blue")
    axes[0].set_ylabel("Speed [m/s]"); axes[0].grid(alpha=.3)
    axes[0].set_title(f"Best lap: {best_dur:.2f}s, t∈[{t0:.1f},{t1:.1f}]s")

    axes[1].plot(d_lap, latacc[mask], lw=0.9, color="tab:red")
    axes[1].axhline(0, color="k", lw=0.4)
    axes[1].set_ylabel("LatAcc [g]"); axes[1].grid(alpha=.3)

    axes[2].plot(d_lap, steer_mag[mask], lw=0.6, alpha=0.5, label="mag")
    axes[2].plot(d_lap, fused[mask], lw=1.2, color="k", label="fused")
    axes[2].axhline(0, color="k", lw=0.4)
    axes[2].set_ylabel("Steering [deg]"); axes[2].legend(); axes[2].grid(alpha=.3)

    axes[3].plot(d_lap, steer_rate[mask], lw=0.6, color="tab:purple")
    axes[3].set_ylabel("Steer rate [deg/s]")
    axes[3].set_xlabel("distance [m]")
    axes[3].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(OUT / "best_lap.png", dpi=140)
    print("wrote", OUT / "best_lap.png")

    # Save full estimate for downstream use
    pd.DataFrame({
        "Time": t,
        "Distance": distance,
        "GPS Speed": speed,
        "GPS LatAcc": latacc,
        "GPS Heading": heading,
        "steer_mag_deg": steer_mag,
        "steer_int_deg": steer_int,
        "steer_fused_deg": fused,
        "steer_rate_dps": steer_rate,
    }).to_csv(OUT / "steering_full.csv", index=False)
    print("wrote", OUT / "steering_full.csv")


if __name__ == "__main__":
    main()
