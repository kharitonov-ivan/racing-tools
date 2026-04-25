# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scipy"]
# ///
"""Reconstruct steering wheel angle from AIM XRK magnetometer + gyro.

Idea
----
The AIM logger is mounted on the steering wheel, so its body frame rotates
with the wheel around the (approximately fixed) steering column axis.

* Magnetometer: in the body frame the Earth field traces a circle in the plane
  perpendicular to the rotation axis.  PCA of (Mx,My,Mz) gives that axis as
  the eigenvector with smallest variance.  The angle of the projection in the
  in-plane basis = -(kart_yaw + steering_angle) + const.  Subtracting the GPS
  heading isolates the steering wheel angle (modulo offset and sign).
* Gyro: the body-frame angular velocity projected onto the rotation axis =
  steering_rate + kart_yaw_rate (along that axis).  Subtract the GPS heading
  rate and integrate to get steering angle independently.

The two estimates cross-check.  The mag-based one is drift-free but noisy and
needs ellipse fitting (hard/soft iron); the gyro one is smooth but drifts.
We fuse them with a high-pass on gyro + low-pass on mag.
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


def fit_ellipse_whiten(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, W) so that W @ (xy-center).T turns the ellipse into
    a unit circle.  Uses 2nd-moment whitening (good enough when the cloud
    samples the full ellipse, which it does here as the kart yaws around
    the track).
    """
    c = xy.mean(axis=0)
    cov = np.cov((xy - c).T)
    evals, evecs = np.linalg.eigh(cov)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return c, W


def unwrap_deg(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(x)))


def main() -> None:
    df, meta = load_raw(XRK, normalize=False)
    t = df["Time"].to_numpy()
    dt = np.median(np.diff(t))
    print(f"dt={dt*1000:.2f} ms, samples={len(t)}, duration={t[-1]-t[0]:.1f} s")

    M = df[["MagnetomX", "MagnetomY", "MagnetomZ"]].to_numpy(dtype=float)
    G = df[["GyroX", "GyroY", "GyroZ"]].to_numpy(dtype=float)  # deg/s assumed

    # 1) Rotation axis in body frame from magnetometer PCA -----------------
    M0 = M - M.mean(axis=0)
    cov = np.cov(M0.T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, 0]                  # smallest var -> rotation axis
    u, v = evecs[:, 1], evecs[:, 2]     # in-plane basis
    print("axis (body):", axis)
    print("eigenvalues:", evals, "-> in-plane stds:", np.sqrt(evals[1:]))

    # 2) Mag angle: project, whiten, atan2 ---------------------------------
    px = M0 @ u
    py = M0 @ v
    c2, W = fit_ellipse_whiten(np.column_stack([px, py]))
    pw = (np.column_stack([px, py]) - c2) @ W.T
    mag_angle = np.unwrap(np.arctan2(pw[:, 1], pw[:, 0]))      # radians
    mag_angle_deg = np.degrees(mag_angle)

    # 3) Kart yaw from GPS -------------------------------------------------
    kart_yaw_deg = unwrap_deg(df["GPS Heading"].to_numpy())     # absolute world heading
    kart_yaw_rad = np.radians(kart_yaw_deg)

    # 4) Mag-based steering: align signs --------------------------------
    # device_world_yaw = kart_yaw + s * steering, where s = +/- 1 depending on
    # which way the column points.  mag_angle = -device_world_yaw + const.
    # We try both signs and pick the one that yields the smaller residual when
    # we expect steering ~ 0 on straights (high speed, low GPS LatAcc).
    speed = df["GPS Speed"].to_numpy()
    latacc = df["GPS LatAcc"].to_numpy()
    straight_mask = (speed > 0.7 * np.nanmax(speed)) & (np.abs(latacc) < 0.2 * np.nanmax(np.abs(latacc)))
    print(f"straight samples: {straight_mask.sum()} / {len(t)}")

    best = None
    for sgn in (+1, -1):
        cand = -mag_angle - sgn * kart_yaw_rad        # = sgn*steering + const
        # remove offset
        offset = np.median(cand[straight_mask]) if straight_mask.any() else np.median(cand)
        steer = (cand - offset) * sgn                  # back to steering, deg
        # quality: variance on straights should be small
        var_straight = np.var(steer[straight_mask]) if straight_mask.any() else np.inf
        print(f"sign={sgn:+d} -> straight-line variance of steering = {np.degrees(np.sqrt(var_straight)):.2f} deg rms")
        if best is None or var_straight < best[0]:
            best = (var_straight, sgn, steer)
    _, sgn_mag, steer_mag = best
    steer_mag_deg = np.degrees(steer_mag)
    print(f"chosen sign: {sgn_mag}")

    # 5) Gyro-based steering ----------------------------------------------
    # ω_device · n = steering_rate + kart_yaw_rate (about column axis).
    # GPS-heading-rate is the kart yaw rate (assuming column ≈ vertical).
    omega_n_dps = G @ axis                                       # deg/s
    yaw_rate_dps = np.gradient(kart_yaw_deg, t)
    steer_rate_dps = omega_n_dps - yaw_rate_dps
    # smooth a touch (50 Hz cutoff already implied by 100 Hz sampling)
    steer_rate_dps = savgol_filter(steer_rate_dps, 21, 3)
    # Integrate, then high-pass to kill drift, then add the magnetometer's
    # low-frequency content => complementary filter.
    steer_int_dps = np.cumsum(steer_rate_dps) * dt              # deg
    steer_int_dps -= np.median(steer_int_dps)
    # complementary fusion: mag for low freq, gyro-int for high freq
    tau = 0.5  # s
    alpha = tau / (tau + dt)
    fused = np.zeros_like(steer_mag_deg)
    fused[0] = steer_mag_deg[0]
    for i in range(1, len(fused)):
        fused[i] = alpha * (fused[i - 1] + steer_rate_dps[i] * dt) + (1 - alpha) * steer_mag_deg[i]

    # 6) Save & plot -------------------------------------------------------
    out_csv = OUT / "steering_estimate.csv"
    pd.DataFrame({
        "Time": t,
        "GPS Speed": speed,
        "GPS Heading": df["GPS Heading"].to_numpy(),
        "GPS LatAcc": latacc,
        "steer_mag_deg": steer_mag_deg,
        "steer_int_deg": steer_int_dps,
        "steer_fused_deg": fused,
    }).to_csv(out_csv, index=False)
    print("wrote", out_csv)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    ax = axes[0]
    ax.plot(t, speed, lw=0.6)
    ax.set_ylabel("GPS Speed [m/s]"); ax.grid(alpha=.3)
    ax = axes[1]
    ax.plot(t, df["GPS LatAcc"], lw=0.6, color="tab:red", label="GPS LatAcc [g]")
    ax.set_ylabel("LatAcc [g]"); ax.grid(alpha=.3); ax.legend()
    ax = axes[2]
    ax.plot(t, steer_mag_deg, lw=0.4, label="mag", alpha=0.6)
    ax.plot(t, steer_int_dps, lw=0.4, label="gyro∫", alpha=0.6)
    ax.plot(t, fused, lw=0.7, label="fused", color="k")
    ax.set_ylabel("Steering [deg]"); ax.grid(alpha=.3); ax.legend(loc="upper right")
    ax = axes[3]
    ax.plot(t, steer_rate_dps, lw=0.4, color="tab:purple")
    ax.set_ylabel("Steering rate [deg/s]"); ax.set_xlabel("time [s]"); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "steering_full.png", dpi=130)
    print("wrote", OUT / "steering_full.png")

    # Best-lap zoom --------------------------------------------------------
    # find longest "moving" segment as a fallback if no lap_info
    if "Distance" in df.columns:
        # take a 60 s window in the middle as best-effort zoom
        i0 = len(t) // 2
        i1 = i0 + int(60 / dt)
        i1 = min(i1, len(t))
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(t[i0:i1], steer_mag_deg[i0:i1], lw=0.7, label="mag", alpha=0.6)
        ax.plot(t[i0:i1], fused[i0:i1], lw=1.0, label="fused", color="k")
        ax2 = ax.twinx()
        ax2.plot(t[i0:i1], df["GPS LatAcc"].iloc[i0:i1], lw=0.6, color="tab:red", alpha=0.6, label="LatAcc")
        ax.set_ylabel("Steering [deg]"); ax2.set_ylabel("LatAcc [g]")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        ax.grid(alpha=.3); ax.set_xlabel("time [s]")
        ax.set_title(f"Steering vs LatAcc, 60 s zoom @ t={t[i0]:.0f}s")
        fig.tight_layout()
        fig.savefig(OUT / "steering_zoom.png", dpi=130)
        print("wrote", OUT / "steering_zoom.png")


if __name__ == "__main__":
    main()
