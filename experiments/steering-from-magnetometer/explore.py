# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scipy"]
# ///
"""Explore magnetometer/gyro/GPS data on the AIM XRK file.

Goal: confirm a rotation axis (steering column) exists in the magnetometer
cloud. PCA on (Mx,My,Mz) — smallest eigenvalue ≈ rotation axis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from racing_tools.session.aim.loader import load_raw

XRK = Path(
    "/mnt/c/users/supra/Desktop/racing-data/new/"
    "2026-04-23_17-17_RIMSportKarting_RotaxMax_IvanKharitonov_Training/"
    "IK_RotaxMax2_RIM Kart_a_1147.xrk"
)
OUT = Path(__file__).parent


def main() -> None:
    df, meta = load_raw(XRK, normalize=False)
    print("Shape:", df.shape, "duration:", df["Time"].iloc[-1] - df["Time"].iloc[0], "s")

    cols = ["MagnetomX", "MagnetomY", "MagnetomZ", "GyroX", "GyroY", "GyroZ", "GPS Heading", "GPS Speed"]
    print(df[cols].describe())

    M = df[["MagnetomX", "MagnetomY", "MagnetomZ"]].to_numpy(dtype=float)
    M0 = M - M.mean(axis=0)

    cov = np.cov(M0.T)
    evals, evecs = np.linalg.eigh(cov)
    print("\nMagnetometer covariance eigenvalues (asc):", evals)
    print("Eigenvectors (columns):\n", evecs)
    # smallest eval -> rotation axis (the field projected on this axis is ~constant
    # while the wheel spins; the other 2 trace a circle).
    axis = evecs[:, 0]
    print("Rotation axis (smallest var):", axis, "std along it:", np.sqrt(evals[0]))
    print("In-plane stds:", np.sqrt(evals[1]), np.sqrt(evals[2]))

    # Project M onto plane perpendicular to axis
    in_plane = M0 - np.outer(M0 @ axis, axis)
    # Pick two orthonormal vectors spanning the plane = evecs[:,1], evecs[:,2]
    u = evecs[:, 1]
    v = evecs[:, 2]
    px = in_plane @ u
    py = in_plane @ v

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    sc = ax.scatter(px[::20], py[::20], s=1, c=df["Time"].iloc[::20], cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title("Magnetometer projected onto rotation plane (color=time)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="time [s]")

    ax = axes[0, 1]
    ax.plot(df["Time"], df["GyroX"], label="GyroX", lw=0.5)
    ax.plot(df["Time"], df["GyroY"], label="GyroY", lw=0.5)
    ax.plot(df["Time"], df["GyroZ"], label="GyroZ", lw=0.5)
    ax.set_title("Gyro raw")
    ax.legend(loc="upper right")
    ax.set_xlabel("time [s]")

    ax = axes[1, 0]
    ax.plot(df["Time"], df["MagnetomX"], label="Mx", lw=0.5)
    ax.plot(df["Time"], df["MagnetomY"], label="My", lw=0.5)
    ax.plot(df["Time"], df["MagnetomZ"], label="Mz", lw=0.5)
    ax.set_title("Magnetometer raw")
    ax.legend(loc="upper right")
    ax.set_xlabel("time [s]")

    ax = axes[1, 1]
    ax.plot(df["Time"], df["GPS Speed"], label="GPS Speed", lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(df["Time"], df["GPS Heading"], color="orange", label="GPS Heading", lw=0.5)
    ax.set_title("Speed & GPS heading")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(OUT / "explore.png", dpi=130)
    print("Saved", OUT / "explore.png")


if __name__ == "__main__":
    main()
