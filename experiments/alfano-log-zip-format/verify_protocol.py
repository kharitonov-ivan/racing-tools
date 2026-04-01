"""
Final verification: reconstruct combined 10Hz + 25Hz signals and compare vs Excel.

Protocol:
- 25Hz columns in row N = measurement taken BETWEEN rows N-1 and N (midpoint ~0.05s)
- Speed GPS 25Hz: direct value (÷10 for km/h), always between prev and curr speed
- Lat/Lon 25Hz: signed 16-bit delta in microdegrees; position = row_pos + delta
  Result falls between row N-1 and row N positions

Combined signal: interleave 10Hz row values with 25Hz intermediate values → ~20Hz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
EXCEL_CSV = DATA_DIR / "Excel_SN1061_170326_16H32_SG__P__A_13_6309.csv"
OUT_DIR = Path(__file__).resolve().parent

# Time offset of the 25Hz sample relative to the current row
# (negative = before the row timestamp)
OFFSET_25HZ = -0.05


def to_signed16(v):
    return v - 65536 if v > 32767 else v


def load_raw():
    import zipfile
    zip_path = next(DATA_DIR.glob("ALFANO7_LAP_*.zip"))
    with zipfile.ZipFile(zip_path) as zf:
        lap1 = [n for n in zf.namelist() if n.startswith("LAP_1_")][0]
        with zf.open(lap1) as f:
            df = pd.read_csv(f)
    df["Time"] = np.arange(len(df)) * 0.1
    return df


def load_excel():
    df = pd.read_csv(EXCEL_CSV, sep=";", low_memory=False)
    for col in ["RPM", "Orientation"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(",", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    for col in ["Speed GPS", "Lat.", "Lon.", "Altitude", "Gf. X", "Gf. Y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Time"])
    df["Lap"] = pd.to_numeric(df["Lap"], errors="coerce").ffill()
    df = df[df["Lap"] == 1].copy().sort_values("Time").reset_index(drop=True)
    return df


def build_combined_signals(raw):
    """Build interleaved 10Hz + 25Hz combined signals."""
    n = len(raw)
    time_10hz = raw["Time"].values

    # Speed: 10Hz (raw) + 25Hz intermediate
    speed_10hz = raw["Speed GPS"].values / 10.0
    speed_25hz = raw["Speed GPS 25Hz"].values / 10.0

    # Position: 10Hz (raw) + 25Hz (row_pos + delta)
    lat_10hz = raw["Lat."].values / 1e6
    lon_10hz = raw["Lon."].values / 1e6
    lat_delta = raw["Lat. 25Hz"].apply(to_signed16).values / 1e6
    lon_delta = raw["Lon. 25Hz"].apply(to_signed16).values / 1e6
    lat_25hz = raw["Lat."].values / 1e6 + lat_delta
    lon_25hz = raw["Lon."].values / 1e6 + lon_delta

    # RPM: 10Hz (raw) + 50Hz sub-channels (from next row)
    rpm_10hz = raw["RPM"].values.astype(float)
    rpm_sub_cols = ["RPM 1 20Hz", "RPM 2 50Hz", "RPM 3 50Hz", "RPM 4 50Hz", "RPM 5 50Hz"]
    rpm_sub_offsets = [0.02, 0.04, 0.06, 0.08, 0.10]

    # Interleave 10Hz + 25Hz for speed/position
    # 25Hz sample from row N goes at time = row_N_time + OFFSET_25HZ
    time_25hz = time_10hz + OFFSET_25HZ

    # Combined speed (sorted by time)
    t_combined_speed = np.concatenate([time_10hz, time_25hz[1:]])  # skip row 0's 25Hz (before t=0)
    v_combined_speed = np.concatenate([speed_10hz, speed_25hz[1:]])
    sort_idx = np.argsort(t_combined_speed)
    t_combined_speed = t_combined_speed[sort_idx]
    v_combined_speed = v_combined_speed[sort_idx]

    # Combined lat/lon
    t_combined_pos = np.concatenate([time_10hz, time_25hz[1:]])
    lat_combined = np.concatenate([lat_10hz, lat_25hz[1:]])
    lon_combined = np.concatenate([lon_10hz, lon_25hz[1:]])
    sort_idx = np.argsort(t_combined_pos)
    t_combined_pos = t_combined_pos[sort_idx]
    lat_combined = lat_combined[sort_idx]
    lon_combined = lon_combined[sort_idx]

    # RPM sub-channels (from next row, as proven earlier)
    rpm_sub_times = []
    rpm_sub_values = []
    for i in range(1, n):
        t_prev = time_10hz[i - 1]
        for col, offset in zip(rpm_sub_cols, rpm_sub_offsets):
            rpm_sub_times.append(t_prev + offset)
            rpm_sub_values.append(raw[col].iloc[i])
    rpm_sub_times = np.array(rpm_sub_times)
    rpm_sub_values = np.array(rpm_sub_values, dtype=float)

    return {
        "time_10hz": time_10hz,
        "speed_10hz": speed_10hz,
        "speed_25hz": speed_25hz,
        "time_25hz": time_25hz,
        "t_combined_speed": t_combined_speed,
        "v_combined_speed": v_combined_speed,
        "lat_10hz": lat_10hz,
        "lon_10hz": lon_10hz,
        "lat_25hz": lat_25hz,
        "lon_25hz": lon_25hz,
        "t_combined_pos": t_combined_pos,
        "lat_combined": lat_combined,
        "lon_combined": lon_combined,
        "rpm_10hz": rpm_10hz,
        "rpm_sub_times": rpm_sub_times,
        "rpm_sub_values": rpm_sub_values,
    }


def plot_final(signals, excel):
    """Create the definitive comparison figure."""
    t0, t1 = 0.0, 2.0
    me = (excel["Time"] >= t0) & (excel["Time"] <= t1)

    fig, axes = plt.subplots(4, 1, figsize=(18, 20), sharex=True)
    fig.suptitle(
        "Alfano7 Combined 20Hz Signal Reconstruction vs Excel 100Hz\n"
        "(SN1061 LAP_1 — 25Hz value in row N = measurement between rows N-1 and N)",
        fontsize=13, fontweight="bold",
    )

    # ── 1. RPM: 10Hz + 50Hz sub-channels ──
    ax = axes[0]
    m = (signals["time_10hz"] >= t0) & (signals["time_10hz"] <= t1)
    ms = (signals["rpm_sub_times"] >= t0) & (signals["rpm_sub_times"] <= t1)

    ax.plot(excel.loc[me, "Time"], excel.loc[me, "RPM"],
            "b-", linewidth=1, alpha=0.4, label="Excel 100Hz", zorder=1)
    ax.plot(signals["rpm_sub_times"][ms], signals["rpm_sub_values"][ms],
            "g*", markersize=6, alpha=0.7, zorder=3,
            label="50Hz sub-channels (RPM1-5, row N → interval before N)")
    ax.plot(signals["time_10hz"][m], signals["rpm_10hz"][m],
            "r^", markersize=5, alpha=0.4, zorder=2,
            label="Main RPM (separate measurement, ≠ RPM5)")
    for t in np.arange(t0, t1 + 0.1, 0.1):
        ax.axvline(t, color="gray", alpha=0.1, linewidth=0.5)
    ax.set_ylabel("RPM")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("RPM: 50Hz sub-channels match Excel exactly. Main RPM (red) is a SEPARATE measurement")

    # ── 2. Speed GPS: 10Hz + 25Hz combined ──
    ax = axes[1]
    mc = (signals["t_combined_speed"] >= t0) & (signals["t_combined_speed"] <= t1)
    m25 = (signals["time_25hz"] >= t0) & (signals["time_25hz"] <= t1)

    ax.plot(excel.loc[me, "Time"], excel.loc[me, "Speed GPS"],
            "b-", linewidth=1.5, alpha=0.4, label="Excel 100Hz", zorder=1)
    ax.plot(signals["t_combined_speed"][mc], signals["v_combined_speed"][mc],
            "k--", linewidth=0.8, alpha=0.5, label="Combined ~20Hz (interpolated)", zorder=2)
    ax.plot(signals["time_25hz"][m25], signals["speed_25hz"][m25],
            "g^", markersize=7, zorder=3,
            label="Speed GPS 25Hz (between prev and curr row)")
    ax.plot(signals["time_10hz"][m], signals["speed_10hz"][m],
            "ro", markersize=7, zorder=4, label="Speed GPS 10Hz")
    for t in np.arange(t0, t1 + 0.1, 0.1):
        ax.axvline(t, color="gray", alpha=0.1, linewidth=0.5)
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Speed: 10Hz + 25Hz interleaved → combined ~20Hz signal")

    # ── 3. Latitude: 10Hz + 25Hz (position + delta) ──
    ax = axes[2]
    mcp = (signals["t_combined_pos"] >= t0) & (signals["t_combined_pos"] <= t1)

    ax.plot(excel.loc[me, "Time"], excel.loc[me, "Lat."],
            "b-", linewidth=1.5, alpha=0.4, label="Excel 100Hz", zorder=1)
    ax.plot(signals["t_combined_pos"][mcp], signals["lat_combined"][mcp],
            "k--", linewidth=0.8, alpha=0.5, label="Combined ~20Hz", zorder=2)
    ax.plot(signals["time_25hz"][m25], signals["lat_25hz"][m25],
            "g^", markersize=7, zorder=3,
            label="Lat 25Hz = row_lat + delta (between prev and curr)")
    ax.plot(signals["time_10hz"][m], signals["lat_10hz"][m],
            "ro", markersize=7, zorder=4, label="Lat 10Hz")
    for t in np.arange(t0, t1 + 0.1, 0.1):
        ax.axvline(t, color="gray", alpha=0.1, linewidth=0.5)
    ax.set_ylabel("Latitude (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Latitude: 10Hz + 25Hz (row_pos + signed_delta) → combined ~20Hz")

    # ── 4. Longitude: 10Hz + 25Hz (position + delta) ──
    ax = axes[3]
    ax.plot(excel.loc[me, "Time"], excel.loc[me, "Lon."],
            "b-", linewidth=1.5, alpha=0.4, label="Excel 100Hz", zorder=1)
    ax.plot(signals["t_combined_pos"][mcp], signals["lon_combined"][mcp],
            "k--", linewidth=0.8, alpha=0.5, label="Combined ~20Hz", zorder=2)
    ax.plot(signals["time_25hz"][m25], signals["lon_25hz"][m25],
            'm^', markersize=7, zorder=3,
            label="Lon 25Hz = row_lon + delta (between prev and curr)")
    ax.plot(signals["time_10hz"][m], signals["lon_10hz"][m],
            "ro", markersize=7, zorder=4, label="Lon 10Hz")
    for t in np.arange(t0, t1 + 0.1, 0.1):
        ax.axvline(t, color="gray", alpha=0.1, linewidth=0.5)
    ax.set_ylabel("Longitude (deg)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Longitude: 10Hz + 25Hz (row_pos + signed_delta) → combined ~20Hz")

    plt.tight_layout()
    path = OUT_DIR / "final_protocol_verification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # ── Zoomed RPM detail (0 - 0.3s) ──
    fig2, ax2 = plt.subplots(figsize=(18, 5))
    t0z, t1z = 0.0, 0.3
    mez = (excel["Time"] >= t0z) & (excel["Time"] <= t1z)
    m10z = (signals["time_10hz"] >= t0z) & (signals["time_10hz"] <= t1z)
    msz = (signals["rpm_sub_times"] >= t0z) & (signals["rpm_sub_times"] <= t1z)

    ax2.plot(excel.loc[mez, "Time"], excel.loc[mez, "RPM"],
             "b-", linewidth=1.5, alpha=0.6, label="Excel 100Hz", zorder=2)
    ax2.plot(excel.loc[mez, "Time"], excel.loc[mez, "RPM"],
             "b.", markersize=5, alpha=0.4, zorder=2)
    ax2.plot(signals["rpm_sub_times"][msz], signals["rpm_sub_values"][msz],
             "g*", markersize=14, zorder=4, label="50Hz sub-channels")
    ax2.plot(signals["time_10hz"][m10z], signals["rpm_10hz"][m10z],
             "r^", markersize=8, alpha=0.5, zorder=3, label="Main RPM (≠ RPM5!)")

    # Annotate sub-channel labels
    labels = ["R1", "R2", "R3", "R4", "R5"]
    for j, (tx, vx) in enumerate(
        zip(signals["rpm_sub_times"][msz], signals["rpm_sub_values"][msz])
    ):
        ax2.annotate(labels[j % 5], (tx, vx), textcoords="offset points",
                     xytext=(0, 12), fontsize=7, ha="center", color="darkgreen",
                     fontweight="bold")

    for t in np.arange(t0z, t1z + 0.02, 0.02):
        ax2.axvline(t, color="lightblue", alpha=0.15, linewidth=0.5)
    for t in np.arange(t0z, t1z + 0.1, 0.1):
        ax2.axvline(t, color="gray", alpha=0.3, linewidth=0.5, linestyle="--")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RPM")
    ax2.set_title(
        "RPM Sub-Channel Detail: green stars land EXACTLY on the Excel blue line\n"
        "Sub-channels in row N fill the 0.1s interval BEFORE row N (verified)"
    )
    ax2.legend(fontsize=9)

    path2 = OUT_DIR / "final_rpm_detail.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2}")
    plt.close()


def compute_errors(signals, excel):
    """Quantify how well the reconstruction matches Excel."""
    print("\n" + "=" * 80)
    print("RECONSTRUCTION ERROR vs EXCEL")
    print("=" * 80)

    # Speed: compare combined signal against Excel
    for label, times, values, col in [
        ("Speed 10Hz only", signals["time_10hz"], signals["speed_10hz"], "Speed GPS"),
        ("Speed 10Hz+25Hz", signals["t_combined_speed"], signals["v_combined_speed"], "Speed GPS"),
        ("Lat 10Hz only", signals["time_10hz"], signals["lat_10hz"], "Lat."),
        ("Lat 10Hz+25Hz", signals["t_combined_pos"], signals["lat_combined"], "Lat."),
        ("Lon 10Hz only", signals["time_10hz"], signals["lon_10hz"], "Lon."),
        ("Lon 10Hz+25Hz", signals["t_combined_pos"], signals["lon_combined"], "Lon."),
    ]:
        errors = []
        mask = (times >= 0.1) & (times <= 3.0)
        for t, v in zip(times[mask], values[mask]):
            idx = (excel["Time"] - t).abs().idxmin()
            errors.append(abs(excel.loc[idx, col] - v))
        if errors:
            print(f"  {label:25s}: mean={np.mean(errors):.6f}, max={np.max(errors):.6f}, "
                  f"n={len(errors)}")


def main():
    print("Loading data...")
    raw = load_raw()
    excel = load_excel()
    print(f"  Raw: {len(raw)} rows, Excel (Lap 1): {len(excel)} rows")

    signals = build_combined_signals(raw)
    compute_errors(signals, excel)
    plot_final(signals, excel)


if __name__ == "__main__":
    main()
