#!/usr/bin/env python
"""
Track visualization script with satellite imagery.

Displays track geometry overlaid on satellite imagery with:
- Inner and outer boundaries
- Centerline
- Start/finish line
- Bestline (if available)
- Straights and turns segmentation

Usage:
    python -m racing_tools.track.visualize_track /path/to/track/dir

Example:
    python -m racing_tools.track.visualize_track racing_tools/track/data/RIMSportKarting
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd

logger = logging.getLogger(__name__)
from shapely.geometry import LineString

from .track import Track
from .utils import load_polyline_geojson, get_transformer, transform_coordinates
from .constants import WGS84_CRS, WEBMERCATOR_CRS


def plot_track(
    track: Track,
    track_dir: Path,
    output_path: Path = None,
    figsize: tuple = (32, 24),
    dpi: int = 300,
    save_to_track_dir: bool = True,
):
    """
    Plot track geometry overlaid on satellite imagery.

    Args:
        track: Track instance
        track_dir: Track directory path
        output_path: Optional path to save figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
    """
    # Get transformer for Web Mercator (for satellite tiles)
    transformer_utm = track.get_transformer()
    transformer_webmerc = get_transformer(track.utm_zone, WEBMERCATOR_CRS)

    # Convert centerline to Web Mercator for plotting
    centerline_utm = track.centerline
    if centerline_utm is None:
        raise ValueError("Track centerline not available")
    centerline_webmerc = transform_coordinates(centerline_utm, track.utm_zone, WEBMERCATOR_CRS)

    # Calculate bounds with padding
    padding = 100  # meters
    xmin, xmax = centerline_webmerc[:, 0].min() - padding, centerline_webmerc[:, 0].max() + padding
    ymin, ymax = centerline_webmerc[:, 1].min() - padding, centerline_webmerc[:, 1].max() + padding

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set extent first (important for contextily)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Add satellite imagery
    imagery_loaded = False
    try:
        # Try multiple satellite imagery sources
        sources = [
            ("Esri WorldImagery", ctx.providers.Esri.WorldImagery),
            ("OpenStreetMap", ctx.providers.OpenStreetMap.Mapnik),
        ]

        for source_name, source in sources:
            try:
                ctx.add_basemap(
                    ax,
                    crs=WEBMERCATOR_CRS,
                    source=source,
                    zoom=18,
                    attribution=False,
                )
                logger.info(f"Loaded imagery from {source_name}")
                imagery_loaded = True
                break
            except Exception as e:
                logger.debug(f"Failed {source_name}: {e}")
                continue

        if not imagery_loaded:
            logger.info("Using light gray background")
            ax.set_facecolor("#f0f0f0")
    except Exception as e:
        logger.warning(f"Could not load satellite imagery: {e}")
        logger.info("Using light gray background")
        ax.set_facecolor("#f0f0f0")
    ax.set_aspect("equal")

    # Plot inner and outer boundaries
    geometry_dir = track_dir / "geometry"
    inner_path = geometry_dir / "track-inner.geojson"
    outer_path = geometry_dir / "track-outer.geojson"

    if inner_path.exists():
        inner_wgs84 = load_polyline_geojson(inner_path)
        if inner_wgs84:
            inner_arr = np.array(inner_wgs84)
            inner_webmerc = transform_coordinates(inner_arr, WGS84_CRS, WEBMERCATOR_CRS)
            ax.plot(inner_webmerc[:, 0], inner_webmerc[:, 1], "b-", linewidth=2, label="Inner Boundary", alpha=0.7)

    if outer_path.exists():
        outer_wgs84 = load_polyline_geojson(outer_path)
        if outer_wgs84:
            outer_arr = np.array(outer_wgs84)
            outer_webmerc = transform_coordinates(outer_arr, WGS84_CRS, WEBMERCATOR_CRS)
            ax.plot(outer_webmerc[:, 0], outer_webmerc[:, 1], "b-", linewidth=2, alpha=0.7)

    # Plot centerline
    ax.plot(centerline_webmerc[:, 0], centerline_webmerc[:, 1], "r--", linewidth=2, label="Centerline")

    # Plot bestline if available
    if track.bestline_wgs84:
        bestline_arr = np.array(track.bestline_wgs84)
        bestline_webmerc = transform_coordinates(bestline_arr, WGS84_CRS, WEBMERCATOR_CRS)
        ax.plot(bestline_webmerc[:, 0], bestline_webmerc[:, 1], "g-", linewidth=2, label="Bestline (Racing Line)", alpha=0.8)

    # Plot sector lines (SF, S1, S2, ...) with coordinate labels
    sector_colors = {"SF": "yellow", "S1": "orange", "S2": "lime"}
    transformer_to_wgs84 = track._get_transformer_to_wgs84()
    for sector_name in track.sectors_utm:
        sector_wgs84 = track.get_sector_wgs84(sector_name)
        if not sector_wgs84:
            continue
        sector_arr = np.array(sector_wgs84)
        sector_webmerc = transform_coordinates(sector_arr, WGS84_CRS, WEBMERCATOR_CRS)
        color = sector_colors.get(sector_name, "white")
        ax.plot(sector_webmerc[:, 0], sector_webmerc[:, 1], "-", color=color, linewidth=4, label=sector_name)
        # Label with name + endpoint coordinates
        p1_lat, p1_lon = sector_wgs84[0][1], sector_wgs84[0][0]
        p2_lat, p2_lon = sector_wgs84[-1][1], sector_wgs84[-1][0]
        mid = sector_webmerc.mean(axis=0)
        label_text = (f"{sector_name}\n"
                      f"({p1_lat:.6f}, {p1_lon:.6f})\n"
                      f"({p2_lat:.6f}, {p2_lon:.6f})")
        ax.text(mid[0], mid[1], label_text, fontsize=8, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85), zorder=5)

    # Plot sector-bestline intersection points with coordinate labels
    if track.bestline_utm and track.sectors_utm:
        bestline_line = LineString(track.bestline_utm)
        for sector_name, sector_pts in track.sectors_utm.items():
            sector_line = LineString(sector_pts)
            ix = sector_line.intersection(bestline_line)
            if ix.is_empty:
                sector_mid = sector_line.interpolate(0.5, normalized=True)
                proj_pt = bestline_line.interpolate(bestline_line.project(sector_mid))
            elif ix.geom_type == "Point":
                proj_pt = ix
            else:
                proj_pt = ix.geoms[0] if hasattr(ix, 'geoms') else ix
            dist_m = bestline_line.project(proj_pt)
            pt_utm = np.array([[proj_pt.x, proj_pt.y]])
            pt_webmerc = transform_coordinates(pt_utm, track.utm_zone, WEBMERCATOR_CRS).flatten()
            lon, lat = transformer_to_wgs84.transform(proj_pt.x, proj_pt.y)
            color = sector_colors.get(sector_name, "white")
            ax.plot(pt_webmerc[0], pt_webmerc[1], "o", color=color, markersize=10,
                    markeredgecolor="black", markeredgewidth=2, zorder=6)
            ax.annotate(f"{sector_name} x bestline\n({lat:.6f}, {lon:.6f})\n{dist_m:.1f}m",
                        xy=(pt_webmerc[0], pt_webmerc[1]),
                        xytext=(15, -25), textcoords="offset points",
                        fontsize=7, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1),
                        zorder=7)

    # Plot segments (straights and turns) with highlighter style
    if track.segments:
        for i, seg in enumerate(track.segments):
            seg_points = np.array(seg["points"])
            seg_webmerc = transform_coordinates(seg_points, track.utm_zone, WEBMERCATOR_CRS)

            # Create LineString for fill_between

            seg_line = LineString(seg_webmerc)

            if seg["type"] == "straight":
                # Highlighter style: cyan transparent fill
                color = "cyan"
                label = "Straight" if i == 0 else None

                # Plot thick line with transparency (highlighter effect)
                ax.plot(seg_webmerc[:, 0], seg_webmerc[:, 1], color=color, linewidth=8, alpha=0.3, solid_capstyle="round", zorder=2)

                # Thinner core line
                ax.plot(seg_webmerc[:, 0], seg_webmerc[:, 1], color=color, linewidth=2, alpha=0.6, zorder=3)

                # Add segment number
                mid_idx = len(seg_webmerc) // 2
                ax.text(
                    seg_webmerc[mid_idx, 0],
                    seg_webmerc[mid_idx, 1],
                    f"Str{i // 2 + 1}",
                    fontsize=9,
                    color="blue",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.8),
                    zorder=4,
                )

            elif seg["type"] == "turn":
                # Highlighter style: magenta transparent fill
                color = "magenta"
                label = "Turn" if i == 0 else None

                # Plot thick line with transparency (highlighter effect)
                ax.plot(seg_webmerc[:, 0], seg_webmerc[:, 1], color=color, linewidth=8, alpha=0.3, solid_capstyle="round", zorder=2)

                # Thinner core line
                ax.plot(seg_webmerc[:, 0], seg_webmerc[:, 1], color=color, linewidth=2, alpha=0.6, zorder=3)

                # Add segment number
                mid_idx = len(seg_webmerc) // 2
                ax.text(
                    seg_webmerc[mid_idx, 0],
                    seg_webmerc[mid_idx, 1],
                    f"T{i // 2 + 1}",
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="magenta", alpha=0.8),
                    zorder=4,
                )

    # Add legend
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Add title
    config_name = track_dir.name
    ax.set_title(
        f"Track: {config_name}\nLength: {track.total_length:.1f}m | Segments: {len(track.segments) if track.segments else 0} | UTM: {track.utm_zone}",
        fontsize=14,
        fontweight="bold",
    )

    # Remove axis ticks for cleaner map
    ax.set_xticks([])
    ax.set_yticks([])

    # Add north arrow
    ax.annotate(
        "N",
        xy=(xmin + (xmax - xmin) * 0.05, ymax - (ymax - ymin) * 0.1),
        fontsize=20,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle", facecolor="white", alpha=0.8),
    )
    ax.arrow(
        xmin + (xmax - xmin) * 0.05,
        ymax - (ymax - ymin) * 0.12,
        0,
        (ymax - ymin) * 0.03,
        head_width=(xmax - xmin) * 0.01,
        head_length=(ymax - ymin) * 0.01,
        fc="black",
        ec="black",
        linewidth=2,
    )

    # Add scale bar (approximate)
    scale_length = 100  # meters
    scale_x = xmin + (xmax - xmin) * 0.05
    scale_y = ymin + (ymax - ymin) * 0.05
    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], "k-", linewidth=3)
    ax.text((scale_x + scale_x + scale_length) / 2, scale_y - (ymax - ymin) * 0.02, f"{scale_length}m", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved visualization to: {output_path}")
    elif save_to_track_dir:
        # Save to track directory by default
        default_output = track_dir / "track_visualization.png"
        plt.savefig(default_output, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved visualization to: {default_output}")
    else:
        plt.show()

    plt.close()


def main():
    """Main entry point for track visualization."""
    parser = argparse.ArgumentParser(description="Visualize track geometry with satellite imagery")
    parser.add_argument("track_dir", type=str, help="Path to track directory containing GeoJSON files")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path for saving visualization (default: show interactively)")
    parser.add_argument("--size", type=int, nargs=2, default=[32, 24], metavar=("WIDTH", "HEIGHT"), help="Figure size in inches (default: 32 24)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for saved figure (default: 300)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively instead of saving")

    args = parser.parse_args()
    track_dir = Path(args.track_dir)

    if not track_dir.exists():
        logger.error(f"Track directory not found: {track_dir}")
        sys.exit(1)

    if not track_dir.is_dir():
        logger.error(f"Path is not a directory: {track_dir}")
        sys.exit(1)

    # Load track
    logger.info(f"Loading track from: {track_dir}")
    try:
        track = Track.load(track_dir)
    except Exception as e:
        logger.error(f"Error loading track: {e}")
        sys.exit(1)

    # Generate output path if not specified
    output_path = None
    save_to_track = True

    if args.output:
        output_path = Path(args.output)
        save_to_track = False

    if args.show:
        save_to_track = False

    # Plot track
    try:
        plot_track(
            track=track,
            track_dir=track_dir,
            output_path=output_path,
            figsize=tuple(args.size),
            dpi=args.dpi,
            save_to_track_dir=save_to_track,
        )
    except Exception as e:
        logger.error(f"Error visualizing track: {e}")
        import traceback

        logger.exception(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
