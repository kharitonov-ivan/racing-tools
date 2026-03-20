"""
Track class for unified track representation.

Combines:
- Layout (bounds, segments)
- Start/finish line
- Centerline projector for distance calculations
- Bestline (optimal racing line)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from pyproj import Transformer
from shapely.geometry import LineString

logger = logging.getLogger(__name__)

from .utils import (
    load_polyline_geojson,
    load_track_config,
    get_transformer,
    compute_centerline,
    transform_coordinates,
)
from .segmentation import segment_track
from .constants import DEFAULT_UTM_ZONE, WGS84_CRS


def _determine_utm_zone_from_coords(
    lons: np.ndarray,
    lats: np.ndarray,
) -> str:
    """
    Determine UTM zone from WGS84 coordinates.

    Args:
        lons: Longitude values in degrees, shape (N,)
        lats: Latitude values in degrees, shape (N,)

    Returns:
        UTM zone EPSG code (e.g., 'EPSG:32638' for zone 38N)

    Note:
        Uses the centroid of the coordinates to determine zone.
        Northern hemisphere zones (326xx) for lat >= 0,
        Southern hemisphere zones (327xx) for lat < 0.
    """
    # Use centroid for zone determination
    center_lon = float(np.mean(lons))
    center_lat = float(np.mean(lats))

    # UTM zone number: 1-60, starting from -180°
    zone = int((center_lon + 180) / 6) + 1

    # Northern hemisphere: EPSG:326XX, Southern: EPSG:327XX
    hemisphere = 326 if center_lat >= 0 else 327

    return f"EPSG:{hemisphere}{zone:02d}"


class Track:
    """
    Unified track representation combining layout, start/finish, and centerline.

    Attributes:
        bounds: Bounding box (xmin, xmax, ymin, ymax) in UTM (property, computed from centerline)
        segments: List of track segments (straights/turns)
        start_finish_utm: Start/finish line coordinates in UTM
        start_finish_wgs84: Start/finish line coordinates in WGS84 (property, computed from UTM)
        bestline_utm: Bestline (optimal racing line) in UTM
        bestline_wgs84: Bestline in WGS84 (property, computed from UTM)
        utm_zone: UTM zone EPSG code (auto-determined from WGS84 coordinates)
        total_length: Total track length in meters (property)
        centerline: Centerline coordinates in UTM (property, computed from boundaries)

    Note:
        - Boundaries are specified in WGS84 (lon/lat) and converted to UTM internally
        - UTM zone is auto-determined from coordinates unless explicitly provided
        - WGS84 coordinates and bounds are computed dynamically to avoid duplication

    Example:
        >>> track = Track.load("/path/to/track")
        >>> distance = track.project(point_utm)
        >>> print(f"Track length: {track.total_length:.1f}m")
        >>> # WGS84 coordinates are computed on-demand
        >>> sf_wgs84 = track.start_finish_wgs84
        >>> # Bounds are computed from centerline
        >>> print(f"Track bounds: {track.bounds}")
    """

    def __init__(
        self,
        inner_boundary_wgs84: Optional[np.ndarray] = None,
        outer_boundary_wgs84: Optional[np.ndarray] = None,
        utm_zone: Optional[str] = None,
        segments: Optional[List[Dict]] = None,
        start_finish_utm: Optional[List[Tuple[float, float]]] = None,
        bestline_utm: Optional[List[Tuple[float, float]]] = None,
    ):
        # Boundaries in WGS84 for zone determination
        self._inner_boundary_wgs84 = inner_boundary_wgs84
        self._outer_boundary_wgs84 = outer_boundary_wgs84
        self._centerline_coords: Optional[np.ndarray] = None
        self._centerline_cache = {}  # Cache for centerline computation

        # Determine UTM zone from WGS84 coordinates if not provided
        if utm_zone is None:
            boundary = inner_boundary_wgs84 if inner_boundary_wgs84 is not None else outer_boundary_wgs84
            if boundary is not None and len(boundary) > 0:
                # WGS84: lon, lat
                lons = boundary[:, 0]
                lats = boundary[:, 1]
                utm_zone = _determine_utm_zone_from_coords(lons, lats)
            else:
                utm_zone = DEFAULT_UTM_ZONE

        # Projection info
        self.utm_zone = utm_zone

        # Transform boundaries to UTM for centerline computation
        transformer = get_transformer(WGS84_CRS, utm_zone)
        self._inner_boundary_utm: Optional[np.ndarray] = None
        self._outer_boundary_utm: Optional[np.ndarray] = None

        if inner_boundary_wgs84 is not None:
            lons, lats = inner_boundary_wgs84[:, 0], inner_boundary_wgs84[:, 1]
            xs, ys = transformer.transform(lons, lats)
            self._inner_boundary_utm = np.column_stack((xs, ys))

        if outer_boundary_wgs84 is not None:
            lons, lats = outer_boundary_wgs84[:, 0], outer_boundary_wgs84[:, 1]
            xs, ys = transformer.transform(lons, lats)
            self._outer_boundary_utm = np.column_stack((xs, ys))

        # Layout data
        self.segments = segments
        self._bounds: Optional[Tuple[float, float, float, float]] = None  # Computed dynamically

        # Start/finish line (UTM only, WGS84 computed dynamically)
        self.start_finish_utm = start_finish_utm

        # Bestline (UTM only, WGS84 computed dynamically)
        self.bestline_utm = bestline_utm

        # Centerline projector (lazy init)
        self._projector_initialized = False
        self._projector_points: Optional[np.ndarray] = None
        self._cumulative_dists: Optional[np.ndarray] = None
        self._segments_vec: Optional[np.ndarray] = None
        self._segment_lengths_sq: Optional[np.ndarray] = None
        self._total_length: float = 0.0

        # Transformer cache (UTM to WGS84)
        self._transformer_to_wgs84: Optional[Transformer] = None

    def get_transformer(self) -> Transformer:
        """Get WGS84 to track UTM zone transformer."""
        return Transformer.from_crs(WGS84_CRS, self.utm_zone, always_xy=True)

    def _get_transformer_to_wgs84(self) -> Transformer:
        """Get UTM to WGS84 transformer (cached)."""
        if self._transformer_to_wgs84 is None:
            self._transformer_to_wgs84 = Transformer.from_crs(self.utm_zone, WGS84_CRS, always_xy=True)
        return self._transformer_to_wgs84

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Bounding box (xmin, xmax, ymin, ymax) in UTM coordinates.

        Computed from centerline (cached after first computation).
        """
        if self._bounds is not None:
            return self._bounds

        # Compute bounds from centerline
        centerline = self.centerline
        if centerline is None:
            return (0.0, 0.0, 0.0, 0.0)

        xs = centerline[:, 0]
        ys = centerline[:, 1]
        self._bounds = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
        return self._bounds

    @bounds.setter
    def bounds(self, value: Tuple[float, float, float, float]):
        """Set bounds explicitly (for caching or manual override)."""
        self._bounds = value

    @property
    def start_finish_wgs84(self) -> Optional[List[Tuple[float, float]]]:
        """Start/finish line coordinates in WGS84 (computed from UTM)."""
        if self.start_finish_utm is None:
            return None

        transformer = self._get_transformer_to_wgs84()
        pts = np.array(self.start_finish_utm)
        lons, lats = transformer.transform(pts[:, 0], pts[:, 1])
        return list(zip(lons, lats))

    @property
    def bestline_wgs84(self) -> Optional[List[Tuple[float, float]]]:
        """Bestline in WGS84 (computed from UTM)."""
        if self.bestline_utm is None:
            return None

        transformer = self._get_transformer_to_wgs84()
        pts = np.array(self.bestline_utm)
        lons, lats = transformer.transform(pts[:, 0], pts[:, 1])
        return list(zip(lons, lats))

    @property
    def polylines(self) -> List[List[Tuple[float, float]]]:
        """
        List of polylines for track rendering.

        Returns:
            List containing centerline as [(x, y), ...] in UTM coordinates.
            If bestline exists, includes that as second polyline.
        """
        lines = []
        if self._centerline_coords is not None:
            lines.append([tuple(p) for p in self._centerline_coords])
        if self.bestline_utm is not None:
            lines.append(list(self.bestline_utm))
        if not lines and self._inner_boundary_utm is not None:
            lines.append([tuple(p) for p in self._inner_boundary_utm])
        if not lines and self._outer_boundary_utm is not None:
            lines.append([tuple(p) for p in self._outer_boundary_utm])
        return lines

    @property
    def start_finish_webmerc(self) -> Optional[List[Tuple[float, float]]]:
        """Start/finish line in Web Mercator (EPSG:3857)."""
        if self.start_finish_utm is None:
            return None

        from pyproj import Transformer

        transformer = Transformer.from_crs(self.utm_zone, "EPSG:3857", always_xy=True)
        pts = np.array(self.start_finish_utm)
        xs, ys = transformer.transform(pts[:, 0], pts[:, 1])
        return list(zip(xs, ys))

    def _init_projector(self):
        """Initialize centerline projector from computed centerline."""
        if self._projector_initialized:
            return

        self._projector_initialized = True

        # Use computed centerline
        pts = self.centerline
        if pts is None or len(pts) < 2:
            return

        self._projector_points = pts

        # Calculate cumulative distance along polyline
        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        self._cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
        self._total_length = self._cumulative_dists[-1]

        # Pre-calculate segment vectors
        self._segments_vec = pts[1:] - pts[:-1]
        self._segment_lengths_sq = np.sum(self._segments_vec**2, axis=1)

    @property
    def total_length(self) -> float:
        """Total length of the centerline in meters."""
        self._init_projector()
        return self._total_length

    @property
    def has_projector(self) -> bool:
        """Check if centerline projector is available."""
        self._init_projector()
        return self._projector_points is not None and len(self._projector_points) >= 2

    @property
    def layout(self) -> "Track":
        """Backward compatibility: return self as layout."""
        return self

    @property
    def geometry(self) -> "Track":
        """Backward compatibility: return self as geometry."""
        return self

    @property
    def centerline(self) -> Optional[np.ndarray]:
        """Return centerline coordinates in UTM, computing lazily if needed."""
        if self._centerline_coords is None:
            if self._inner_boundary_utm is not None and self._outer_boundary_utm is not None:
                # Check cache first
                cache_key = (id(self._inner_boundary_utm), id(self._outer_boundary_utm))
                if cache_key in self._centerline_cache:
                    self._centerline_coords = self._centerline_cache[cache_key]
                else:
                    self._centerline_coords = compute_centerline(self._inner_boundary_utm, self._outer_boundary_utm, n_samples=512)
                    self._centerline_cache[cache_key] = self._centerline_coords
        return self._centerline_coords

    @property
    def start_finish_intersection(self) -> Optional[Dict]:
        """
        Calculate intersection of start-finish line with bestline (or centerline if no bestline).

        If lines don't intersect, finds the closest point on bestline/centerline to the SF line.

        Returns:
            Dict with keys:
                - 'bestline_distance': Distance along bestline at intersection (meters)
                - 'centerline_distance': Distance along centerline at intersection (meters)
                - 'point': Intersection/closest point in UTM coordinates
            Returns None if no start-finish line defined.
        """
        if not self.start_finish_utm:
            return None

        # Convert start-finish to Shapely LineString (use first and last point for line)
        sf_points = np.array(self.start_finish_utm)
        if len(sf_points) < 2:
            return None

        # Use first and last point to create line
        sf_line = LineString([sf_points[0], sf_points[-1]])

        # Use bestline for intersection if available, otherwise centerline
        if self.bestline_utm:
            # Use bestline for intersection
            bestline_arr = np.array(self.bestline_utm)
            reference_line = LineString(bestline_arr)
            reference_type = "bestline"
        elif self.has_projector and self._projector_points is not None:
            # Fallback to centerline
            reference_line = LineString(self._projector_points)
            reference_type = "centerline"
        else:
            return None

        # Try to find intersection first
        intersection = sf_line.intersection(reference_line)

        # If no intersection, find closest point
        if intersection.is_empty or intersection is None:
            # Find closest point on reference line to SF line
            # Project SF line midpoint onto reference line
            sf_midpoint = sf_line.interpolate(0.5, normalized=True)
            ref_distance = reference_line.project(sf_midpoint)
            point = reference_line.interpolate(ref_distance)
        else:
            # Get intersection point
            if intersection.geom_type == "Point":
                point = intersection
            elif intersection.geom_type == "MultiPoint":
                point = intersection.geoms[0]
            elif intersection.geom_type == "LineString":
                # Intersection is a line segment, take midpoint
                point = intersection.interpolate(0.5, normalized=True)
            else:
                point = intersection

        intersection_coords = np.array(point.coords)[0]

        # Calculate distance along reference line (bestline or centerline)
        ref_dist = reference_line.project(point)

        result = {
            "point": tuple(intersection_coords),
            f"{reference_type}_distance": ref_dist,
        }

        # If we used bestline, also calculate centerline distance
        if reference_type == "bestline" and self.has_projector:
            centerline_dist = self.project(intersection_coords)
            result["centerline_distance"] = centerline_dist

        # If we used centerline but bestline exists, also calculate bestline distance
        if reference_type == "centerline" and self.bestline_utm:
            bestline_line = LineString(np.array(self.bestline_utm))
            bestline_dist = bestline_line.project(point)
            result["bestline_distance"] = bestline_dist

        return result

    def project(self, point: np.ndarray) -> float:
        """
        Project point onto nearest segment and return distance along centerline.

        Args:
            point: [x, y] coordinates in UTM meters, shape (2,)

        Returns:
            Distance along centerline in meters

        Algorithm:
            1. Find closest segment on centerline
            2. Project point onto segment
            3. Return cumulative distance to projection point
        """
        self._init_projector()

        if self._projector_points is None:
            return 0.0

        # Vector from segment start to point
        v_start_point = point - self._projector_points[:-1]

        # Project v_start_point onto segment vector
        # t = dot(v_start_point, segment) / |segment|^2
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.sum(v_start_point * self._segments_vec, axis=1) / self._segment_lengths_sq
            t = np.nan_to_num(t)

        # Clamp t to [0, 1]
        t_clamped = np.clip(t, 0, 1)

        # Find closest point on each segment
        closest_points = self._projector_points[:-1] + self._segments_vec * t_clamped[:, np.newaxis]

        # Distance from query point to closest point on each segment
        dists_sq = np.sum((closest_points - point) ** 2, axis=1)

        # Find index of closest segment
        min_idx = np.argmin(dists_sq)

        # Calculate distance along centerline
        seg_len = np.sqrt(self._segment_lengths_sq[min_idx])
        dist_along = self._cumulative_dists[min_idx] + t_clamped[min_idx] * seg_len

        return float(dist_along)

    @classmethod
    def load(cls, track_dir: Path, use_bestline_for_segments: bool = True) -> "Track":
        """
        Load track from directory containing GeoJSON files.

        Generates centerline from inner/outer boundaries.
        Reads UTM zone from track_config.json (defaults to EPSG:32638).
        Loads bestline if available.

        Args:
            track_dir: Path to track directory
            use_bestline_for_segments: If True, calculate segments from bestline,
                                       otherwise from centerline (default: True)

        Returns:
            Track instance

        Required files:
            - track-inner.geojson: Inner track boundary
            - track-outer.geojson: Outer track boundary

        Optional files:
            - start-finish.geojson: Start/finish line
            - bestline.geojson: Optimal racing line
            - track_config.json: Track configuration (UTM zone, name, etc.)
        """
        from .validation import validate_track_directory

        track_dir = Path(track_dir)

        # Validate track directory
        is_valid, errors = validate_track_directory(track_dir)
        if not is_valid:
            raise ValueError(f"Invalid track directory:\n" + "\n".join(errors))

        # Load track configuration (optional UTM zone override)
        config = load_track_config(track_dir)
        utm_zone_override = config.get("utm_zone", None)

        # Load inner/outer boundaries from GeoJSON (WGS84)
        geometry_dir = track_dir / "geometry"
        inner_path = geometry_dir / "track-inner.geojson"
        outer_path = geometry_dir / "track-outer.geojson"

        logger.info("Loading boundaries from GeoJSON...")

        inner_pts_wgs84 = load_polyline_geojson(inner_path)
        outer_pts_wgs84 = load_polyline_geojson(outer_path)

        if not inner_pts_wgs84 or not outer_pts_wgs84:
            raise ValueError(f"Failed to load inner/outer boundaries from {track_dir}")

        inner_arr_wgs84 = np.array(inner_pts_wgs84)  # Shape: (N, 2) lon/lat in WGS84
        outer_arr_wgs84 = np.array(outer_pts_wgs84)  # Shape: (M, 2) lon/lat in WGS84

        # Determine UTM zone from coordinates (unless overridden)
        utm_zone = utm_zone_override or _determine_utm_zone_from_coords(inner_arr_wgs84[:, 0], inner_arr_wgs84[:, 1])
        logger.info(f"Using UTM zone: {utm_zone}")

        # Get transformer for WGS84 -> UTM
        transformer = get_transformer(WGS84_CRS, utm_zone)

        # Load start-finish line
        sf_path = geometry_dir / "start-finish.geojson"
        start_finish_utm = None

        if sf_path.exists():
            start_finish_wgs84 = load_polyline_geojson(sf_path)
            if start_finish_wgs84:
                lons, lats = zip(*start_finish_wgs84)
                xs, ys = transformer.transform(np.array(lons), np.array(lats))
                start_finish_utm = list(zip(xs, ys))
                logger.info("Loaded start-finish line")

        # Load bestline
        bestline_path = geometry_dir / "bestline.geojson"
        bestline_utm = None
        bestline_utm_arr = None

        if bestline_path.exists():
            bestline_wgs84 = load_polyline_geojson(bestline_path)
            if bestline_wgs84:
                bestline_arr_wgs84 = np.array(bestline_wgs84)
                bestline_utm_arr = transform_coordinates(bestline_arr_wgs84, WGS84_CRS, utm_zone)
                bestline_utm = list(map(tuple, bestline_utm_arr))
                logger.info(f"Loaded bestline with {len(bestline_wgs84)} points")

        # Try to load centerline from file (optional, for faster initialization)
        centerline_path = geometry_dir / "centerline.geojson"
        centerline_utm = None

        if centerline_path.exists():
            try:
                centerline_wgs84 = load_polyline_geojson(centerline_path)
                if centerline_wgs84:
                    centerline_wgs84_arr = np.array(centerline_wgs84)
                    centerline_utm = transform_coordinates(centerline_wgs84_arr, WGS84_CRS, utm_zone)
                    logger.info(f"Loaded centerline from file with {len(centerline_utm)} points")
            except Exception as e:
                logger.warning(f"Failed to load centerline.geojson: {e}")
                logger.info("Will compute centerline from boundaries when needed")

        # Determine reference line for segmentation
        use_bestline = use_bestline_for_segments and bestline_utm_arr is not None

        # Generate segments from reference line (bestline or centerline)
        # If bestline exists, create averaged trajectory for segmentation
        if use_bestline and centerline_utm is not None:
            # Resample both lines to same number of points for averaging
            from .utils import resample_linestring

            n_samples = min(len(bestline_utm), len(centerline_utm))
            bestline_resampled = resample_linestring(np.array(bestline_utm), n_samples)
            centerline_resampled = resample_linestring(centerline_utm, n_samples)

            # Create averaged trajectory
            averaged_polyline = (bestline_resampled + centerline_resampled) / 2.0
            averaged_polylines = [list(map(tuple, averaged_polyline))]

            logger.info(f"Using averaged bestline+centerline for segmentation ({n_samples} points)")

            # Calculate adaptive threshold from averaged trajectory
            polyline_points = averaged_polyline

            # Calculate headings
            headings = []
            for i in range(len(polyline_points) - 1):
                p1 = polyline_points[i]
                p2 = polyline_points[i + 1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                rads = np.arctan2(dy, dx)
                deg = np.degrees(rads)
                headings.append(deg)

            # Calculate curvature (change in heading)
            curvatures = []
            for i in range(len(headings) - 1):
                diff = ((headings[i + 1] - headings[i]) + 180) % 360 - 180
                curvatures.append(abs(diff))

            curvatures = np.array(curvatures)

            # Use median as threshold - adaptive to track characteristics
            median_curvature = np.median(curvatures)
            mean_curvature = np.mean(curvatures)

            # Threshold is mean of median and mean - robust to outliers
            turn_threshold = (median_curvature + mean_curvature) / 2

            logger.debug(f"Averaged curvature - Median: {median_curvature:.3f}, Mean: {mean_curvature:.3f}")
            logger.debug(f"Using adaptive threshold: {turn_threshold:.3f}")

            segments = segment_track(averaged_polylines, turn_threshold=turn_threshold)
            logger.info(f"Generated {len(segments)} segments (straights/turns) from averaged trajectory")
        else:
            # Use fixed threshold for centerline
            from .constants import CENTERLINE_TURN_THRESHOLD

            turn_threshold = CENTERLINE_TURN_THRESHOLD

            # Compute centerline if not loaded from file
            if centerline_utm is None:
                centerline_utm = compute_centerline(
                    np.column_stack((transformer.transform(inner_arr_wgs84[:, 0], inner_arr_wgs84[:, 1]))),
                    np.column_stack((transformer.transform(outer_arr_wgs84[:, 0], outer_arr_wgs84[:, 1]))),
                    n_samples=512,
                )
                logger.info(f"Generated centerline with {len(centerline_utm)} points")

            reference_line = [list(map(tuple, centerline_utm))]
            logger.debug(f"Using centerline turn threshold: {turn_threshold} (centerline)")

            segments = segment_track(reference_line, turn_threshold=turn_threshold)
            logger.info(f"Generated {len(segments)} segments (straights/turns) from centerline")

        track = cls(
            inner_boundary_wgs84=inner_arr_wgs84,
            outer_boundary_wgs84=outer_arr_wgs84,
            utm_zone=utm_zone,
            segments=segments,
            start_finish_utm=start_finish_utm,
            bestline_utm=bestline_utm,
        )

        # Pre-set the computed centerline to avoid recomputation
        if centerline_utm is not None:
            track._centerline_coords = centerline_utm

        # Calculate and log start-finish intersection
        if track.start_finish_intersection:
            intersection = track.start_finish_intersection
            logger.info(f"Start-finish intersection at {intersection['centerline_distance']:.1f}m")
            if "bestline_distance" in intersection:
                logger.info(f"Bestline crossing at {intersection['bestline_distance']:.1f}m")

        return track

    def load_bestline(self, directory: Path) -> bool:
        """
        Load bestline from a specific directory (not the track directory).

        Args:
            directory: Directory to load bestline.geojson from

        Returns:
            True if bestline was loaded, False otherwise
        """
        from .utils import load_polyline_geojson

        bestline_path = Path(directory) / "bestline.geojson"
        if not bestline_path.exists():
            logger.debug(f"No bestline found at {bestline_path}")
            return False

        bestline_wgs84 = load_polyline_geojson(bestline_path)
        if bestline_wgs84:
            bestline_arr_wgs84 = np.array(bestline_wgs84)
            transformer = get_transformer(WGS84_CRS, self.utm_zone)
            self.bestline_utm = list(map(tuple, transformer.transform(bestline_arr_wgs84[:, 0], bestline_arr_wgs84[:, 1])))
            logger.info(f"Loaded bestline with {len(self.bestline_utm)} points from {directory}")
            return True
        return False

    def set_bestline_from_gps(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        n_samples: int = 512,
    ) -> None:
        """
        Set bestline from GPS coordinates.

        Args:
            lons: Longitude values in degrees, shape (N,)
            lats: Latitude values in degrees, shape (N,)
            n_samples: Number of samples for resampling
        """
        import json
        from .utils import resample_linestring, get_transformer

        # Filter valid coordinates
        valid_mask = ~(np.isnan(lons) | np.isnan(lats))
        valid_lons = lons[valid_mask]
        valid_lats = lats[valid_mask]

        if len(valid_lons) < 2:
            logger.warning("Not enough valid GPS points to create bestline")
            return

        # Transform to UTM
        transformer = get_transformer(WGS84_CRS, self.utm_zone)
        xs, ys = transformer.transform(valid_lons, valid_lats)
        bestline_utm_arr = np.column_stack([xs, ys])

        # Resample to specified number of points
        bestline_resampled = resample_linestring(bestline_utm_arr, n_samples)
        self.bestline_utm = list(map(tuple, bestline_resampled))
        logger.info(f"Set bestline from GPS with {len(self.bestline_utm)} points")

    def save_bestline(self, directory: Path) -> None:
        """
        Save bestline to a specific directory as bestline.geojson.

        Args:
            directory: Directory to save bestline.geojson to
        """
        import json
        from .utils import get_transformer

        if self.bestline_utm is None:
            logger.warning("No bestline to save")
            return

        # Transform back to WGS84
        transformer = get_transformer(self.utm_zone, WGS84_CRS)
        pts = np.array(self.bestline_utm)
        lons, lats = transformer.transform(pts[:, 0], pts[:, 1])

        # Create GeoJSON Feature
        coordinates = [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]
        # Close the loop if needed
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        geojson = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": coordinates}}],
        }

        output_path = Path(directory) / "bestline.geojson"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)
            logger.info(f"Saved bestline to {output_path}")
