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
    Unified track representation combining layout, sectors, and centerline.

    Attributes:
        bounds: Bounding box (xmin, xmax, ymin, ymax) in UTM (property, computed from centerline)
        segments: List of track segments (straights/turns)
        sectors_utm: Named sector lines in UTM (e.g. {"SF": [...], "S1": [...], "S2": [...]})
        bestline_utm: Bestline (optimal racing line) in UTM
        bestline_wgs84: Bestline in WGS84 (property, computed from UTM)
        utm_zone: UTM zone EPSG code (auto-determined from WGS84 coordinates)
        total_length: Total track length in meters (property)
        centerline: Centerline coordinates in UTM (property, computed from boundaries)

    Example:
        >>> track = Track.load("/path/to/track")
        >>> distance = track.project(point_utm)
        >>> sf_wgs84 = track.get_sector_wgs84("SF")
        >>> print(f"Sectors: {list(track.sectors_utm.keys())}")
    """

    def __init__(
        self,
        inner_boundary_wgs84: Optional[np.ndarray] = None,
        outer_boundary_wgs84: Optional[np.ndarray] = None,
        utm_zone: Optional[str] = None,
        segments: Optional[List[Dict]] = None,
        sectors_utm: Optional[Dict[str, List[Tuple[float, float]]]] = None,
        bestline_utm: Optional[List[Tuple[float, float]]] = None,
        name: str = "",
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

        # Named sector lines in UTM (e.g. {"SF": [...], "S1": [...], "S2": [...]})
        self.sectors_utm: Dict[str, List[Tuple[float, float]]] = sectors_utm or {}

        # Bestline (UTM only, WGS84 computed dynamically)
        self.bestline_utm = bestline_utm
        self.bestline_alt: list[float] | None = None

        # Track name from config
        self.name = name

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

    def get_sector_wgs84(self, name: str) -> Optional[List[Tuple[float, float]]]:
        """Get sector line coordinates in WGS84 by name."""
        pts_utm = self.sectors_utm.get(name)
        if not pts_utm:
            return None
        transformer = self._get_transformer_to_wgs84()
        pts = np.array(pts_utm)
        lons, lats = transformer.transform(pts[:, 0], pts[:, 1])
        return list(zip(lons, lats))

    @property
    def start_finish_wgs84(self) -> Optional[List[Tuple[float, float]]]:
        """Start/finish line coordinates in WGS84 (shortcut for sectors_utm["SF"])."""
        return self.get_sector_wgs84("SF")

    @property
    def start_finish_utm(self) -> Optional[List[Tuple[float, float]]]:
        """Start/finish line in UTM (shortcut for sectors_utm["SF"])."""
        return self.sectors_utm.get("SF")

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
        sf_utm = self.sectors_utm.get("SF")
        if not sf_utm:
            return None

        from pyproj import Transformer

        transformer = Transformer.from_crs(self.utm_zone, "EPSG:3857", always_xy=True)
        pts = np.array(sf_utm)
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
        if not self.sectors_utm.get("SF"):
            return None

        # Convert start-finish to Shapely LineString (use first and last point for line)
        sf_points = np.array(self.sectors_utm["SF"])
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
            - sectors.geojson: Named sector lines (SF, S1, S2, ...)
            - start-finish.geojson: Legacy start/finish line (fallback if no sectors.geojson)
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
        track_name = config.get("name", "")

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

        # Load sector lines (SF, S1, S2, ...)
        sectors_utm: Dict[str, List[Tuple[float, float]]] = {}
        sectors_path = geometry_dir / "sectors.geojson"
        sf_path = geometry_dir / "start-finish.geojson"

        if sectors_path.exists():
            import json as _json
            with open(sectors_path) as f:
                sectors_geojson = _json.load(f)
            for feature in sectors_geojson.get("features", []):
                sector_name = feature.get("properties", {}).get("name")
                coords = feature.get("geometry", {}).get("coordinates", [])
                if sector_name and coords:
                    lons, lats = zip(*coords)
                    xs, ys = transformer.transform(np.array(lons), np.array(lats))
                    sectors_utm[sector_name] = list(zip(xs, ys))
            logger.info(f"Loaded {len(sectors_utm)} sectors: {list(sectors_utm.keys())}")
        elif sf_path.exists():
            # Fallback: legacy start-finish.geojson
            start_finish_wgs84 = load_polyline_geojson(sf_path)
            if start_finish_wgs84:
                lons, lats = zip(*start_finish_wgs84)
                xs, ys = transformer.transform(np.array(lons), np.array(lats))
                sectors_utm["SF"] = list(zip(xs, ys))
                logger.info("Loaded start-finish line (legacy fallback)")

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
            sectors_utm=sectors_utm,
            bestline_utm=bestline_utm,
            name=track_name,
        )

        # Pre-set the computed centerline to avoid recomputation
        if centerline_utm is not None:
            track._centerline_coords = centerline_utm

        # Calculate and log start-finish intersection
        intersection = track.start_finish_intersection
        if intersection:
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
            xs, ys = transformer.transform(bestline_arr_wgs84[:, 0], bestline_arr_wgs84[:, 1])
            self.bestline_utm = list(map(tuple, np.column_stack([xs, ys])))
            logger.info(f"Loaded bestline with {len(self.bestline_utm)} points from {directory}")
            return True
        return False

    def set_bestline_from_gps(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        alts: np.ndarray | None = None,
        n_samples: int = 512,
    ) -> None:
        """
        Set bestline from GPS coordinates.

        Args:
            lons: Longitude values in degrees, shape (N,)
            lats: Latitude values in degrees, shape (N,)
            alts: Altitude values in meters, shape (N,) (optional)
            n_samples: Number of samples for resampling
        """
        from .utils import resample_linestring, get_transformer

        # Filter valid coordinates
        valid_mask = ~(np.isnan(lons) | np.isnan(lats))
        valid_lons = lons[valid_mask]
        valid_lats = lats[valid_mask]
        valid_alts = alts[valid_mask] if alts is not None else None

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

        # Resample altitude along the same arc-length parameterization
        if valid_alts is not None and np.any(valid_alts != 0):
            arc_orig = np.zeros(len(bestline_utm_arr))
            for i in range(1, len(arc_orig)):
                arc_orig[i] = arc_orig[i - 1] + np.linalg.norm(bestline_utm_arr[i] - bestline_utm_arr[i - 1])
            arc_new = np.zeros(len(bestline_resampled))
            for i in range(1, len(arc_new)):
                arc_new[i] = arc_new[i - 1] + np.linalg.norm(bestline_resampled[i] - bestline_resampled[i - 1])
            from scipy.signal import savgol_filter
            alt_resampled = np.interp(arc_new, arc_orig, valid_alts)
            # Heavy smoothing for noisy GPS altitude
            window = min(51, len(alt_resampled) // 4 * 2 + 1)
            if window >= 5:
                alt_resampled = savgol_filter(alt_resampled, window, polyorder=2, mode="wrap")
            self.bestline_alt = alt_resampled.tolist()
        else:
            self.bestline_alt = None

        logger.info(f"Set bestline from GPS with {len(self.bestline_utm)} points" +
                     (f" (alt {min(self.bestline_alt):.0f}-{max(self.bestline_alt):.0f}m)" if self.bestline_alt else ""))

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

    def _utm_to_wgs84(self, pts_utm: np.ndarray) -> list[tuple[float, float]]:
        """Convert UTM coordinates to WGS84 (lon, lat) pairs."""
        transformer = self._get_transformer_to_wgs84()
        lons, lats = transformer.transform(pts_utm[:, 0], pts_utm[:, 1])
        return list(zip(lons.tolist(), lats.tolist()))

    def export_gpx(self, output_dir: Path) -> None:
        """Export each track polyline as a separate GPX file into output_dir."""
        import gpxpy

        layers: list[tuple[str, list[tuple[float, float]] | None]] = [
            ("inner-boundary", self._utm_to_wgs84(self._inner_boundary_utm) if self._inner_boundary_utm is not None else None),
            ("outer-boundary", self._utm_to_wgs84(self._outer_boundary_utm) if self._outer_boundary_utm is not None else None),
            ("centerline", self._utm_to_wgs84(self.centerline) if self.centerline is not None else None),
            ("bestline", self.bestline_wgs84),
        ]
        # Export each sector line
        for sector_name in self.sectors_utm:
            layers.append((f"sector-{sector_name}", self.get_sector_wgs84(sector_name)))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for name, coords in layers:
            if not coords:
                continue
            gpx = gpxpy.gpx.GPX()
            track = gpxpy.gpx.GPXTrack(name=name)
            gpx.tracks.append(track)
            segment = gpxpy.gpx.GPXTrackSegment()
            track.segments.append(segment)
            # Add altitude and time for bestline if available
            alts = self.bestline_alt if name == "bestline" and self.bestline_alt else None
            from datetime import datetime, timedelta
            base_time = datetime(2026, 1, 1)
            for i, (lon, lat) in enumerate(coords):
                ele = round(alts[i], 1) if alts and i < len(alts) else None
                t = base_time + timedelta(seconds=i * 0.05) if name == "bestline" else None
                segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, elevation=ele, time=t))

            path = output_dir / f"{name}.gpx"
            path.write_text(gpx.to_xml(), encoding="utf-8")
            exported += 1

        # Add start-finish intersection as waypoint
        sf = self.start_finish_intersection
        if sf:
            gpx_wp = gpxpy.gpx.GPX()
            transformer = self._get_transformer_to_wgs84()
            pt = sf["point"]
            lon, lat = transformer.transform(pt[0], pt[1])
            gpx_wp.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=float(lat), longitude=float(lon), name="SF"))
            (output_dir / "sector-SF-point.gpx").write_text(gpx_wp.to_xml(), encoding="utf-8")
            exported += 1

        print(f"[Track] Exported {exported} GPX files to {output_dir}")

    def export_kml(self, output_path: Path) -> None:
        """Export track geometry as a single KML file for Google Earth.

        Includes: boundaries, centerline, bestline, sector lines,
        and sector-bestline intersection points.
        """
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom.minidom import parseString

        kml = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        doc = SubElement(kml, "Document")
        SubElement(doc, "name").text = self.name or "Track"

        def _add_style(style_id: str, color: str, width: int = 3):
            style = SubElement(doc, "Style", id=style_id)
            ls = SubElement(style, "LineStyle")
            SubElement(ls, "color").text = color
            SubElement(ls, "width").text = str(width)

        def _add_line(name: str, coords_wgs84: list, style_id: str):
            pm = SubElement(doc, "Placemark")
            SubElement(pm, "name").text = name
            SubElement(pm, "styleUrl").text = f"#{style_id}"
            ls = SubElement(pm, "LineString")
            coord_str = " ".join(f"{lon},{lat},0" for lon, lat in coords_wgs84)
            SubElement(ls, "coordinates").text = coord_str

        def _add_point(name: str, lon: float, lat: float, description: str = ""):
            pm = SubElement(doc, "Placemark")
            SubElement(pm, "name").text = name
            if description:
                SubElement(pm, "description").text = description
            pt = SubElement(pm, "Point")
            SubElement(pt, "coordinates").text = f"{lon},{lat},0"

        # Styles (KML colors are aaBBGGRR)
        _add_style("boundary", "ffff0000", 2)       # blue
        _add_style("centerline", "ff0000ff", 2)      # red
        _add_style("bestline", "ff00ff00", 3)         # green
        _add_style("sector-SF", "ff00ffff", 4)        # yellow
        _add_style("sector-S1", "ff00a5ff", 4)        # orange
        _add_style("sector-S2", "ff00ff00", 4)        # lime
        _add_style("sector-default", "ffffffff", 4)   # white

        # Boundaries
        if self._inner_boundary_utm is not None:
            _add_line("Inner boundary", self._utm_to_wgs84(self._inner_boundary_utm), "boundary")
        if self._outer_boundary_utm is not None:
            _add_line("Outer boundary", self._utm_to_wgs84(self._outer_boundary_utm), "boundary")

        # Centerline
        if self.centerline is not None:
            _add_line("Centerline", self._utm_to_wgs84(self.centerline), "centerline")

        # Bestline
        if self.bestline_wgs84:
            _add_line("Bestline", self.bestline_wgs84, "bestline")

        # Sector lines
        sector_style_map = {"SF": "sector-SF", "S1": "sector-S1", "S2": "sector-S2"}
        for sector_name in self.sectors_utm:
            coords = self.get_sector_wgs84(sector_name)
            if coords:
                style = sector_style_map.get(sector_name, "sector-default")
                _add_line(sector_name, coords, style)

        # Sector-bestline intersection points
        if self.bestline_utm and self.sectors_utm:
            bestline_line = LineString(self.bestline_utm)
            transformer = self._get_transformer_to_wgs84()
            for sector_name, sector_pts in self.sectors_utm.items():
                sector_line = LineString(sector_pts)
                ix = sector_line.intersection(bestline_line)
                if ix.is_empty:
                    mid = sector_line.interpolate(0.5, normalized=True)
                    proj_pt = bestline_line.interpolate(bestline_line.project(mid))
                elif ix.geom_type == "Point":
                    proj_pt = ix
                else:
                    from shapely.geometry import Point as ShapelyPoint
                    proj_pt = ix.geoms[0] if hasattr(ix, 'geoms') else ShapelyPoint(ix.coords[0])
                dist_m = bestline_line.project(proj_pt)
                lon, lat = transformer.transform(proj_pt.x, proj_pt.y)
                _add_point(
                    f"{sector_name} x bestline",
                    float(lon), float(lat),
                    f"Distance: {dist_m:.1f}m\nLat: {lat:.6f}\nLon: {lon:.6f}",
                )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        xml_str = parseString(tostring(kml, encoding="unicode")).toprettyxml(indent="  ")
        # Remove extra XML declaration from minidom
        lines = xml_str.split("\n")
        xml_str = "\n".join(lines[1:]) if lines[0].startswith("<?xml") else xml_str
        output_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str, encoding="utf-8")
        print(f"[Track] Exported KML to {output_path}")

    @staticmethod
    def import_sectors_kml(kml_path: Path) -> dict[str, list[tuple[float, float]]]:
        """Import named sector lines from a KML file.

        Returns dict of {name: [(lon, lat), ...]} in WGS84.
        """
        from xml.etree.ElementTree import parse

        kml_path = Path(kml_path)
        tree = parse(kml_path)
        root = tree.getroot()

        # Handle KML namespace
        ns = ""
        tag = root.tag
        if tag.startswith("{"):
            ns = tag[: tag.index("}") + 1]

        sectors: dict[str, list[tuple[float, float]]] = {}
        for placemark in root.iter(f"{ns}Placemark"):
            name_el = placemark.find(f"{ns}name")
            if name_el is None or not name_el.text:
                continue
            name = name_el.text.strip()

            coords_el = placemark.find(f".//{ns}coordinates")
            if coords_el is None or not coords_el.text:
                continue

            points = []
            for part in coords_el.text.strip().split():
                vals = part.split(",")
                if len(vals) >= 2:
                    lon, lat = float(vals[0]), float(vals[1])
                    points.append((lon, lat))

            if len(points) >= 2:
                sectors[name] = points

        return sectors

    def load_sectors_from_kml(self, kml_path: Path) -> None:
        """Load sector lines from KML and set them on this track (transforms to UTM)."""
        from .utils import get_transformer
        from .constants import WGS84_CRS

        sectors_wgs84 = self.import_sectors_kml(kml_path)
        transformer = get_transformer(WGS84_CRS, self.utm_zone)

        for name, coords in sectors_wgs84.items():
            lons, lats = zip(*coords)
            xs, ys = transformer.transform(np.array(lons), np.array(lats))
            self.sectors_utm[name] = list(zip(xs, ys))

        logger.info(f"Loaded {len(sectors_wgs84)} sectors from KML: {list(sectors_wgs84.keys())}")

    def export_ztracks(
        self,
        output_path: Path,
        venue_name: str = "",
        track_name: str = "",
        country_code: str = "  ",
        timezone: str = "",
    ) -> None:
        """Export track as AIM RaceStudio3 .ztracks file.

        See experiments/aim-ztracks-format/AIM_ZTRACKS_FORMAT.md for format docs.
        """
        import random
        import string
        import struct
        import zipfile
        from io import BytesIO

        if not self.bestline_utm:
            logger.warning("No bestline to export")
            return

        # Bestline to WGS84
        bestline_wgs84 = self.bestline_wgs84
        if not bestline_wgs84:
            return

        # Sector intersection points on bestline
        sector_pts_wgs84: dict[str, tuple[float, float]] = {}
        bestline_line = LineString(self.bestline_utm)
        transformer = self._get_transformer_to_wgs84()
        for sname, spts in self.sectors_utm.items():
            sline = LineString(spts)
            ix = sline.intersection(bestline_line)
            if not ix.is_empty:
                pt = ix if ix.geom_type == "Point" else (ix.geoms[0] if hasattr(ix, 'geoms') else ix)
                lon, lat = transformer.transform(pt.x, pt.y)
                sector_pts_wgs84[sname] = (float(lat), float(lon))
            else:
                mid = sline.interpolate(0.5, normalized=True)
                proj = bestline_line.interpolate(bestline_line.project(mid))
                lon, lat = transformer.transform(proj.x, proj.y)
                sector_pts_wgs84[sname] = (float(lat), float(lon))

        def _coord_i32(deg: float) -> int:
            return int(round(deg * 1e7))

        def _chunk(tag: bytes, payload: bytes) -> bytes:
            """Build a TKK chunk: header + data + footer."""
            size = len(payload)
            header = b'<h' + tag + struct.pack('<I', size) + b'\x00>'
            checksum = sum(payload) & 0xFFFF
            footer = b'<' + tag + struct.pack('<BB', checksum & 0xFF, (checksum >> 8) & 0xFF) + b'>'
            return header + payload + footer

        # Load metadata from track_config if available
        import json as _json
        _config = {}
        _cfg_path = output_path.parent.parent.parent / "track_config.json" if output_path else None
        if _cfg_path and Path(_cfg_path).exists():
            _config = _json.loads(Path(_cfg_path).read_text())

        file_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        vname = venue_name or self.name or _config.get("name", "Track")
        tname = track_name or _config.get("full_name", vname)
        country_code = country_code.strip() or _config.get("country", "")
        timezone = timezone or _config.get("timezone", "")
        bestline_length = bestline_line.length

        # Center point
        pts_arr = np.array(bestline_wgs84)
        center_lon = float(pts_arr[:, 0].mean())
        center_lat = float(pts_arr[:, 1].mean())

        # === Ptkk (268 bytes) ===
        ptkk = bytearray(268)
        # Name at 0-255 (leave empty like reference)
        # File ID at 256
        # ID at offset 260 (4 bytes padding before, matching reference)
        ptkk[260:260+len(file_id)] = file_id.encode('ascii')

        # === Vnfo (476 bytes) ===
        vnfo = bytearray(476)
        vname_bytes = vname.encode('utf-8')[:23]
        vnfo[:len(vname_bytes)] = vname_bytes
        cc = country_code.encode('ascii')[:2].ljust(2)
        vnfo[28:30] = cc
        struct.pack_into('<f', vnfo, 44, bestline_length)
        struct.pack_into('<i', vnfo, 48, _coord_i32(center_lat))
        struct.pack_into('<i', vnfo, 52, _coord_i32(center_lon))
        struct.pack_into('<I', vnfo, 60, 50)
        sector_names = [s for s in self.sectors_utm if s != "SF"]
        struct.pack_into('<I', vnfo, 72, len(sector_names))
        struct.pack_into('<I', vnfo, 76, 0x00040000)
        # SF at 368
        if "SF" in sector_pts_wgs84:
            lat, lon = sector_pts_wgs84["SF"]
            struct.pack_into('<i', vnfo, 368, _coord_i32(lat))
            struct.pack_into('<i', vnfo, 372, _coord_i32(lon))
        # S1 at 384, S2 at 400
        for idx, sname in enumerate(["S1", "S2"]):
            if sname in sector_pts_wgs84:
                lat, lon = sector_pts_wgs84[sname]
                struct.pack_into('<i', vnfo, 384 + idx * 16, _coord_i32(lat))
                struct.pack_into('<i', vnfo, 388 + idx * 16, _coord_i32(lon))

        # === V_sw (256 bytes) ===
        vsw = bytearray(256)
        tname_bytes = tname.encode('utf-8')[:255]
        vsw[:len(tname_bytes)] = tname_bytes

        # === Vidx (8 bytes) ===
        vidx = bytearray(8)

        # === pts (N x 12 bytes) ===
        pts_payload = bytearray()
        alts = self.bestline_alt or [0.0] * len(bestline_wgs84)
        for i, (lon, lat) in enumerate(bestline_wgs84):
            alt = alts[i] if i < len(alts) else 0.0
            pts_payload += struct.pack('<iii', _coord_i32(lat), _coord_i32(lon), int(round(alt * 1000)))

        # === zots (408 bytes) ===
        zots = bytearray(408)
        tz = timezone or "UTC"
        tz_bytes = tz.encode('utf-8')[:407]
        zots[:len(tz_bytes)] = tz_bytes

        # === srfs (4 bytes) ===
        srfs = struct.pack('<I', 1)

        # === lgo (minimal — empty filename + no JPEG) ===
        lgo_filename = f"{file_id}.logo.jpg\x00".encode('ascii')
        # Minimal 1x1 white JPEG
        minimal_jpeg = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
            0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
            0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
            0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x7B, 0x94,
            0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xFF, 0xD9,
        ])
        lgo_payload = lgo_filename + minimal_jpeg

        # === plus (XML from track_config metadata) ===
        import json as _json2
        config = {}
        config_path = output_path.parent.parent.parent / "track_config.json"
        if config_path.exists():
            config = _json2.loads(config_path.read_text())
        xml = '<?xml version="1.0" encoding="utf-8"?>\n<DplRoot>\n  <a>\n'
        xml += f'    <p n="Cty">{config.get("city", "")}</p>\n'
        xml += f'    <p n="Adr">{config.get("address", "")}</p>\n'
        if config.get("postal_code"):
            xml += f'    <p n="Pco">{config["postal_code"]}</p>\n'
        if config.get("phone"):
            xml += f'    <p n="Tel">{config["phone"]}</p>\n'
        if config.get("url"):
            xml += f'    <p n="Url">{config["url"]}</p>\n'
        xml += '  </a>\n</DplRoot>\n'
        plus_payload = xml.encode('utf-8')

        # Build TKK
        tkk = bytearray()
        tkk += _chunk(b'Ptkk', bytes(ptkk))
        tkk += _chunk(b'Vnfo', bytes(vnfo))
        tkk += _chunk(b'V_sw', bytes(vsw))
        tkk += _chunk(b'Vidx', bytes(vidx))
        tkk += _chunk(b'pts\x00', bytes(pts_payload))
        tkk += _chunk(b'zots', bytes(zots))
        tkk += _chunk(b'srfs', bytes(srfs))
        tkk += _chunk(b'lgo\x00', bytes(lgo_payload))
        tkk += _chunk(b'plus', bytes(plus_payload))

        # Write ZIP
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f'{file_id}.tkk', bytes(tkk))

        print(f"[Track] Exported ztracks to {output_path} ({len(bestline_wgs84)} pts, {len(sector_pts_wgs84)} sectors)")

    def export_alfano(self, output_path: Path, track_name: str = "TRACK") -> None:
        """Export track as .trackALFANO binary for Alfano GPS devices.

        Format: 8192 bytes fixed. Header + points (11 bytes each) + 0xFF padding.
        See racing_tools/track/data/RIMSportKarting/export/alfano/decode_alfano.py
        """
        import struct

        if not self.bestline_utm:
            logger.warning("No bestline to export")
            return

        bestline_wgs84 = self.bestline_wgs84
        if not bestline_wgs84:
            return

        FILE_SIZE = 8192
        RECORD_SIZE = 11
        DATA_OFFSET = 0x7B
        COORD_SCALE = 1_000_000

        buf = bytearray(b'\xff' * FILE_SIZE)

        # Magic
        buf[0:5] = b'*P* *'

        # @0x32-0x33: firmware (observed values)
        buf[0x32] = 5
        buf[0x33] = 0xDE

        # @0x34: device_id (use 0)
        struct.pack_into('<H', buf, 0x34, 0)

        # @0x36-0x3B: flags
        buf[0x36:0x3C] = b'\xff\xff\xff\xff\xff\x03'

        # @0x40-0x44: track name (5 chars, space-padded)
        name = track_name.upper()[:5].ljust(5)
        buf[0x40:0x45] = name.encode('ascii')

        # @0x47-0x48: hemisphere
        pts_arr = np.array(bestline_wgs84)
        center_lat = float(pts_arr[:, 1].mean())
        center_lon = float(pts_arr[:, 0].mean())
        ns = 'N' if center_lat >= 0 else 'S'
        ew = 'E' if center_lon >= 0 else 'W'
        buf[0x47:0x49] = f'{ns}{ew}'.encode('ascii')

        # @0x50-0x57: center coordinates (microdegrees)
        struct.pack_into('<i', buf, 0x50, int(round(center_lat * COORD_SCALE)))
        struct.pack_into('<i', buf, 0x54, int(round(center_lon * COORD_SCALE)))

        # @0x58: track length (meters, uint16)
        bestline_length = LineString(self.bestline_utm).length
        struct.pack_into('<H', buf, 0x58, min(65535, int(round(bestline_length))))

        # Max points that fit
        max_points = (FILE_SIZE - DATA_OFFSET) // RECORD_SIZE  # = 738

        # Downsample if needed
        n_pts = len(bestline_wgs84)
        if n_pts > max_points:
            step = n_pts / max_points
            indices = [int(i * step) for i in range(max_points)]
        else:
            indices = list(range(n_pts))

        # Data section size
        data_len = len(indices) * RECORD_SIZE
        buf[0x78:0x7B] = data_len.to_bytes(3, 'little')

        # Write points
        alts = self.bestline_alt or [0.0] * n_pts
        for i, idx in enumerate(indices):
            lon, lat = bestline_wgs84[idx]
            alt = alts[idx] if idx < len(alts) else 0.0
            off = DATA_OFFSET + i * RECORD_SIZE

            struct.pack_into('<i', buf, off, int(round(lat * COORD_SCALE)))
            struct.pack_into('<i', buf, off + 4, int(round(lon * COORD_SCALE)))
            struct.pack_into('<H', buf, off + 8, max(0, min(65535, int(round(alt)))))
            buf[off + 10] = 100  # speed_raw (no telemetry speed)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bytes(buf))
        print(f"[Track] Exported Alfano track to {output_path} ({len(indices)} pts)")

    def save_config(self, track_dir: Path) -> None:
        """Save track_config.json with computed metadata including SF intersection."""
        import json

        config_path = Path(track_dir) / "track_config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())

        config["utm_zone"] = self.utm_zone

        sf = self.start_finish_intersection
        if sf:
            transformer = self._get_transformer_to_wgs84()
            pt = sf["point"]
            lon, lat = transformer.transform(pt[0], pt[1])
            config["start_finish_point"] = {
                "lat": round(float(lat), 10),
                "lon": round(float(lon), 10),
                "utm": [round(float(pt[0]), 3), round(float(pt[1]), 3)],
                "bestline_distance_m": round(sf.get("bestline_distance", 0), 3),
                "centerline_distance_m": round(sf.get("centerline_distance", 0), 3),
            }

        # Sector distances along bestline
        if self.bestline_utm and self.sectors_utm:
            bestline_line = LineString(self.bestline_utm)
            transformer = self._get_transformer_to_wgs84()
            sectors_info = {}
            for sname, spts in self.sectors_utm.items():
                sector_line = LineString(spts)
                ix = sector_line.intersection(bestline_line)
                if not ix.is_empty:
                    pt = ix if ix.geom_type == "Point" else (ix.geoms[0] if hasattr(ix, 'geoms') else ix)
                    dist = bestline_line.project(pt)
                    lon, lat = transformer.transform(pt.x, pt.y)
                else:
                    mid = sector_line.interpolate(0.5, normalized=True)
                    proj = bestline_line.interpolate(bestline_line.project(mid))
                    dist = bestline_line.project(proj)
                    lon, lat = transformer.transform(proj.x, proj.y)
                sectors_info[sname] = {
                    "lat": round(float(lat), 7),
                    "lon": round(float(lon), 7),
                    "bestline_distance_m": round(float(dist), 1),
                }
            config["sectors"] = sectors_info

        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"[Track] Saved config to {config_path}")
