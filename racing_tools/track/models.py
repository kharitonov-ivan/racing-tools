import json
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Use UTM Zone 38N for Georgia - accurate scaling (vs Web Mercator which distorts ~35%)
WGS84_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32638", always_xy=True)
# Keep Web Mercator for backward compatibility where needed
WGS84_TO_WEBMERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def normalize_angle(angle):
    """Normalize angle to -180 to 180."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def calculate_heading(p1, p2):
    """Calculate heading between two points in degrees."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = np.arctan2(dy, dx)
    deg = np.degrees(rads)
    return deg


def load_polyline_geojson(path: Path) -> Optional[List[Tuple[float, float]]]:
    """Load polyline from GeoJSON file."""
    if not path.is_file():
        return None
    gdf = gpd.read_file(path)
    if gdf.empty:
        return None
    # Get first geometry
    geom = gdf.geometry.iloc[0]
    if geom is None:
        return None
    # Handle MultiLineString by taking first part
    if geom.geom_type == 'MultiLineString':
        geom = geom.geoms[0]
    # Extract coordinates (ignore Z if present)
    coords = list(geom.coords)
    return [(c[0], c[1]) for c in coords]


def resample_linestring(coords: np.ndarray, n_samples: int, tolerance: float = 1e-3) -> np.ndarray:
    """
    Resample closed loop uniformly by arc length.
    Removes duplicate points, enforces a closed loop, samples equally spaced points.
    """
    from scipy.spatial import cKDTree
    
    # Remove duplicates
    unique_coords = [coords[0]]
    for i in range(1, len(coords)):
        dist = np.linalg.norm(coords[i] - unique_coords[-1])
        if dist > tolerance:
            unique_coords.append(coords[i])
    
    if len(unique_coords) > 1:
        dist_to_first = np.linalg.norm(unique_coords[-1] - unique_coords[0])
        if dist_to_first < tolerance:
            unique_coords = unique_coords[:-1]
    
    coords = np.array(unique_coords)
    
    if len(coords) < 3:
        return coords
    
    # Calculate segments
    segments = []
    total = 0.0
    for i in range(len(coords)):
        p0 = coords[i]
        p1 = coords[(i + 1) % len(coords)]
        d = np.linalg.norm(p1 - p0)
        segments.append((p0, p1, d))
        total += d
    
    if total < 1e-6:
        return coords
    
    # Resample
    result = []
    step = total / n_samples
    acc = 0.0
    si = 0
    
    for k in range(n_samples):
        target = k * step
        while target > acc + segments[si][2] and segments[si][2] > 0.0:
            acc += segments[si][2]
            si = (si + 1) % len(segments)
        
        p0, p1, seg_len = segments[si]
        t = 0.0 if seg_len == 0.0 else (target - acc) / seg_len
        result.append(p0 * (1 - t) + p1 * t)
    
    return np.array(result)


def compute_centerline(
    inner_coords: np.ndarray,
    outer_coords: np.ndarray,
    n_samples: int = 1024,
    k_neigh: int = 8,
) -> np.ndarray:
    """
    Compute centerline by pairing resampled outer points with nearest inner points.
    """
    from scipy.spatial import cKDTree
    
    inner_resampled = resample_linestring(inner_coords, n_samples)
    outer_resampled = resample_linestring(outer_coords, n_samples)
    
    tree = cKDTree(inner_resampled)
    pairs = []
    _, j0 = tree.query(outer_resampled[0])
    prev_j = j0
    pairs.append((0, prev_j))
    
    for i in range(1, n_samples):
        _, candidates = tree.query(outer_resampled[i], k=k_neigh)
        if np.isscalar(candidates):
            candidates = [int(candidates)]
        
        best_j = None
        best_cost = float("inf")
        for cand in candidates:
            diff = (cand - prev_j) % n_samples
            wrap_diff = (prev_j - cand) % n_samples
            continuity_cost = min(diff, wrap_diff)
            dist_cost = np.linalg.norm(inner_resampled[cand] - outer_resampled[i])
            cost = dist_cost + 0.05 * continuity_cost
            if cost < best_cost:
                best_cost = cost
                best_j = cand
        
        prev_j = best_j
        pairs.append((i, best_j))
    
    centerline = []
    for i, j in pairs:
        mid = (outer_resampled[i] + inner_resampled[j]) / 2.0
        centerline.append(mid)
    
    centerline = np.array(centerline)
    
    # Smooth centerline
    for _ in range(2):
        smoothed = centerline.copy()
        for i in range(n_samples):
            prev_pt = centerline[(i - 1) % n_samples]
            next_pt = centerline[(i + 1) % n_samples]
            avg = (prev_pt + next_pt) / 2.0
            smoothed[i] = centerline[i] * 0.5 + avg * 0.5
        centerline = smoothed
    
    return centerline


def segment_track(polylines: List[List[Tuple[float, float]]], turn_threshold: float = 0.8) -> List[Dict]:
    """
    Split track into segments based on curvature.
    turn_threshold: degrees of heading change per point to consider a turn.
    """
    # Flatten polylines into a single list of points (assuming single loop)
    points = []
    for poly in polylines:
        points.extend(poly)
    
    if not points:
        return []

    # Convert to numpy for easier handling
    points_arr = np.array(points)
    
    # Calculate headings
    headings = []
    for i in range(len(points_arr) - 1):
        h = calculate_heading(points_arr[i], points_arr[i+1])
        headings.append(h)
    
    # Calculate curvature (change in heading)
    curvatures = []
    for i in range(len(headings) - 1):
        diff = normalize_angle(headings[i+1] - headings[i])
        curvatures.append(abs(diff))
    
    curvatures = [0] + curvatures + [0] 
    
    segments = []
    current_type = None # 'straight' or 'turn'
    current_points = []
    
    # Window size for smoothing (tuned value)
    window = 12
    
    for i in range(len(points_arr) - 1):
        # Simple smoothing
        start = max(0, i - window)
        end = min(len(curvatures), i + window + 1)
        avg_curv = np.mean(curvatures[start:end])
        
        segment_type = 'turn' if avg_curv > turn_threshold else 'straight'
        
        if segment_type != current_type:
            if current_points:
                segments.append({"type": current_type, "points": current_points})
            current_type = segment_type
            current_points = [points_arr[i]]
        else:
            current_points.append(points_arr[i])
            
    # Add last segment
    if current_points:
        current_points.append(points_arr[-1])
        segments.append({"type": current_type, "points": current_points})
    
    # Merge small segments
    min_points = 10 # Minimum points to be a valid segment
    
    if len(segments) > 1:
        cleaned_segments = []
        cleaned_segments.append(segments[0])
        
        for i in range(1, len(segments)):
            seg = segments[i]
            last = cleaned_segments[-1]
            
            if len(seg["points"]) < min_points:
                # Too small, merge into last
                last["points"].extend(seg["points"])
            else:
                # If type matches last (because we absorbed something), merge
                if seg["type"] == last["type"]:
                    last["points"].extend(seg["points"])
                else:
                    cleaned_segments.append(seg)
        
        segments = cleaned_segments

    return segments


def create_sectors_from_distances(
    polylines: List[List[Tuple[float, float]]], 
    distances: List[float]
) -> List[Dict]:
    """
    Create sector segments based on distance breakpoints.
    
    Args:
        polylines: Track polylines (WGS84 or WebMercator)
        distances: List of cumulative distances defining sector boundaries [0, 500, 1200, ...]
        
    Returns:
        List of segment dicts with 'type': 'sector', 'points', 'start_dist', 'end_dist'
    """
    # Flatten polylines
    all_points = []
    for poly in polylines:
        all_points.extend(poly)
    
    if not all_points:
        return []
    
    pts = np.array(all_points)
    
    # Project to meters if WGS84
    if np.max(np.abs(pts[:, 0])) <= 180 and np.max(np.abs(pts[:, 1])) <= 90:
        xs, ys = WGS84_TO_WEBMERC.transform(pts[:, 0], pts[:, 1])
        pts_m = np.column_stack((xs, ys))
    else:
        pts_m = pts
    
    # Calculate cumulative distance along track
    diffs = np.linalg.norm(pts_m[1:] - pts_m[:-1], axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(diffs)))
    
    # Create sectors
    sectors = []
    for i in range(len(distances) - 1):
        start_d = distances[i]
        end_d = distances[i + 1]
        
        # Find points in this distance range
        mask = (cum_dists >= start_d) & (cum_dists <= end_d)
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            sector_points = [tuple(all_points[j]) for j in indices]
            sectors.append({
                "type": "sector",
                "points": sector_points,
                "start_dist": start_d,
                "end_dist": end_d,
                "index": i,
            })
    
    return sectors


def load_sectors_json(path: Path) -> Optional[List[float]]:
    """
    Load sector distances from JSON file.
    Expected format: {"distances": [0, 500, 1200, 2000, ...]}
    """
    path = Path(path)
    if not path.is_file():
        return None
    
    with open(path, "r") as f:
        data = json.load(f)
    
    distances = data.get("distances", [])
    if not distances or len(distances) < 2:
        return None
    
    return sorted(distances)


class Track:
    """
    Unified track representation combining:
    - Layout (polylines, bounds, segments)
    - Start/finish line
    - Centerline projector for distance calculations
    """
    
    def __init__(
        self,
        polylines: List[List[Tuple[float, float]]],
        bounds: Tuple[float, float, float, float],
        segments: Optional[List[Dict]] = None,
        start_finish_webmerc: Optional[List[Tuple[float, float]]] = None,
        start_finish_wgs84: Optional[List[Tuple[float, float]]] = None,
        inner_boundary: Optional[np.ndarray] = None,
        outer_boundary: Optional[np.ndarray] = None,
    ):
        # Layout data
        self.polylines = polylines
        self.bounds = bounds
        self.segments = segments
        
        # Start/finish line
        self.start_finish_webmerc = start_finish_webmerc
        self.start_finish_wgs84 = start_finish_wgs84
        
        # Boundaries for dynamic centerline computation
        self._inner_boundary = inner_boundary
        self._outer_boundary = outer_boundary
        self._centerline_coords: Optional[np.ndarray] = None
        
        # Centerline projector (lazy init)
        self._projector_initialized = False
        self._projector_points: Optional[np.ndarray] = None
        self._cumulative_dists: Optional[np.ndarray] = None
        self._segments_vec: Optional[np.ndarray] = None
        self._segment_lengths_sq: Optional[np.ndarray] = None
        self._total_length: float = 0.0
    
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
        """Return centerline coordinates, computing lazily if needed."""
        if self._centerline_coords is None:
            if self._inner_boundary is not None and self._outer_boundary is not None:
                self._centerline_coords = compute_centerline(
                    self._inner_boundary, 
                    self._outer_boundary, 
                    n_samples=512
                )
        return self._centerline_coords
    
    def project(self, point: np.ndarray) -> float:
        """
        Project point onto nearest segment and return distance along centerline.
        
        Args:
            point: [x, y] coordinates in Web Mercator
            
        Returns:
            Distance along centerline in meters
        """
        self._init_projector()
        
        if self._projector_points is None:
            return 0.0
            
        # Vector from segment start to point
        v_start_point = point - self._projector_points[:-1]
        
        # Project v_start_point onto segment vector
        # t = dot(v_start_point, segment) / |segment|^2
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.sum(v_start_point * self._segments_vec, axis=1) / self._segment_lengths_sq
            t = np.nan_to_num(t)
        
        # Clamp t to [0, 1]
        t_clamped = np.clip(t, 0, 1)
        
        # Find closest point on each segment
        closest_points = self._projector_points[:-1] + self._segments_vec * t_clamped[:, np.newaxis]
        
        # Distance from query point to closest point on each segment
        dists_sq = np.sum((closest_points - point)**2, axis=1)
        
        # Find index of closest segment
        min_idx = np.argmin(dists_sq)
        
        # Calculate distance along centerline
        seg_len = np.sqrt(self._segment_lengths_sq[min_idx])
        dist_along = self._cumulative_dists[min_idx] + t_clamped[min_idx] * seg_len
        
        return float(dist_along)

    @classmethod
    def load(cls, track_dir: Path) -> "Track":
        """Load track from directory containing GeoJSON files.
        
        Generates centerline from inner/outer boundaries.
        Uses UTM Zone 38N (EPSG:32638) for accurate scaling.
        """
        track_dir = Path(track_dir)
        
        # Load inner/outer boundaries from GeoJSON
        inner_path = track_dir / "track-inner.geojson"
        outer_path = track_dir / "track-outer.geojson"
        
        if not inner_path.is_file() or not outer_path.is_file():
            raise FileNotFoundError(
                f"track-inner.geojson or track-outer.geojson not found in {track_dir}"
            )
        
        print(f"[Track] Loading boundaries from GeoJSON...")
        
        inner_pts = load_polyline_geojson(inner_path)
        outer_pts = load_polyline_geojson(outer_path)
        
        if not inner_pts or not outer_pts:
            raise ValueError(f"Failed to load inner/outer boundaries from {track_dir}")
        
        inner_arr = np.array(inner_pts)
        outer_arr = np.array(outer_pts)
        
        # Project to UTM Zone 38N for accurate scaling
        print(f"[Track] Projecting to UTM Zone 38N (EPSG:32638)...")
        inner_x, inner_y = WGS84_TO_UTM.transform(inner_arr[:, 0], inner_arr[:, 1])
        inner_arr = np.column_stack((inner_x, inner_y))
        outer_x, outer_y = WGS84_TO_UTM.transform(outer_arr[:, 0], outer_arr[:, 1])
        outer_arr = np.column_stack((outer_x, outer_y))
        
        # Compute centerline for polylines (needed for segments and projector)
        centerline = compute_centerline(inner_arr, outer_arr, n_samples=512)
        polylines = [list(map(tuple, centerline))]
        print(f"[Track] Generated centerline with {len(centerline)} points")
        
        # Calculate bounds from centerline
        xs = centerline[:, 0]
        ys = centerline[:, 1]
        bounds = (xs.min(), xs.max(), ys.min(), ys.max())
        
        # Generate segments from centerline
        segments = segment_track(polylines, turn_threshold=0.8)
        print(f"[Track] Generated {len(segments)} segments (straights/turns)")
        
        # Load start-finish line
        sf_path = track_dir / "start-finish.geojson"
        start_finish = load_polyline_geojson(sf_path)
        start_finish_wgs84 = None
        start_finish_utm = None
        
        if start_finish:
            start_finish_wgs84 = start_finish
            lons, lats = zip(*start_finish)
            xs, ys = WGS84_TO_UTM.transform(np.array(lons), np.array(lats))
            start_finish_utm = list(zip(xs, ys))
        
        track = cls(
            polylines=polylines,
            bounds=bounds,
            segments=segments,
            start_finish_webmerc=start_finish_utm,  # Now UTM, name kept for compat
            start_finish_wgs84=start_finish_wgs84,
            inner_boundary=inner_arr,
            outer_boundary=outer_arr,
        )
        # Pre-set the computed centerline to avoid recomputation
        track._centerline_coords = centerline
        return track


# Backward compatibility aliases
TrackLayout = Track  # For code that uses TrackLayout.polylines, .bounds, .segments
TrackGeometry = Track  # For code that uses TrackGeometry.layout, .start_finish_*

# Projector class for direct usage
class CenterlineProjector:
    """Wrapper for Track's projection functionality for backward compatibility."""
    
    def __init__(self, points: np.ndarray):
        self.points = points
        # Calculate cumulative distance along polyline
        dists = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)
        self.cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
        self.total_length = self.cumulative_dists[-1]
        
        # Pre-calculate segment vectors
        self.segments = self.points[1:] - self.points[:-1]
        self.segment_lengths_sq = np.sum(self.segments**2, axis=1)
        
    def project(self, point: np.ndarray) -> float:
        """Project point onto nearest segment and return distance along centerline."""
        v_start_point = point - self.points[:-1]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.sum(v_start_point * self.segments, axis=1) / self.segment_lengths_sq
            t = np.nan_to_num(t)
        
        t_clamped = np.clip(t, 0, 1)
        closest_points = self.points[:-1] + self.segments * t_clamped[:, np.newaxis]
        dists_sq = np.sum((closest_points - point)**2, axis=1)
        min_idx = np.argmin(dists_sq)
        seg_len = np.sqrt(self.segment_lengths_sq[min_idx])
        dist_along = self.cumulative_dists[min_idx] + t_clamped[min_idx] * seg_len
        
        return float(dist_along)
