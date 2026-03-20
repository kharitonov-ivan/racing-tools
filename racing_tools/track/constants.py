"""
Constants for track geometry processing.

All magic numbers are documented here with their purpose and rationale.
"""

# Constants for centerline computation
CONTINUITY_WEIGHT = 0.05
"""
Weight for continuity cost relative to distance in compute_centerline.

Higher values prioritize smooth centerline (fewer jumps), lower values prioritize
minimum distance between inner and outer boundaries. Range: 0.0-1.0, recommended 0.05.
"""

# Constants for track segmentation
SMOOTHING_WINDOW = 12
"""
Window size for smoothing curvature values (number of points).

Larger values = smoother but less responsive segmentation. Smaller values = more noise.
Recommended range: 8-20 for typical kart tracks (512-1024 points).
"""

MIN_SEGMENT_POINTS = 10
"""
Minimum points required to form a valid segment.

Prevents creation of tiny segments from noise or outliers. Should be at least 2x
smoothing window to avoid edge effects.
"""

DEFAULT_TURN_THRESHOLD = 0.8
"""
Default threshold for turn detection (degrees of heading change per point).

Values above this threshold are considered turns, below are straights.
Rationale: at 512 points and typical kart track (~1km), point spacing ~2m.
0.8 degrees over 2m ≈ 0.7% curvature, reasonable turn/straight boundary.
Adjust based on track complexity: 0.5-1.2 typical range.
"""

# Turn thresholds for different line types
BESTLINE_TURN_THRESHOLD = 0.5
"""
Turn detection threshold for bestline (optimal racing line).

Lower than centerline threshold because bestline is smoother with more gradual
turns. Value in degrees of heading change per point.
Typical range: 0.3-0.6 for racing lines.
"""

CENTERLINE_TURN_THRESHOLD = 0.8
"""
Turn detection threshold for centerline (geometric middle of track).

Higher than bestline threshold because centerline follows the track geometry more
directly. Value in degrees of heading change per point.
Typical range: 0.6-1.0 for geometric centerlines.
"""

# Constants for centerline resampling
DEFAULT_CENTERLINE_SAMPLES = 512
"""
Number of points for resampled centerline.

Higher = more accurate distance projection, slower processing.
512 points gives ~2m resolution on 1km track, good balance for kart telemetry.
"""

# Constants for centerline computation
K_NEIGHBORS = 8
"""
Number of neighbors to consider for pairing outer boundary points to inner boundary.

More neighbors = better continuity but slower. 8 is good balance for typical tracks.
"""

# Coordinate reference systems
WGS84_CRS = "EPSG:4326"
"""
WGS84 - standard GPS coordinate system (latitude/longitude in degrees).
Used for input data and storage.
"""

WEBMERCATOR_CRS = "EPSG:3857"
"""
Web Mercator - used for web mapping (Google Maps, OpenStreetMap).
Distorts distances ~35% at mid-latitudes, use only for display.
"""

DEFAULT_UTM_ZONE = "EPSG:32638"
"""
UTM Zone 38N - covers Georgia (40°N to 44°N, 36°E to 42°E).
Provides accurate distance measurements (<0.1% error).
Override in track_config.json for tracks in other zones.
"""

# Constants for polyline processing
DUPLICATE_TOLERANCE = 1e-3
"""
Minimum distance in projected coordinates (meters) to consider points distinct.

Prevents duplicate points from causing issues in resampling and projection.
UTM coordinates: 1e-3 ≈ 1mm, safe threshold.
"""

# Constants for lap validation
MIN_VALID_LAP_TIME = 20.0
"""
Minimum lap time in seconds to be considered valid.

Laps shorter than this are typically out-laps, pit-in, or noise.
Used for filtering lap statistics and best lap selection.
"""

# Constants for delta filtering
MAX_DELTA_FOR_DISPLAY = 20.0
"""
Maximum delta time in seconds to display.

Deltas larger than this are typically from out-laps vs flying laps
and should be filtered out as they don't represent meaningful comparisons.
"""

# Constants for telemetry display
DEFAULT_MAX_RPM = 14000
"""
Default maximum RPM for gauge display.

Used when actual max RPM cannot be determined from telemetry.
Typical range for karting: 12000-16000.
"""
