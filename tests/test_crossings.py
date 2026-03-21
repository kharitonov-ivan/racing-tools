"""Tests for start-finish line crossing detection."""

import numpy as np
import pytest

from racing_tools.session.utils import (
    detect_sf_crossings,
    interpolate_crossing_time,
    point_side_of_line,
    segments_intersect,
)


class TestPointSideOfLine:
    """Tests for point_side_of_line function."""

    def test_point_on_left_side(self):
        """Point above horizontal line should return negative (cross product sign)."""
        lons = np.array([0.5])
        lats = np.array([1.0])
        sf_p1 = (0.0, 0.0)
        sf_p2 = (1.0, 0.0)  # Horizontal line

        result = point_side_of_line(lons, lats, sf_p1, sf_p2)
        assert result[0] < 0  # Above the line = negative (cross product convention)

    def test_point_on_right_side(self):
        """Point below horizontal line should return positive (cross product sign)."""
        lons = np.array([0.5])
        lats = np.array([-1.0])
        sf_p1 = (0.0, 0.0)
        sf_p2 = (1.0, 0.0)  # Horizontal line

        result = point_side_of_line(lons, lats, sf_p1, sf_p2)
        assert result[0] > 0  # Below the line = positive (cross product convention)

    def test_point_on_line(self):
        """Point on the line should return zero."""
        lons = np.array([0.5])
        lats = np.array([0.0])
        sf_p1 = (0.0, 0.0)
        sf_p2 = (1.0, 0.0)  # Horizontal line

        result = point_side_of_line(lons, lats, sf_p1, sf_p2)
        assert result[0] == 0.0

    def test_multiple_points(self):
        """Test with multiple points."""
        lons = np.array([0.5, 0.5, 0.5])
        lats = np.array([1.0, 0.0, -1.0])
        sf_p1 = (0.0, 0.0)
        sf_p2 = (1.0, 0.0)

        result = point_side_of_line(lons, lats, sf_p1, sf_p2)
        assert result[0] < 0  # Above = negative
        assert result[1] == 0.0  # On line = 0
        assert result[2] > 0  # Below = positive


class TestInterpolateCrossingTime:
    """Tests for interpolate_crossing_time function."""

    def test_midpoint_crossing(self):
        """Crossing at midpoint should return average time."""
        t_prev = np.array([0.0])
        t_curr = np.array([10.0])
        side_prev = np.array([-1.0])
        side_curr = np.array([1.0])

        result = interpolate_crossing_time(t_prev, t_curr, side_prev, side_curr)
        assert result[0] == 5.0

    def test_offset_crossing(self):
        """Crossing closer to current point."""
        t_prev = np.array([0.0])
        t_curr = np.array([10.0])
        side_prev = np.array([-1.0])
        side_curr = np.array([4.0])  # Crossing at t=0.2

        result = interpolate_crossing_time(t_prev, t_curr, side_prev, side_curr)
        assert result[0] == 2.0

    def test_multiple_crossings(self):
        """Test interpolation of multiple crossings."""
        t_prev = np.array([0.0, 10.0, 20.0])
        t_curr = np.array([10.0, 20.0, 30.0])
        side_prev = np.array([-1.0, -2.0, -3.0])
        side_curr = np.array([1.0, 2.0, 3.0])

        result = interpolate_crossing_time(t_prev, t_curr, side_prev, side_curr)
        assert len(result) == 3
        assert result[0] == 5.0
        assert result[1] == 15.0
        assert result[2] == 25.0


class TestSegmentsIntersect:
    """Tests for segments_intersect function."""

    def test_crossing_segments(self):
        """Two crossing segments should intersect."""
        hit, t, sign = segments_intersect((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))
        assert hit is True
        assert abs(t - 0.5) < 1e-9

    def test_parallel_segments(self):
        """Parallel segments should not intersect."""
        hit, t, sign = segments_intersect((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        assert hit is False

    def test_non_overlapping_segments(self):
        """Non-overlapping segments (would intersect if extended) should not intersect."""
        hit, t, sign = segments_intersect((0.0, 0.0), (0.5, 0.0), (1.0, -1.0), (1.0, 1.0))
        assert hit is False

    def test_crossing_sign(self):
        """Crossing sign should differ for opposite directions."""
        _, _, sign1 = segments_intersect((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))
        _, _, sign2 = segments_intersect((0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0))
        assert sign1 != 0
        assert sign2 != 0
        assert sign1 == -sign2


class TestDetectSfCrossings:
    """Tests for detect_sf_crossings function (segment intersection based)."""

    def test_no_crossings_single_side(self):
        """GPS trajectory staying on one side should have no crossings."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        lons = np.array([0.5, 0.5, 0.5, 0.5])
        lats = np.array([1.0, 1.5, 2.0, 2.5])  # All above the line
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        result = detect_sf_crossings(times, lons, lats, sf_line)
        assert result == []

    def test_no_crossings_trajectory_misses_segment(self):
        """GPS trajectory crossing the infinite line but NOT the finite SF segment."""
        # SF segment from (0,0) to (0.1, 0) — very short segment at origin
        # Trajectory at lon=5.0 crosses lat=0 but is far from the segment
        times = np.array([0.0, 50.0])
        lons = np.array([5.0, 5.0])
        lats = np.array([-0.1, 0.1])
        sf_line = [(0.0, 0.0), (0.1, 0.0)]

        result = detect_sf_crossings(times, lons, lats, sf_line)
        assert result == [], "Should NOT detect crossing of infinite line extension"

    def test_single_crossing(self):
        """GPS trajectory crossing the SF segment once."""
        # SF line from (0, -0.5) to (1, -0.5) — horizontal at lat=-0.5
        # Wait, let's use a clear geometry:
        # SF segment: horizontal line from (0, 0) to (1, 0)
        # Trajectory: lon=0.5, going from lat=-0.1 to lat=0.1 (crosses at midpoint)
        times = np.array([0.0, 10.0])
        lons = np.array([0.5, 0.5])
        lats = np.array([-0.1, 0.1])
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        result = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0)
        assert len(result) == 1
        assert abs(result[0] - 5.0) < 0.1

    def test_multiple_crossings_same_direction(self):
        """Multiple laps crossing in the same direction."""
        # Simulate a trajectory that crosses y=0 line at x=0.5 multiple times
        # Going up (negative to positive lat) each time
        n_laps = 5
        times = []
        lons = []
        lats = []
        for lap in range(n_laps):
            base_t = lap * 60.0
            # Approach from below
            times.extend([base_t, base_t + 29.0])
            lons.extend([0.5, 0.5])
            lats.extend([-0.5, -0.01])
            # Cross the line
            times.append(base_t + 30.0)
            lons.append(0.5)
            lats.append(0.01)
            # Move away above
            times.append(base_t + 31.0)
            lons.append(0.5)
            lats.append(0.5)
            # Come back below (crossing in opposite direction — should be filtered)
            times.append(base_t + 45.0)
            lons.append(0.5)
            lats.append(-0.5)

        times = np.array(times)
        lons = np.array(lons)
        lats = np.array(lats)
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        result = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=30.0)
        # Should detect crossings in dominant direction only
        assert len(result) >= 3  # At least 3 of the 5 forward crossings

    def test_dominant_direction_filter(self):
        """Only crossings in the dominant direction should be kept."""
        # 3 crossings going up, 1 going down → only 3 should be kept
        times = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        lons = np.array([0.5] * 8)
        # up, down, up, up pattern
        lats = np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        result = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=5.0)
        # 7 sign changes total, but only one direction should be kept
        # All crossings alternate direction, so we get ~3-4 in dominant direction
        assert len(result) >= 2

    def test_min_lap_time_filter(self):
        """Crossings too close together should be filtered."""
        times = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        lons = np.array([0.5] * 5)
        lats = np.array([-0.1, 0.1, -0.1, 0.1, -0.1])
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        # With min_lap_time=30, only first crossing should be kept
        result = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=30.0)
        assert len(result) <= 1

    def test_empty_inputs(self):
        """Empty inputs should return empty list."""
        assert detect_sf_crossings(np.array([]), np.array([]), np.array([]), []) == []
        assert detect_sf_crossings(np.array([1.0]), np.array([0.5]), np.array([0.5]), []) == []

    def test_nan_values(self):
        """NaN GPS values should be handled correctly."""
        times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        lons = np.array([0.5, np.nan, 0.5, 0.5, 0.5])
        lats = np.array([-0.1, -0.1, -0.01, 0.01, 0.1])
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        result = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0)
        # Should still detect crossing between points 2→3 despite NaN at point 1
        assert len(result) >= 1

    def test_diagonal_sf_line(self):
        """Test with diagonal start-finish line."""
        # SF line: from (0,0) to (1,1) — diagonal
        # Trajectory must actually cross this finite segment
        # Point at (0.3, 0.7) → (0.7, 0.3) crosses the diagonal y=x at (0.5, 0.5)
        times = np.array([0.0, 50.0])
        lons = np.array([0.3, 0.7])
        lats = np.array([0.7, 0.3])
        sf_line = [(0.0, 0.0), (1.0, 1.0)]

        result = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0)
        assert len(result) == 1
        assert 0 < result[0] < 50


class TestSpeedWeightedCrossings:
    """Tests for speed-weighted crossing time interpolation."""

    def test_constant_speed_matches_linear(self):
        """With constant speed, result should match linear interpolation."""
        times = np.array([0.0, 10.0])
        lons = np.array([0.5, 0.5])
        lats = np.array([-0.1, 0.1])
        speeds = np.array([60.0, 60.0])  # constant 60 km/h
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        result_with_speed = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0, speeds=speeds)
        result_no_speed = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0)

        assert len(result_with_speed) == 1
        assert len(result_no_speed) == 1
        assert abs(result_with_speed[0] - result_no_speed[0]) < 0.01

    def test_accelerating_shifts_crossing_later(self):
        """When accelerating, crossing should be slightly later than linear."""
        # If kart is accelerating (slow at start, fast at end), it takes
        # longer to reach the midpoint → crossing time > linear midpoint
        times = np.array([0.0, 10.0])
        lons = np.array([0.5, 0.5])
        lats = np.array([-0.1, 0.1])  # crossing at geometric midpoint (t_frac=0.5)
        speeds_accel = np.array([30.0, 90.0])  # accelerating
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        result_accel = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0, speeds=speeds_accel)
        result_linear = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0)

        assert len(result_accel) == 1
        assert len(result_linear) == 1
        # When accelerating, it takes longer to reach the midpoint
        assert result_accel[0] > result_linear[0]

    def test_decelerating_shifts_crossing_earlier(self):
        """When decelerating, crossing should be slightly earlier than linear."""
        times = np.array([0.0, 10.0])
        lons = np.array([0.5, 0.5])
        lats = np.array([-0.1, 0.1])
        speeds_decel = np.array([90.0, 30.0])  # decelerating
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        result_decel = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0, speeds=speeds_decel)
        result_linear = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0)

        assert len(result_decel) == 1
        assert len(result_linear) == 1
        # When decelerating, kart reaches midpoint faster
        assert result_decel[0] < result_linear[0]

    def test_zero_speed_falls_back_to_linear(self):
        """Zero speed should fall back to linear interpolation."""
        times = np.array([0.0, 10.0])
        lons = np.array([0.5, 0.5])
        lats = np.array([-0.1, 0.1])
        speeds = np.array([0.0, 0.0])
        sf_line = [(0.0, 0.0), (1.0, 0.0)]

        result = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0, speeds=speeds)
        result_linear = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0)

        assert len(result) == 1
        assert abs(result[0] - result_linear[0]) < 0.01


class TestAlignCrossings:
    """Tests for _align_crossings video/telemetry alignment."""

    def test_perfect_match(self):
        """Identical crossing patterns should align perfectly."""
        from racing_tools.utils.telemetry_sync import _align_crossings

        video = [10.0, 70.0, 130.0, 190.0]
        telem = [100.0, 160.0, 220.0, 280.0]  # same intervals, different offset

        anchors, offset = _align_crossings(video, telem)
        assert len(anchors) == 4
        assert anchors[0] == (10.0, 100.0)
        assert anchors[-1] == (190.0, 280.0)

    def test_video_has_extra_laps_at_start(self):
        """Video starts recording earlier — extra crossings at beginning."""
        from racing_tools.utils.telemetry_sync import _align_crossings

        # Video: 5 laps with varying intervals (distinguishable pattern)
        video = [10.0, 68.0, 130.0, 189.0, 250.0]
        # intervals: [58, 62, 59, 61]
        # Telem: only last 3 laps matching video[2:]
        # telem intervals should match video intervals [62, 59, 61] starting at video[1:]
        # Actually telem has 3 crossings → 2 intervals matching video's last 2 intervals
        telem = [200.0, 259.0, 320.0]
        # telem intervals: [59, 61] — matches video intervals[2:] = [59, 61]

        anchors, offset = _align_crossings(video, telem)
        assert len(anchors) == 3
        # Should align video[2:] with telem[0:] since intervals [59, 61] match
        assert anchors[0][0] == 130.0
        assert anchors[0][1] == 200.0

    def test_telem_has_extra_laps_at_start(self):
        """Telemetry starts recording earlier — extra crossings at beginning."""
        from racing_tools.utils.telemetry_sync import _align_crossings

        # Video: 3 laps with specific intervals
        video = [10.0, 69.0, 130.0]
        # video intervals: [59, 61]
        # Telem: 5 laps, video matches telem[2:]
        telem = [100.0, 158.0, 220.0, 279.0, 340.0]
        # telem intervals: [58, 62, 59, 61]
        # video intervals [59, 61] match telem intervals[2:] = [59, 61]

        anchors, offset = _align_crossings(video, telem)
        assert len(anchors) == 3
        # Should align video[0:] with telem[2:]
        assert anchors[0][1] == 220.0

    def test_single_crossing_each_raises(self):
        """Single crossing in each — should fail (need >= 2 for piecewise sync)."""
        from racing_tools.utils.telemetry_sync import _align_crossings

        video = [50.0]
        telem = [120.0]

        with pytest.raises(AssertionError, match="Need at least 2 anchor pairs"):
            _align_crossings(video, telem)

    def test_empty_inputs(self):
        """Empty inputs should return empty."""
        from racing_tools.utils.telemetry_sync import _align_crossings

        assert _align_crossings([], [100.0]) == ([], 0)
        assert _align_crossings([10.0], []) == ([], 0)
        assert _align_crossings([], []) == ([], 0)

    def test_different_lap_counts_raises(self):
        """Different number of laps after alignment — must fail."""
        from racing_tools.utils.telemetry_sync import _align_crossings

        # Video: 4 crossings, telem: 6 crossings — cannot match all
        video = [10.0, 70.5, 131.0, 191.5]
        telem = [50.0, 110.0, 170.5, 231.0, 291.5, 352.0]

        with pytest.raises(AssertionError, match="Lap count mismatch"):
            _align_crossings(video, telem)

    def test_same_lap_count_different_offset(self):
        """Same number of laps with offset — should align."""
        from racing_tools.utils.telemetry_sync import _align_crossings

        # Video: 4 crossings, telem: 4 crossings with different start
        video = [10.0, 70.5, 131.0, 191.5]
        telem = [200.0, 260.5, 321.0, 381.5]

        anchors, offset = _align_crossings(video, telem)
        assert len(anchors) == 4
        for i in range(len(anchors) - 1):
            v_int = anchors[i + 1][0] - anchors[i][0]
            t_int = anchors[i + 1][1] - anchors[i][1]
            assert abs(v_int - t_int) < 0.3


class TestBestlineSfIntersection:
    """Test that bestline/centerline intersects the start-finish line."""

    def test_rim_sport_karting_sf_bestline_intersection(self):
        """The RIMSportKarting track SF line should intersect the centerline."""
        from racing_tools.track.track import Track

        track = Track.load("racing_tools/track/data/RIMSportKarting")

        assert track.start_finish_utm is not None, "SF line should be loaded"
        assert track.centerline is not None, "Centerline should be computed"

        intersection = track.start_finish_intersection
        assert intersection is not None, "SF line should intersect centerline/bestline"
        assert "point" in intersection, "Intersection should have a point"

        # The intersection point should be within the track bounds
        xmin, xmax, ymin, ymax = track.bounds
        px, py = intersection["point"]
        assert xmin <= px <= xmax, f"Intersection x={px} outside bounds [{xmin}, {xmax}]"
        assert ymin <= py <= ymax, f"Intersection y={py} outside bounds [{ymin}, {ymax}]"

    def test_sf_line_crosses_centerline_in_wgs84(self):
        """Verify SF line actually crosses the centerline using segment intersection."""
        from racing_tools.track.track import Track

        track = Track.load("racing_tools/track/data/RIMSportKarting")

        sf_wgs84 = track.start_finish_wgs84
        assert sf_wgs84 is not None and len(sf_wgs84) >= 2

        # Get centerline in WGS84
        centerline = track.centerline
        assert centerline is not None

        transformer = track._get_transformer_to_wgs84()
        xs, ys = centerline[:, 0], centerline[:, 1]
        lons, lats = transformer.transform(xs, ys)

        # Check that at least one centerline segment crosses the SF line
        sf_p1, sf_p2 = sf_wgs84[0], sf_wgs84[-1]
        found_crossing = False
        for i in range(len(lons) - 1):
            hit, t, sign = segments_intersect(
                (lons[i], lats[i]), (lons[i + 1], lats[i + 1]),
                sf_p1, sf_p2,
            )
            if hit:
                found_crossing = True
                break

        assert found_crossing, (
            "SF line should cross the centerline at least once. "
            "If this fails, the SF line geometry may be misplaced."
        )

    def test_simulated_lap_detects_crossing(self):
        """Simulate a GPS trajectory along the centerline and verify crossing detection."""
        from racing_tools.track.track import Track

        track = Track.load("racing_tools/track/data/RIMSportKarting")

        sf_wgs84 = track.start_finish_wgs84
        assert sf_wgs84 is not None

        # Get centerline in WGS84 as a simulated GPS trajectory
        centerline = track.centerline
        assert centerline is not None

        transformer = track._get_transformer_to_wgs84()
        xs, ys = centerline[:, 0], centerline[:, 1]
        lons, lats = transformer.transform(xs, ys)

        # Create time array (simulate ~60s lap)
        n = len(lons)
        times = np.linspace(0.0, 60.0, n)

        sf_line = list(sf_wgs84)
        crossings = detect_sf_crossings(times, lons, lats, sf_line, min_lap_time=0.0)

        assert len(crossings) >= 1, (
            f"Simulated lap along centerline should cross SF line at least once, "
            f"got {len(crossings)} crossings"
        )
