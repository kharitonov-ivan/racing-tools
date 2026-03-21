"""Tests for PiecewiseSync linear extrapolation beyond anchor boundaries."""

import numpy as np
import pytest

from racing_tools.session.session import PiecewiseSync


@pytest.fixture
def sync_two_anchors() -> PiecewiseSync:
    """Two anchors: video 100↔telem 50, video 200↔telem 150 (offset = -50)."""
    return PiecewiseSync(anchors=[(100.0, 50.0), (200.0, 150.0)])


@pytest.fixture
def sync_three_anchors() -> PiecewiseSync:
    """Three anchors with non-uniform spacing to verify segment selection."""
    return PiecewiseSync(anchors=[(100.0, 50.0), (200.0, 150.0), (250.0, 210.0)])


class TestInterpolation:
    """Verify basic interpolation between anchors still works."""

    def test_at_anchor_points(self, sync_two_anchors: PiecewiseSync) -> None:
        result = sync_two_anchors.video_to_telemetry(np.array([100.0, 200.0]))
        np.testing.assert_allclose(result, [50.0, 150.0])

    def test_midpoint(self, sync_two_anchors: PiecewiseSync) -> None:
        result = sync_two_anchors.video_to_telemetry(150.0)
        np.testing.assert_allclose(result, [100.0])

    def test_reverse_at_anchors(self, sync_two_anchors: PiecewiseSync) -> None:
        result = sync_two_anchors.telemetry_to_video(np.array([50.0, 150.0]))
        np.testing.assert_allclose(result, [100.0, 200.0])


class TestExtrapolation:
    """Verify linear extrapolation beyond the anchor range."""

    def test_before_first_anchor(self, sync_two_anchors: PiecewiseSync) -> None:
        """5 seconds before the first anchor should extrapolate, not clamp."""
        result = sync_two_anchors.video_to_telemetry(95.0)
        # slope = (150 - 50) / (200 - 100) = 1.0
        # expected = 50.0 + 1.0 * (95.0 - 100.0) = 45.0
        np.testing.assert_allclose(result, [45.0])

    def test_after_last_anchor(self, sync_two_anchors: PiecewiseSync) -> None:
        """5 seconds after the last anchor should extrapolate, not clamp."""
        result = sync_two_anchors.video_to_telemetry(205.0)
        # expected = 150.0 + 1.0 * (205.0 - 200.0) = 155.0
        np.testing.assert_allclose(result, [155.0])

    def test_before_first_anchor_reverse(self, sync_two_anchors: PiecewiseSync) -> None:
        result = sync_two_anchors.telemetry_to_video(45.0)
        np.testing.assert_allclose(result, [95.0])

    def test_three_anchors_left_extrapolation(self, sync_three_anchors: PiecewiseSync) -> None:
        """Left extrapolation uses the first segment slope."""
        result = sync_three_anchors.video_to_telemetry(90.0)
        # slope = (150 - 50) / (200 - 100) = 1.0
        # expected = 50.0 + 1.0 * (90.0 - 100.0) = 40.0
        np.testing.assert_allclose(result, [40.0])

    def test_three_anchors_right_extrapolation(self, sync_three_anchors: PiecewiseSync) -> None:
        """Right extrapolation uses the last segment slope."""
        result = sync_three_anchors.video_to_telemetry(260.0)
        # slope = (210 - 150) / (250 - 200) = 60/50 = 1.2
        # expected = 210.0 + 1.2 * (260.0 - 250.0) = 222.0
        np.testing.assert_allclose(result, [222.0])

    def test_array_with_mixed_ranges(self, sync_two_anchors: PiecewiseSync) -> None:
        """Array with values before, between, and after anchors."""
        times = np.array([90.0, 100.0, 150.0, 200.0, 210.0])
        result = sync_two_anchors.video_to_telemetry(times)
        np.testing.assert_allclose(result, [40.0, 50.0, 100.0, 150.0, 160.0])


class TestFromOffset:
    """Verify that from_offset still works correctly."""

    def test_constant_offset(self) -> None:
        sync = PiecewiseSync.from_offset(5.0)
        result = sync.video_to_telemetry(np.array([0.0, 10.0, 100.0]))
        np.testing.assert_allclose(result, [5.0, 15.0, 105.0])

    def test_negative_offset(self) -> None:
        sync = PiecewiseSync.from_offset(-3.0)
        result = sync.video_to_telemetry(1.0)
        np.testing.assert_allclose(result, [-2.0])


class TestEdgeCases:
    """Edge cases for single anchor and round-trip consistency."""

    def test_single_anchor_no_extrapolation(self) -> None:
        """Single anchor: np.interp returns constant, no extrapolation possible."""
        sync = PiecewiseSync(anchors=[(100.0, 50.0)])
        result = sync.video_to_telemetry(np.array([90.0, 100.0, 110.0]))
        # With single anchor, np.interp clamps to 50.0 (no slope to extrapolate)
        np.testing.assert_allclose(result, [50.0, 50.0, 50.0])

    def test_round_trip(self, sync_two_anchors: PiecewiseSync) -> None:
        """video → telemetry → video should return original value."""
        original = np.array([95.0, 100.0, 150.0, 200.0, 205.0])
        telem = sync_two_anchors.video_to_telemetry(original)
        recovered = sync_two_anchors.telemetry_to_video(telem)
        np.testing.assert_allclose(recovered, original, atol=1e-10)
