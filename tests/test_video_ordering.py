"""Tests for optical flow-based video ordering fallback."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "racing_tools" / "video" / "concat"))
from concat import (
    VideoData,
    compute_flow_magnitude,
    extract_first_last_frames,
    order_videos_by_optical_flow,
)


class TestComputeFlowMagnitude:
    """Tests for compute_flow_magnitude function."""

    def test_identical_frames_zero_flow(self):
        """Identical frames should produce near-zero flow."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        flow = compute_flow_magnitude(frame, frame)
        assert flow < 0.5, f"Identical frames should have near-zero flow, got {flow}"

    def test_different_frames_high_flow(self):
        """Completely different frames should produce high flow."""
        frame1 = np.random.randint(0, 128, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(128, 256, (480, 640, 3), dtype=np.uint8)
        flow = compute_flow_magnitude(frame1, frame2)
        assert flow > 0.5, f"Different random frames should have high flow, got {flow}"

    def test_horizontal_shift_detected(self):
        """Horizontal shift should be detected as flow."""
        np.random.seed(42)
        frame1 = np.random.randint(50, 200, (240, 640, 3), dtype=np.uint8)
        frame2 = np.roll(frame1, shift=50, axis=1)
        flow = compute_flow_magnitude(frame1, frame2)
        assert flow > 0.5, f"Shifted frames should have measurable flow, got {flow}"

    def test_none_frames_return_infinity(self):
        """None frames should return infinity."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        flow = compute_flow_magnitude(None, frame)
        assert flow == float("inf")
        flow = compute_flow_magnitude(frame, None)
        assert flow == float("inf")

    def test_small_shift_small_flow(self):
        """Small horizontal shift should produce small flow."""
        np.random.seed(123)
        frame1 = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        frame2 = np.roll(frame1, shift=10, axis=1)
        flow = compute_flow_magnitude(frame1, frame2)
        assert 0.5 < flow < 15.0, f"Shifted frames should produce moderate flow, got {flow}"


class TestExtractExtremeFrames:
    """Tests for extract_first_last_frames function."""

    @pytest.fixture
    def sample_video(self, tmp_path):
        """Create a minimal test video using ffmpeg."""
        video_path = tmp_path / "test_video.mp4"
        import subprocess

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=5:size=320x240:rate=30",
            "-c:v",
            "libx264",
            "-frames:v",
            "150",
            str(video_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return video_path

    def test_extract_both_frames(self, sample_video):
        """Should extract both first and last frames."""
        first, last = extract_first_last_frames(sample_video)
        assert first is not None, "Failed to extract first frame"
        assert last is not None, "Failed to extract last frame"
        assert first.size > 0, "First frame is empty"
        assert last.size > 0, "Last frame is empty"

    def test_none_for_nonexistent_video(self, tmp_path):
        """Should return (None, None) for non-existent video."""
        fake_path = tmp_path / "nonexistent.mp4"
        first, last = extract_first_last_frames(fake_path)
        assert first is None
        assert last is None


class TestOrderVideosByOpticalFlow:
    """Tests for order_videos_by_optical_flow function."""

    def _make_video_data(self, name: str) -> VideoData:
        """Create a mock VideoData dict."""
        return {
            "file": Path(f"/test/{name}.mp4"),
            "duration": 10.0,
            "start_time": None,
            "end_time": None,
        }

    def test_empty_list_returns_empty(self):
        """Empty input should return empty list."""
        result = order_videos_by_optical_flow([])
        assert result == []

    def test_single_video_single_group(self):
        """Single video should return one group with one video."""
        video = self._make_video_data("video_a")
        result = order_videos_by_optical_flow([video])
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0]["file"].name == "video_a.mp4"

    def test_two_videos_ordered_by_flow(self):
        """Two videos should be ordered by flow similarity."""
        videos = [
            self._make_video_data("video_b"),
            self._make_video_data("video_a"),
        ]
        frames = {
            "video_a": (
                np.zeros((240, 320, 3), dtype=np.uint8),
                np.zeros((240, 320, 3), dtype=np.uint8),
            ),
            "video_b": (
                np.zeros((240, 320, 3), dtype=np.uint8),
                np.zeros((240, 320, 3), dtype=np.uint8),
            ),
        }

        def mock_extract(path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            for name, f in frames.items():
                if name in str(path):
                    return f
            return None, None

        with patch("concat.extract_first_last_frames", side_effect=mock_extract):
            result = order_videos_by_optical_flow(videos)

        assert len(result) == 1, "Should be one group"
        assert len(result[0]) == 2, "Should have two videos"

    def test_lexicographic_start_when_flow_equal(self):
        """When flows are equal, should start with lexicographically first."""
        videos = [
            self._make_video_data("video_z"),
            self._make_video_data("video_a"),
        ]
        frames = {
            "video_a": (
                np.zeros((240, 320, 3), dtype=np.uint8),
                np.zeros((240, 320, 3), dtype=np.uint8),
            ),
            "video_z": (
                np.zeros((240, 320, 3), dtype=np.uint8),
                np.zeros((240, 320, 3), dtype=np.uint8),
            ),
        }

        def mock_extract(path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            for name, f in frames.items():
                if name in str(path):
                    return f
            return None, None

        with patch("concat.extract_first_last_frames", side_effect=mock_extract):
            result = order_videos_by_optical_flow(videos)

        assert result[0][0]["file"].stem == "video_a", "Should start with 'video_a'"

    def test_preserves_videos_with_timestamps(self):
        """Videos with start_time should not be included in flow ordering."""
        from datetime import datetime

        video_with_ts = self._make_video_data("video_ts")
        video_with_ts["start_time"] = datetime(2024, 1, 1, 12, 0, 0)
        video_without_ts = self._make_video_data("video_no_ts")

        result = order_videos_by_optical_flow([video_with_ts, video_without_ts])
        flow_groups = [g for g in result if g and g[0] == video_without_ts]
        assert len(flow_groups) >= 1, "Video without timestamp should be in flow groups"

    def test_lexicographic_order_when_flows_equal(self):
        """When all flows are equal, output should follow lexicographic order."""
        videos = [
            self._make_video_data("c_video"),
            self._make_video_data("a_video"),
            self._make_video_data("b_video"),
        ]
        np.random.seed(42)
        identical_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        frames = {
            "a_video": (identical_frame.copy(), identical_frame.copy()),
            "b_video": (identical_frame.copy(), identical_frame.copy()),
            "c_video": (identical_frame.copy(), identical_frame.copy()),
        }

        def mock_extract(path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            for name, f in frames.items():
                if name in str(path):
                    return f
            return None, None

        with patch("concat.extract_first_last_frames", side_effect=mock_extract):
            result = order_videos_by_optical_flow(videos)

        assert len(result) == 1, "Should be one group"
        names = [v["file"].stem for v in result[0]]
        assert names == ["a_video", "b_video", "c_video"], f"Expected lexicographic order, got {names}"

    def test_lexicographic_first_regardless_of_flow(self):
        """First video in output should always be lexicographically first."""
        np.random.seed(42)
        frame_a = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        frame_z = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        frame_z[:, :, 0] = 255 - frame_z[:, :, 0]
        videos = [
            self._make_video_data("z_video"),
            self._make_video_data("a_video"),
        ]
        frames = {
            "a_video": (frame_a, frame_a),
            "z_video": (frame_z, frame_z),
        }

        def mock_extract(path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            for name, f in frames.items():
                if name in str(path):
                    return f
            return None, None

        with patch("concat.extract_first_last_frames", side_effect=mock_extract):
            result = order_videos_by_optical_flow(videos)

        assert len(result) == 1, "Should be one group"
        first_name = result[0][0]["file"].stem
        assert first_name == "a_video", f"First should be 'a_video' (lexicographic), got '{first_name}'"

    def test_real_videos_ordered_by_optical_flow(self):
        """Test ordering with real video files from test_data."""
        test_dir = Path(__file__).parent / "test_data"
        if not test_dir.exists():
            pytest.skip("test_data directory not found")

        videos = list(test_dir.glob("video_*.mp4"))
        if len(videos) < 2:
            pytest.skip("Need at least 2 test videos")

        video_data: list[VideoData] = [{"file": v, "duration": 15.0, "start_time": None, "end_time": None} for v in videos]

        result = order_videos_by_optical_flow(video_data)

        assert len(result) >= 1, "Should have at least one group"
        assert len(result[0]) == len(videos), f"Should have all {len(videos)} videos in first group"

        first_name = result[0][0]["file"].stem
        assert first_name == "video_a", f"First video should be 'video_a' (lexicographically first), got '{first_name}'"

    def test_crop_middle_excludes_timestamp_area(self):
        """Verify that frames are cropped to middle region."""
        test_dir = Path(__file__).parent / "test_data"
        if not test_dir.exists():
            pytest.skip("test_data directory not found")

        videos = list(test_dir.glob("video_*.mp4"))
        if not videos:
            pytest.skip("No test videos found")

        first, last = extract_first_last_frames(videos[0])
        assert first is not None, "Should extract frame"
        assert first.shape[1] < 1920, "Width should be cropped (was 1920, now < 1920)"
        assert first.shape[0] < 1080, "Height should be cropped (was 1080, now < 1080)"
