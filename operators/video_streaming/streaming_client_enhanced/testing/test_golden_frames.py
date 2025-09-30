#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Golden frame testing for StreamingClientOp Enhanced.

This module provides visual regression testing using golden reference frames
to ensure consistent video processing output across code changes.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
import yaml
from PIL import Image


class GoldenFrameComparator:
    """Utility class for comparing frames against golden references."""

    def __init__(self, golden_frames_dir: str, tolerance: float = 0.05):
        """
        Initialize the golden frame comparator.

        Args:
            golden_frames_dir: Directory containing golden reference frames
            tolerance: Acceptable difference ratio (0.0 to 1.0)
        """
        self.golden_frames_dir = Path(golden_frames_dir)
        self.tolerance = tolerance

    def load_golden_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Load a golden reference frame."""
        frame_path = self.golden_frames_dir / f"{frame_number:04d}.png"

        if not frame_path.exists():
            return None

        try:
            img = Image.open(frame_path)
            return np.array(img)
        except Exception as e:
            print(f"Error loading golden frame {frame_path}: {e}")
            return None

    def compare_frames(
        self, actual_frame: np.ndarray, golden_frame: np.ndarray
    ) -> Tuple[bool, float, dict]:
        """
        Compare an actual frame against a golden reference frame.

        Args:
            actual_frame: The frame to test
            golden_frame: The golden reference frame

        Returns:
            Tuple of (is_match, difference_ratio, comparison_stats)
        """
        if actual_frame.shape != golden_frame.shape:
            return (
                False,
                1.0,
                {
                    "error": "Shape mismatch",
                    "actual_shape": actual_frame.shape,
                    "golden_shape": golden_frame.shape,
                },
            )

        # Calculate pixel-wise differences
        diff = np.abs(actual_frame.astype(np.float32) - golden_frame.astype(np.float32))

        # Calculate various difference metrics
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        # Calculate percentage of pixels that differ significantly
        significant_diff_threshold = 10  # Consider differences > 10 as significant
        significant_diff_pixels = np.sum(diff > significant_diff_threshold)
        total_pixels = diff.size
        significant_diff_ratio = significant_diff_pixels / total_pixels

        # Overall difference ratio (normalized to 0-1)
        overall_diff_ratio = mean_diff / 255.0

        is_match = overall_diff_ratio <= self.tolerance

        stats = {
            "max_difference": float(max_diff),
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "significant_diff_ratio": float(significant_diff_ratio),
            "overall_diff_ratio": float(overall_diff_ratio),
            "tolerance": self.tolerance,
            "pixels_compared": total_pixels,
        }

        return is_match, overall_diff_ratio, stats

    def generate_diff_visualization(
        self, actual_frame: np.ndarray, golden_frame: np.ndarray, output_path: Optional[str] = None
    ) -> np.ndarray:
        """Generate a visualization showing the differences between frames."""
        if actual_frame.shape != golden_frame.shape:
            # Create a simple error visualization
            error_frame = np.full_like(actual_frame, [255, 0, 0])  # Red for error
            return error_frame

        # Create difference visualization
        diff = np.abs(actual_frame.astype(np.float32) - golden_frame.astype(np.float32))

        # Normalize difference to 0-255 range
        diff_normalized = (
            (diff / np.max(diff) * 255).astype(np.uint8)
            if np.max(diff) > 0
            else diff.astype(np.uint8)
        )

        # Create a heat map style visualization
        diff_viz = np.zeros_like(actual_frame)
        diff_viz[:, :, 0] = diff_normalized[:, :, 0]  # Red channel shows differences
        diff_viz[:, :, 1] = 255 - diff_normalized[:, :, 1]  # Green shows similarity
        diff_viz[:, :, 2] = diff_normalized[:, :, 2]  # Blue channel shows differences

        if output_path:
            Image.fromarray(diff_viz).save(output_path)

        return diff_viz


class TestGoldenFrames:
    """Test class for golden frame regression testing."""

    @pytest.fixture
    def golden_frame_config(self):
        """Load golden frame test configuration."""
        config_path = Path(__file__).parent / "golden_frames" / "golden_frame_test_config.yaml"

        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                "width": 854,
                "height": 480,
                "fps": 30,
                "tolerance": 0.05,
                "max_test_frames": 10,
            }

    @pytest.fixture
    def golden_frame_comparator(self, golden_frame_config):
        """Create a golden frame comparator instance."""
        golden_frames_dir = Path(__file__).parent / "golden_frames"
        tolerance = golden_frame_config.get("tolerance", 0.05)
        return GoldenFrameComparator(str(golden_frames_dir), tolerance)

    @pytest.mark.unit
    def test_golden_frames_exist(self, golden_frame_config):
        """Test that golden reference frames exist."""
        golden_frames_dir = Path(__file__).parent / "golden_frames"
        max_frames = golden_frame_config.get("max_test_frames", 10)

        assert golden_frames_dir.exists(), f"Golden frames directory not found: {golden_frames_dir}"

        for i in range(1, max_frames + 1):
            frame_path = golden_frames_dir / f"{i:04d}.png"
            assert frame_path.exists(), f"Golden frame {i:04d}.png not found"

    @pytest.mark.unit
    def test_golden_frame_loading(self, golden_frame_comparator, golden_frame_config):
        """Test that golden frames can be loaded correctly."""
        max_frames = golden_frame_config.get("max_test_frames", 10)

        for i in range(1, max_frames + 1):
            frame = golden_frame_comparator.load_golden_frame(i)
            assert frame is not None, f"Failed to load golden frame {i:04d}.png"
            assert len(frame.shape) == 3, f"Golden frame {i:04d}.png should be 3D (H,W,C)"
            assert frame.shape[2] == 3, f"Golden frame {i:04d}.png should have 3 channels (RGB)"

    @pytest.mark.unit
    def test_frame_comparison_identical(self, golden_frame_comparator):
        """Test frame comparison with identical frames."""
        # Create a test frame
        test_frame = np.random.randint(0, 256, (480, 854, 3), dtype=np.uint8)

        # Compare frame with itself
        is_match, diff_ratio, stats = golden_frame_comparator.compare_frames(test_frame, test_frame)

        assert is_match, "Identical frames should match"
        assert diff_ratio == 0.0, "Identical frames should have 0 difference"
        assert stats["mean_difference"] == 0.0, "Identical frames should have 0 mean difference"

    @pytest.mark.unit
    def test_frame_comparison_different(self, golden_frame_comparator):
        """Test frame comparison with different frames."""
        # Create two different test frames
        frame1 = np.zeros((480, 854, 3), dtype=np.uint8)
        frame2 = np.full((480, 854, 3), 255, dtype=np.uint8)

        # Compare different frames
        is_match, diff_ratio, stats = golden_frame_comparator.compare_frames(frame1, frame2)

        assert not is_match, "Completely different frames should not match"
        assert diff_ratio > 0.5, "Completely different frames should have high difference ratio"
        assert stats["max_difference"] == 255.0, "Max difference should be 255 for black vs white"

    @pytest.mark.unit
    def test_frame_comparison_shape_mismatch(self, golden_frame_comparator):
        """Test frame comparison with mismatched shapes."""
        frame1 = np.zeros((480, 854, 3), dtype=np.uint8)
        frame2 = np.zeros((240, 427, 3), dtype=np.uint8)  # Different size

        is_match, diff_ratio, stats = golden_frame_comparator.compare_frames(frame1, frame2)

        assert not is_match, "Frames with different shapes should not match"
        assert diff_ratio == 1.0, "Shape mismatch should result in maximum difference"
        assert "error" in stats, "Stats should contain error information"

    @pytest.mark.unit
    def test_diff_visualization_generation(self, golden_frame_comparator):
        """Test generation of difference visualization."""
        # Create two slightly different frames
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[50:60, 50:60] = 255  # Add a white square

        diff_viz = golden_frame_comparator.generate_diff_visualization(frame1, frame2)

        assert diff_viz.shape == frame1.shape, "Diff visualization should have same shape as input"
        assert diff_viz.dtype == np.uint8, "Diff visualization should be uint8"

        # Check that differences are highlighted (should be non-zero in difference region)
        diff_region = diff_viz[50:60, 50:60]
        assert np.any(diff_region > 0), "Difference region should be highlighted"

    @pytest.mark.integration
    @pytest.mark.parametrize("frame_number", [1, 2, 3, 4, 5])
    def test_mock_frame_against_golden(self, golden_frame_comparator, frame_number):
        """Test mock-generated frames against golden references."""
        # This is a placeholder for integration testing where we would:
        # 1. Generate a frame using the streaming client operator
        # 2. Compare it against the golden reference
        # 3. Validate the results

        golden_frame = golden_frame_comparator.load_golden_frame(frame_number)

        if golden_frame is None:
            pytest.skip(f"Golden frame {frame_number:04d}.png not available")

        # For this test, we'll simulate a "perfect" match by using the golden frame itself
        # In a real scenario, this would be the output from StreamingClientOp
        simulated_output = golden_frame.copy()

        is_match, diff_ratio, stats = golden_frame_comparator.compare_frames(
            simulated_output, golden_frame
        )

        assert is_match, f"Mock frame {frame_number} should match golden reference"
        assert (
            diff_ratio <= golden_frame_comparator.tolerance
        ), f"Difference ratio {diff_ratio} exceeds tolerance {golden_frame_comparator.tolerance}"

    @pytest.mark.integration
    def test_golden_frame_tolerance_sensitivity(self, golden_frame_comparator):
        """Test that tolerance settings work correctly."""
        # Load a golden frame for testing
        golden_frame = golden_frame_comparator.load_golden_frame(1)

        if golden_frame is None:
            pytest.skip("Golden frame 0001.png not available")

        # Create a more significantly modified version
        modified_frame = golden_frame.copy()
        # Make a significant change that should be detected by strict tolerance
        modified_frame[0:50, 0:50] = 255  # Make a large white region

        # Test with strict tolerance
        strict_comparator = GoldenFrameComparator(
            golden_frame_comparator.golden_frames_dir, tolerance=0.001
        )
        is_match_strict, _, _ = strict_comparator.compare_frames(modified_frame, golden_frame)

        # Test with loose tolerance
        loose_comparator = GoldenFrameComparator(
            golden_frame_comparator.golden_frames_dir, tolerance=0.1
        )
        is_match_loose, _, _ = loose_comparator.compare_frames(modified_frame, golden_frame)

        # Strict tolerance should reject, loose tolerance should accept
        assert not is_match_strict, "Strict tolerance should reject small differences"
        assert is_match_loose, "Loose tolerance should accept small differences"


# Utility functions for golden frame testing


def create_test_frame_from_golden(
    golden_frames_dir: str, frame_number: int, noise_level: float = 0.0
) -> np.ndarray:
    """
    Create a test frame based on a golden reference frame with optional noise.

    Args:
        golden_frames_dir: Directory containing golden frames
        frame_number: Frame number to load
        noise_level: Amount of noise to add (0.0 to 1.0)

    Returns:
        Test frame as numpy array
    """
    comparator = GoldenFrameComparator(golden_frames_dir)
    golden_frame = comparator.load_golden_frame(frame_number)

    if golden_frame is None:
        raise ValueError(f"Could not load golden frame {frame_number:04d}.png")

    test_frame = golden_frame.copy()

    if noise_level > 0:
        # Add random noise
        noise = np.random.normal(0, noise_level * 255, golden_frame.shape).astype(np.float32)
        test_frame = np.clip(test_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return test_frame


def run_golden_frame_validation(
    output_frames: List[np.ndarray], golden_frames_dir: str, tolerance: float = 0.05
) -> dict:
    """
    Run golden frame validation on a list of output frames.

    Args:
        output_frames: List of frames to validate
        golden_frames_dir: Directory containing golden reference frames
        tolerance: Acceptable difference tolerance

    Returns:
        Validation results dictionary
    """
    comparator = GoldenFrameComparator(golden_frames_dir, tolerance)
    results = {
        "total_frames": len(output_frames),
        "passed_frames": 0,
        "failed_frames": 0,
        "frame_results": [],
        "overall_success": False,
    }

    for i, frame in enumerate(output_frames):
        frame_number = i + 1
        golden_frame = comparator.load_golden_frame(frame_number)

        if golden_frame is None:
            results["frame_results"].append(
                {
                    "frame_number": frame_number,
                    "status": "skipped",
                    "reason": "Golden frame not found",
                }
            )
            continue

        is_match, diff_ratio, stats = comparator.compare_frames(frame, golden_frame)

        if is_match:
            results["passed_frames"] += 1
            status = "passed"
        else:
            results["failed_frames"] += 1
            status = "failed"

        results["frame_results"].append(
            {
                "frame_number": frame_number,
                "status": status,
                "difference_ratio": diff_ratio,
                "stats": stats,
            }
        )

    # Overall success if more than 80% of frames pass
    success_rate = results["passed_frames"] / max(1, results["total_frames"])
    results["overall_success"] = success_rate >= 0.8
    results["success_rate"] = success_rate

    return results
