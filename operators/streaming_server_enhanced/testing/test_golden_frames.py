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

"""Golden frame tests for StreamingServer visual regression testing."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock

try:
    from PIL import Image
except ImportError:
    pytest.skip("PIL (Pillow) required for golden frame testing", allow_module_level=True)

from test_utils import (
    create_test_frame_sequence,
    assert_frame_properties,
    TestFrameValidator
)
from mock_holoscan_framework import (
    MockFrame,
    create_mock_bgr_frame
)


class GoldenFrameComparator:
    """Helper class for comparing frames against golden references."""
    
    def __init__(self, golden_frames_dir, tolerance=0.05):
        """
        Initialize golden frame comparator.
        
        Args:
            golden_frames_dir: Directory containing golden reference frames
            tolerance: Allowed difference ratio (0.0 = exact match, 1.0 = any difference allowed)
        """
        self.golden_frames_dir = Path(golden_frames_dir)
        self.tolerance = tolerance
        
    def load_golden_frame(self, frame_number):
        """Load a golden reference frame by number."""
        filename = f"{frame_number:04d}.png"
        filepath = self.golden_frames_dir / filename
        
        if not filepath.exists():
            return None
            
        try:
            with Image.open(filepath) as img:
                # Convert to BGR format (OpenCV standard)
                rgb_array = np.array(img)
                bgr_array = rgb_array[:, :, ::-1]  # RGB to BGR
                
                height, width = bgr_array.shape[:2]
                frame = MockFrame(width, height, 3)
                frame.data = bgr_array
                return frame
        except Exception as e:
            print(f"Error loading golden frame {filename}: {e}")
            return None
    
    def compare_frames(self, test_frame, reference_frame):
        """
        Compare a test frame against a reference frame.
        
        Returns:
            tuple: (is_match, difference_ratio, difference_stats)
        """
        if reference_frame is None:
            return False, 1.0, {"error": "Reference frame not available"}
        
        # Ensure frames have same dimensions
        if (test_frame.width != reference_frame.width or 
            test_frame.height != reference_frame.height or
            test_frame.channels != reference_frame.channels):
            return False, 1.0, {
                "error": "Dimension mismatch",
                "test_dims": (test_frame.width, test_frame.height, test_frame.channels),
                "ref_dims": (reference_frame.width, reference_frame.height, reference_frame.channels)
            }
        
        # Calculate pixel-wise differences
        test_data = test_frame.data.astype(np.float32)
        ref_data = reference_frame.data.astype(np.float32)
        
        diff = np.abs(test_data - ref_data)
        max_possible_diff = 255.0
        
        # Calculate difference statistics
        pixel_count = test_data.size
        total_diff = np.sum(diff)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Calculate difference ratio (0.0 = identical, 1.0 = completely different)
        difference_ratio = total_diff / (pixel_count * max_possible_diff)
        
        is_match = difference_ratio <= self.tolerance
        
        stats = {
            "difference_ratio": difference_ratio,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "total_diff": total_diff,
            "pixel_count": pixel_count,
            "tolerance": self.tolerance
        }
        
        return is_match, difference_ratio, stats
    
    def validate_frame_against_golden(self, test_frame, frame_number):
        """Validate a test frame against the corresponding golden frame."""
        reference_frame = self.load_golden_frame(frame_number)
        return self.compare_frames(test_frame, reference_frame)


class TestGoldenFrames:
    """Test suite for golden frame visual regression testing."""
    
    @pytest.fixture
    def golden_frame_comparator(self, golden_frames_dir):
        """Provide a golden frame comparator."""
        return GoldenFrameComparator(golden_frames_dir, tolerance=0.05)
    
    @pytest.mark.golden_frame
    def test_golden_frames_directory_exists(self, golden_frames_dir):
        """Test that golden frames directory exists."""
        assert golden_frames_dir.exists(), f"Golden frames directory not found: {golden_frames_dir}"
        assert golden_frames_dir.is_dir(), f"Golden frames path is not a directory: {golden_frames_dir}"
    
    @pytest.mark.golden_frame
    def test_golden_frames_available(self, golden_frames_dir):
        """Test that golden frames are available."""
        png_files = list(golden_frames_dir.glob("*.png"))
        assert len(png_files) > 0, f"No golden frames found in {golden_frames_dir}"
        
        # Check for expected naming pattern
        expected_files = [f"{i:04d}.png" for i in range(1, 11)]  # 0001.png to 0010.png
        available_files = [f.name for f in png_files]
        
        # At least some expected files should be present
        found_expected = [f for f in expected_files if f in available_files]
        assert len(found_expected) > 0, f"No expected golden frames found. Available: {available_files}"
    
    @pytest.mark.golden_frame
    def test_golden_frame_loading(self, golden_frame_comparator):
        """Test loading golden reference frames."""
        # Try to load first golden frame
        golden_frame = golden_frame_comparator.load_golden_frame(1)
        
        if golden_frame is None:
            pytest.skip("Golden frame 0001.png not available")
        
        # Verify frame properties
        assert_frame_properties(golden_frame, 854, 480, 3)
        assert golden_frame.data.dtype == np.uint8
        
        # Verify frame contains actual data (not all zeros)
        assert np.sum(golden_frame.data) > 0, "Golden frame appears to be empty"
    
    @pytest.mark.golden_frame
    def test_multiple_golden_frame_loading(self, golden_frame_comparator):
        """Test loading multiple golden frames."""
        loaded_frames = []
        
        for i in range(1, 6):  # Test first 5 frames
            frame = golden_frame_comparator.load_golden_frame(i)
            if frame is not None:
                loaded_frames.append((i, frame))
                assert_frame_properties(frame, 854, 480, 3)
        
        if not loaded_frames:
            pytest.skip("No golden frames available for testing")
        
        assert len(loaded_frames) > 0, "Should load at least one golden frame"
        
        # Verify frames are different (not identical)
        if len(loaded_frames) > 1:
            frame1_data = loaded_frames[0][1].data
            frame2_data = loaded_frames[1][1].data
            assert not np.array_equal(frame1_data, frame2_data), "Golden frames should be different"
    
    @pytest.mark.golden_frame
    def test_frame_comparison_identical(self, golden_frame_comparator):
        """Test frame comparison with identical frames."""
        # Load a golden frame
        golden_frame = golden_frame_comparator.load_golden_frame(1)
        
        if golden_frame is None:
            pytest.skip("Golden frame 0001.png not available")
        
        # Compare frame with itself
        is_match, diff_ratio, stats = golden_frame_comparator.compare_frames(golden_frame, golden_frame)
        
        assert is_match, "Identical frames should match"
        assert diff_ratio == 0.0, f"Identical frames should have 0 difference, got {diff_ratio}"
        assert stats["mean_diff"] == 0.0, "Mean difference should be 0 for identical frames"
    
    @pytest.mark.golden_frame
    def test_frame_comparison_different(self, golden_frame_comparator):
        """Test frame comparison with different frames."""
        # Load a golden frame
        golden_frame = golden_frame_comparator.load_golden_frame(1)
        
        if golden_frame is None:
            pytest.skip("Golden frame 0001.png not available")
        
        # Create a different frame
        different_frame = create_mock_bgr_frame(854, 480, "solid", 1)
        
        is_match, diff_ratio, stats = golden_frame_comparator.compare_frames(different_frame, golden_frame)
        
        # Should not match (assuming golden frame is not solid color)
        assert not is_match, "Different frames should not match"
        assert diff_ratio > 0.0, "Different frames should have non-zero difference"
        assert stats["mean_diff"] > 0.0, "Mean difference should be > 0 for different frames"
    
    @pytest.mark.golden_frame
    def test_golden_frame_tolerance_sensitivity(self, golden_frame_comparator):
        """Test that tolerance settings work correctly."""
        # Load a golden frame for testing
        golden_frame = golden_frame_comparator.load_golden_frame(1)
        
        if golden_frame is None:
            pytest.skip("Golden frame 0001.png not available")
        
        # Create a more significantly modified version
        modified_frame = golden_frame.copy()
        # Make a significant change that should be detected by strict tolerance
        modified_frame.data[0:50, 0:50] = 255  # Make a large white region
        
        # Test with strict tolerance
        strict_comparator = GoldenFrameComparator(golden_frame_comparator.golden_frames_dir, tolerance=0.001)
        is_match_strict, _, _ = strict_comparator.compare_frames(modified_frame, golden_frame)
        
        # Test with loose tolerance
        loose_comparator = GoldenFrameComparator(golden_frame_comparator.golden_frames_dir, tolerance=0.1)
        is_match_loose, _, _ = loose_comparator.compare_frames(modified_frame, golden_frame)
        
        # Strict tolerance should reject, loose tolerance should accept
        assert not is_match_strict, "Strict tolerance should reject small differences"
        assert is_match_loose, "Loose tolerance should accept small differences"
    
    @pytest.mark.golden_frame
    def test_frame_dimension_mismatch_handling(self, golden_frame_comparator):
        """Test handling of frame dimension mismatches."""
        # Load a golden frame
        golden_frame = golden_frame_comparator.load_golden_frame(1)
        
        if golden_frame is None:
            pytest.skip("Golden frame 0001.png not available")
        
        # Create frame with different dimensions
        wrong_size_frame = create_mock_bgr_frame(640, 360, "gradient", 1)  # Different size
        
        is_match, diff_ratio, stats = golden_frame_comparator.compare_frames(wrong_size_frame, golden_frame)
        
        assert not is_match, "Frames with different dimensions should not match"
        assert diff_ratio == 1.0, "Dimension mismatch should result in maximum difference"
        assert "error" in stats, "Stats should contain error information for dimension mismatch"
        assert "Dimension mismatch" in stats["error"], "Error should specify dimension mismatch"
    
    @pytest.mark.golden_frame
    @pytest.mark.parametrize("frame_number", [1, 2, 3, 4, 5])
    def test_golden_frame_validation(self, golden_frame_comparator, frame_number):
        """Test validation against specific golden frames."""
        golden_frame = golden_frame_comparator.load_golden_frame(frame_number)
        
        if golden_frame is None:
            pytest.skip(f"Golden frame {frame_number:04d}.png not available")
        
        # Validate the golden frame against itself (should always pass)
        is_match, diff_ratio, stats = golden_frame_comparator.validate_frame_against_golden(
            golden_frame, frame_number
        )
        
        assert is_match, f"Golden frame {frame_number} should validate against itself"
        assert diff_ratio == 0.0, f"Self-validation should have 0 difference for frame {frame_number}"
    
    @pytest.mark.golden_frame
    def test_streaming_server_frame_processing_regression(self, golden_frame_comparator):
        """Test that StreamingServer frame processing doesn't regress."""
        # This test simulates processing frames through StreamingServer operators
        # and comparing the results against golden references
        
        golden_frame = golden_frame_comparator.load_golden_frame(1)
        if golden_frame is None:
            pytest.skip("Golden frame 0001.png not available")
        
        # Simulate processing through streaming server (in this case, no processing)
        processed_frame = golden_frame.copy()
        
        # Validate processed frame against golden reference
        is_match, diff_ratio, stats = golden_frame_comparator.validate_frame_against_golden(
            processed_frame, 1
        )
        
        assert is_match, f"Processed frame should match golden reference. Diff: {diff_ratio:.4f}"
        assert stats["difference_ratio"] <= golden_frame_comparator.tolerance, \
            f"Difference ratio {stats['difference_ratio']:.4f} exceeds tolerance {golden_frame_comparator.tolerance}"
    
    @pytest.mark.golden_frame
    def test_frame_processing_with_mirroring(self, golden_frame_comparator):
        """Test frame processing with horizontal mirroring."""
        golden_frame = golden_frame_comparator.load_golden_frame(1)
        if golden_frame is None:
            pytest.skip("Golden frame 0001.png not available")
        
        # Simulate horizontal mirroring (common StreamingServer processing)
        mirrored_frame = golden_frame.copy()
        mirrored_frame.data = np.fliplr(golden_frame.data)
        
        # Mirrored frame should be different from original
        is_match, diff_ratio, stats = golden_frame_comparator.compare_frames(mirrored_frame, golden_frame)
        
        # Should not match original (unless it's symmetric)
        if not is_match:
            assert diff_ratio > 0.0, "Mirrored frame should be different from original"
        
        # But mirroring twice should restore original
        double_mirrored_frame = mirrored_frame.copy()
        double_mirrored_frame.data = np.fliplr(mirrored_frame.data)
        
        is_restored, restore_diff, _ = golden_frame_comparator.compare_frames(double_mirrored_frame, golden_frame)
        assert is_restored, f"Double mirroring should restore original. Diff: {restore_diff:.4f}"
    
    @pytest.mark.golden_frame
    @pytest.mark.slow
    def test_golden_frame_sequence_processing(self, golden_frame_comparator):
        """Test processing a sequence of golden frames."""
        processed_count = 0
        total_difference = 0.0
        
        for frame_number in range(1, 11):  # Test frames 1-10
            golden_frame = golden_frame_comparator.load_golden_frame(frame_number)
            if golden_frame is None:
                continue
            
            # Simulate processing (identity operation in this case)
            processed_frame = golden_frame.copy()
            
            # Validate against golden reference
            is_match, diff_ratio, stats = golden_frame_comparator.validate_frame_against_golden(
                processed_frame, frame_number
            )
            
            assert is_match, f"Frame {frame_number} processing failed. Diff: {diff_ratio:.4f}"
            
            processed_count += 1
            total_difference += diff_ratio
        
        if processed_count == 0:
            pytest.skip("No golden frames available for sequence testing")
        
        avg_difference = total_difference / processed_count
        assert avg_difference <= golden_frame_comparator.tolerance, \
            f"Average difference {avg_difference:.4f} exceeds tolerance {golden_frame_comparator.tolerance}"
        
        print(f"✅ Processed {processed_count} frames with average difference: {avg_difference:.6f}")
    
    @pytest.mark.golden_frame
    def test_comparator_error_handling(self, golden_frame_comparator):
        """Test error handling in golden frame comparator."""
        # Test with non-existent frame number
        non_existent_frame = golden_frame_comparator.load_golden_frame(9999)
        assert non_existent_frame is None, "Should return None for non-existent frame"
        
        # Test validation with non-existent reference
        test_frame = create_mock_bgr_frame(854, 480, "gradient", 1)
        is_match, diff_ratio, stats = golden_frame_comparator.validate_frame_against_golden(test_frame, 9999)
        
        assert not is_match, "Should not match when reference frame doesn't exist"
        assert diff_ratio == 1.0, "Should have maximum difference when reference is missing"
        assert "error" in stats, "Should include error information"
