# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
EndoNeRF Data Loader Operator

This operator loads EndoNeRF dataset frames (RGB, depth, masks) and camera poses.
Designed for simple PNG-based loading without GXF conversion.

Phase 1.1: Basic data loading and verification
"""

from pathlib import Path

import holoscan as hs
import numpy as np
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity
from PIL import Image


class EndoNeRFLoaderOp(Operator):
    """
    Load EndoNeRF dataset frames sequentially.

    Loads:
    - RGB images (PNG)
    - Depth maps (PNG)
    - Tool masks (PNG)
    - Camera poses (from poses_bounds.npy)

    Outputs one frame at a time with all associated data.
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

        # Initialize parameters with defaults (will be overridden by spec.param values)
        self.data_dir = ""
        self.downsample = 1.0
        self.loop = True
        self.max_frames = -1

        # Will be set in start()
        self.poses = None
        self.intrinsics = None
        self.image_paths = []
        self.depth_paths = []
        self.mask_paths = []
        self.num_frames = 0
        self.current_frame = 0

    def setup(self, spec: OperatorSpec):
        """Define operator interface."""
        # Outputs
        spec.output("frame_data")  # Entity with all data for current frame

        # Parameters
        spec.param("data_dir", "")  # Path to EndoNeRF/pulling directory
        spec.param("downsample", 1.0)  # Downsample factor (applied to focal length only)
        spec.param("loop", True)  # Loop back to start when all frames emitted
        spec.param("max_frames", -1)  # Max frames to emit (-1 or 0 = unlimited)

    def start(self):
        """Load dataset metadata and file lists."""
        print("[EndoNeRFLoader] Starting...")
        print(f"[EndoNeRFLoader] Data directory: {self.data_dir}")

        # Validate data directory
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        # Load camera poses
        poses_file = data_path / "poses_bounds.npy"
        if not poses_file.exists():
            raise ValueError(f"poses_bounds.npy not found in {self.data_dir}")

        print(f"[EndoNeRFLoader] Loading poses from: {poses_file}")
        self._load_poses(poses_file)

        # Get file lists
        print("[EndoNeRFLoader] Loading file lists...")
        self._load_file_lists(data_path)

        # Validate counts match
        if not (
            len(self.image_paths)
            == len(self.depth_paths)
            == len(self.mask_paths)
            == self.num_frames
        ):
            raise ValueError(
                f"Mismatch in data counts: "
                f"images={len(self.image_paths)}, "
                f"depths={len(self.depth_paths)}, "
                f"masks={len(self.mask_paths)}, "
                f"poses={self.num_frames}"
            )

        print("[EndoNeRFLoader] Dataset loaded successfully!")
        print(f"[EndoNeRFLoader]   - Frames: {self.num_frames}")
        print(
            f"[EndoNeRFLoader]   - Image size: {self.intrinsics['width']}x{self.intrinsics['height']}"
        )
        print(f"[EndoNeRFLoader]   - Focal length: {self.intrinsics['focal']:.2f}")
        print("[EndoNeRFLoader] Ready to stream!")

    def _load_poses(self, poses_file):
        """
        Load and parse poses_bounds.npy file.

        Format: [N, 17] array where each row is:
        [R(3x3) | T(3x1) | H | W | focal]
        """
        poses_arr = np.load(poses_file)
        print(f"[EndoNeRFLoader] Poses array shape: {poses_arr.shape}")

        # Parse format: [N_cams, 17] -> [N_cams, 3, 5]
        # Each 3x5 block: [R|T|HWF] where last column is [height, width, focal]
        poses_raw = poses_arr[:, :-2].reshape([-1, 3, 5])
        self.num_frames = poses_raw.shape[0]

        # Extract intrinsics from first frame (same for all frames)
        H, W, focal = poses_raw[0, :, -1]
        focal = focal / self.downsample  # Apply downsampling

        self.intrinsics = {
            "width": int(W),
            "height": int(H),
            "focal": focal,
            "K": np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32),
        }

        # Parse poses for each frame
        # EndoNeRF format: OpenGL convention (right-handed, y-up, z-back)
        # We keep it as-is for now
        self.poses = []
        for idx in range(self.num_frames):
            pose_3x4 = poses_raw[idx, :, :4]  # [R|T] 3x4 matrix

            # Build 4x4 homogeneous matrix
            c2w = np.concatenate([pose_3x4, np.array([[0, 0, 0, 1]])], axis=0)

            # Convert to w2c (world to camera)
            w2c = np.linalg.inv(c2w)

            # Extract rotation and translation
            R = w2c[:3, :3].T  # Transpose for Holoscan convention
            T = w2c[:3, 3]

            self.poses.append(
                {
                    "R": R.astype(np.float32),
                    "T": T.astype(np.float32),
                    "c2w": c2w.astype(np.float32),
                    "w2c": w2c.astype(np.float32),
                }
            )

        print(f"[EndoNeRFLoader] Loaded {len(self.poses)} camera poses")

    def _load_file_lists(self, data_path):
        """Get sorted lists of image, depth, and mask files."""
        # Images
        images_dir = data_path / "images"
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        self.image_paths = sorted(images_dir.glob("*.png"))

        # Depths
        depth_dir = data_path / "depth"
        if not depth_dir.exists():
            raise ValueError(f"Depth directory not found: {depth_dir}")
        self.depth_paths = sorted(depth_dir.glob("*.png"))

        # Masks
        masks_dir = data_path / "masks"
        if not masks_dir.exists():
            raise ValueError(f"Masks directory not found: {masks_dir}")
        self.mask_paths = sorted(masks_dir.glob("*.png"))

        print("[EndoNeRFLoader] Found files:")
        print(f"  - Images: {len(self.image_paths)}")
        print(f"  - Depths: {len(self.depth_paths)}")
        print(f"  - Masks: {len(self.mask_paths)}")

        # Determine actual number of usable frames (minimum across all data sources)
        num_poses = self.num_frames  # Set by _load_poses
        num_images = len(self.image_paths)
        num_depths = len(self.depth_paths)
        num_masks = len(self.mask_paths)

        usable_frames = min(num_poses, num_images, num_depths, num_masks)

        if usable_frames < num_poses:
            print(
                f"[EndoNeRFLoader] WARNING: Poses ({num_poses}) > available frames. "
                f"Using first {usable_frames} frames."
            )
            # Truncate to usable frames
            self.poses = self.poses[:usable_frames]
            self.image_paths = self.image_paths[:usable_frames]
            self.depth_paths = self.depth_paths[:usable_frames]
            self.mask_paths = self.mask_paths[:usable_frames]
            self.num_frames = usable_frames

        print(f"[EndoNeRFLoader] Using {usable_frames} frames")

    def compute(self, op_input, op_output, context):
        """Load and emit one frame of data."""
        # Check if we've reached the max_frames limit (hard stop)
        if self.max_frames > 0 and self.current_frame >= self.max_frames:
            return  # Stop emitting

        # Get current frame index
        if self.loop:
            # Wrap around when reaching end of dataset
            frame_idx = self.current_frame % self.num_frames
        else:
            # No wrapping - stop when we've processed all frames
            frame_idx = self.current_frame
            if frame_idx >= self.num_frames:
                return

        # DEBUG: Print every 10 frames to avoid spam
        if frame_idx % 10 == 0:
            print(f"[EndoNeRFLoader] Loading frame {frame_idx}/{self.num_frames}")

        # Load RGB image
        rgb_img = Image.open(self.image_paths[frame_idx])
        rgb_array = np.array(rgb_img).astype(np.uint8)  # [H, W, 3]

        # Load depth map
        depth_img = Image.open(self.depth_paths[frame_idx])
        depth_array = np.array(depth_img).astype(np.float32)  # [H, W] or [H, W, 1]
        if depth_array.ndim == 2:
            depth_array = depth_array[:, :, np.newaxis]  # Add channel dim

        # Load mask
        mask_img = Image.open(self.mask_paths[frame_idx])
        mask_array = np.array(mask_img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        if mask_array.ndim == 2:
            mask_array = mask_array[:, :, np.newaxis]
        # Invert mask: 1 = tool, 0 = tissue -> 0 = tool, 1 = tissue
        mask_array = 1.0 - mask_array

        # Get camera pose for this frame
        pose = self.poses[frame_idx]

        # Compute time value (normalized to [0, 1] across sequence)
        # Time is critical for temporal deformation network
        time_value = np.array([frame_idx / max(self.num_frames - 1, 1)], dtype=np.float32)

        # DEBUG: Print detailed info for first frame
        if frame_idx == 0:
            print("\n[EndoNeRFLoader] ===== FIRST FRAME DEBUG =====")
            print(f"  RGB shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")
            print(f"  RGB range: [{rgb_array.min()}, {rgb_array.max()}]")
            print(f"  Depth shape: {depth_array.shape}, dtype: {depth_array.dtype}")
            print(f"  Depth range: [{depth_array.min():.2f}, {depth_array.max():.2f}]")
            print(f"  Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
            print(f"  Mask range: [{mask_array.min():.2f}, {mask_array.max():.2f}]")
            print(f"  Mask tissue ratio: {mask_array.mean():.2%}")
            print(f"  Pose R shape: {pose['R'].shape}")
            print(f"  Pose T shape: {pose['T'].shape}")
            print(f"  Intrinsics K:\n{self.intrinsics['K']}")
            print(f"  Time value: {time_value[0]:.4f} (frame {frame_idx}/{self.num_frames-1})")
            print("[EndoNeRFLoader] =============================\n")

        # Create output entity
        out_message = Entity(context)

        # Add tensors
        out_message.add(hs.as_tensor(rgb_array), "rgb")
        out_message.add(hs.as_tensor(depth_array), "depth")
        out_message.add(hs.as_tensor(mask_array), "mask")
        out_message.add(hs.as_tensor(pose["R"]), "R")
        out_message.add(hs.as_tensor(pose["T"]), "T")
        out_message.add(hs.as_tensor(self.intrinsics["K"]), "K")
        out_message.add(hs.as_tensor(np.array([frame_idx], dtype=np.int32)), "frame_idx")
        out_message.add(hs.as_tensor(time_value), "time")  # For temporal deformation

        # Emit
        op_output.emit(out_message, "frame_data")

        # Advance to next frame
        self.current_frame += 1

    def stop(self):
        """Cleanup."""
        print(f"[EndoNeRFLoader] Stopped after processing {self.current_frame} frames")
