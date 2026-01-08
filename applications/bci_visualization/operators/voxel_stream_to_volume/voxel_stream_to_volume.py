"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0

VoxelStreamToVolume operator: converts streaming voxel data to dense 3D volume.
"""

import cupy as cp
import cupyx.scipy.ndimage
import nibabel as nib
import numpy as np
from holoscan.core import ConditionType, Operator, OperatorSpec
from nibabel.orientations import aff2axcodes


class VoxelStreamToVolumeOp(Operator):
    """
    Convert streaming HbO/HbR voxel data [I, J, K, 2] into a 3D volume tensor for VolumeRendererOp.

    Inputs:
    - affine_4x4: np.ndarray shape (4, 4) (processed once if provided)
    - hb_voxel_data: np.ndarray shape (I, J, K, n_channels) where last dim is channels [HbO, HbR] (HbO: 0, HbR: 1)

    Outputs:
    - volume: holoscan.gxf.Entity containing a tensor named "volume" with shape (Z,Y,X)
    - spacing: np.ndarray shape (3,) derived from affine
    - permute_axis: np.ndarray shape (3,) derived from affine
    - flip_axes: np.ndarray shape (3,) derived from affine
    """

    def __init__(self, fragment, *args, **kwargs):
        # Anatomy mask NIfTI file
        self.mask_nifti_path = kwargs.pop("mask_nifti_path", None)

        # Exponential moving average factor for running statistics (0 < alpha <= 1)
        # Higher alpha = faster adaptation, lower alpha = more stable
        self.stats_alpha = kwargs.pop("stats_alpha", 0.1)

        # Visualization scale factor for amplifying activations
        # Needed because global min/max includes the whole brain (larger values),
        # but we only visualize white/gray matter (smaller activations).
        # Higher scale = more sensitive visualization of small activations
        self.visualization_scale = kwargs.pop("visualization_scale", 10)

        # Density range, must be same as the VolumeRendererOp's density range
        self.density_min = kwargs.pop("density_min", -100)
        self.density_max = kwargs.pop("density_max", 100)

        super().__init__(fragment, *args, **kwargs)

        # Internal state
        self.affine = None

        # Metadata, set from the first frame, reused for subsequent frames
        self.dims = None  # np.array([X, Y, Z], dtype=np.uint32)
        self.out_spacing = None  # np.ndarray float32 (3,)
        self.permute_axis = None  # np.ndarray uint32 (3,)
        self.flip_axes = None  # np.ndarray bool (3,)
        self.roi_mask = None  # np.ndarray bool (I, J, K)

        # Raw incoming mask (I, J, K) for pass-through emission (loaded from file if provided)
        self.mask_voxel_raw = None
        self.mask_volume_gpu = None
        self.mask_affine = None
        self.mask_shape = None

        # Running statistics for adaptive normalization (initialized from first frame)
        self.global_min = None
        self.global_max = None
        self.frame_count = 0

    def start(self):
        if not self.mask_nifti_path:
            raise ValueError("VoxelStreamToVolume: No mask NIfTI path provided")

        try:
            img = nib.load(self.mask_nifti_path)
            mask_3d = img.get_fdata()
            # Segmentation volumes must be unsigned 8-bit integer
            self.mask_voxel_raw = np.asarray(mask_3d, dtype=np.uint8)
            self.mask_affine = img.affine
            self.mask_shape = mask_3d.shape
            print(
                f"VoxelStreamToVolume: Loaded mask from {self.mask_nifti_path}, "
                f"shape: {self.mask_voxel_raw.shape}, values: {np.unique(self.mask_voxel_raw)}"
            )
        except Exception as e:
            raise RuntimeError(
                f"VoxelStreamToVolume: Failed to load mask NIfTI '{self.mask_nifti_path}': {e}"
            ) from e

    def setup(self, spec: OperatorSpec):
        spec.input("affine_4x4").condition(
            ConditionType.NONE
        )  # (4, 4), only emit at the first frame
        spec.input("hb_voxel_data")  # (I, J, K)

        spec.output("volume")
        spec.output("spacing")
        spec.output("permute_axis")
        spec.output("flip_axes")

        # brain anatomy mask
        spec.output("mask_volume").condition(ConditionType.NONE)
        spec.output("mask_spacing").condition(ConditionType.NONE)
        spec.output("mask_permute_axis").condition(ConditionType.NONE)
        spec.output("mask_flip_axes").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        # Receive Hb voxel data (cupy array)
        hb_voxel = op_input.receive("hb_voxel_data")  # (I, J, K)
        cuda_stream = op_input.receive_cuda_stream("hb_voxel_data")

        # Check voxel data is valid
        if not isinstance(hb_voxel, cp.ndarray):
            raise TypeError(
                f"VoxelStreamToVolume: Invalid voxel data type: {type(hb_voxel)}, expected cupy array"
            )
        if hb_voxel.ndim != 3:
            raise ValueError(
                f"VoxelStreamToVolume: Invalid voxel data shape: {hb_voxel.shape}, expected 3D"
            )

        # Receive affine matrix only at the first frame
        affine = op_input.receive("affine_4x4")
        if affine is not None:
            self.affine = np.array(affine, dtype=np.float32).reshape(4, 4)
            # Derive spacing/orientation from affine - use mask's affine as we will resample data to mask's size
            self.out_spacing, self.permute_axis, self.flip_axes = (
                self._derive_orientation_from_affine(self.mask_affine)
            )
            print("VoxelStreamToVolume: Received affine matrix")

        # Check if affine has been set at least once
        if self.affine is None:
            raise ValueError("VoxelStreamToVolume: No affine matrix received")

        with cp.cuda.ExternalStream(cuda_stream):
            # Update running statistics from incoming data
            self._update_running_statistics(hb_voxel)

            # Note: +-1 to add a buffer avoiding edge case in ClaraViz boundaries.
            hb_voxel_normalized = self._normalize_and_process_activated_voxels(
                hb_voxel,
                normalize_min_value=self.density_min + 1,
                normalize_max_value=self.density_max - 1,
            )

            # Resample to mask's size
            volume_gpu = self._cupy_resample(
                hb_voxel_normalized, self.affine, self.mask_affine, self.mask_shape
            )

            volume_gpu = cp.transpose(volume_gpu, (2, 1, 0))
            volume_gpu = cp.ascontiguousarray(volume_gpu, dtype=cp.float32)

        # If we have a mask, emit oriented mask every frame for the renderer
        if self.mask_volume_gpu is None:
            with cp.cuda.ExternalStream(cuda_stream):
                self.mask_volume_gpu = cp.asarray(self.mask_voxel_raw, dtype=cp.uint8)
                self.mask_volume_gpu = cp.transpose(self.mask_volume_gpu, (2, 1, 0))
                self.mask_volume_gpu = cp.ascontiguousarray(self.mask_volume_gpu)

        # Emit mask outputs
        op_output.emit({"volume": self.mask_volume_gpu}, "mask_volume")
        op_output.emit(self.out_spacing, "mask_spacing", "std::array<float, 3>")
        op_output.emit(self.permute_axis, "mask_permute_axis", "std::array<uint32_t, 3>")
        op_output.emit(self.flip_axes, "mask_flip_axes", "std::array<bool, 3>")

        # Emit density outputs
        op_output.emit({"volume": volume_gpu}, "volume")
        op_output.emit(self.out_spacing, "spacing", "std::array<float, 3>")
        op_output.emit(self.permute_axis, "permute_axis", "std::array<uint32_t, 3>")
        op_output.emit(self.flip_axes, "flip_axes", "std::array<bool, 3>")

    def _derive_orientation_from_affine(self, affine_4x4: np.ndarray):
        """
        Derive spacing, axis permutation, and flips from affine.
        spacing: voxel sizes along data axes (I,J,K) mapped to [X,Y,Z] ordering
        permute_axis: for each data axis (I,J,K), index of world axis (X=0,Y=1,Z=2)
        flip_axes: whether the axis is flipped (negative orientation)
        """

        R = affine_4x4[:3, :3].astype(np.float32)
        # spacing along data axes (length of each column)
        spacing_ijk = np.linalg.norm(R, axis=0).astype(np.float32)
        # Avoid zeros
        spacing_ijk[spacing_ijk == 0] = 1.0

        # 1. Get the Orientation String from Nibabel
        # nibabel returns where the axis points TO (e.g., 'RAS')
        orientation_codes = aff2axcodes(affine_4x4)
        print(f"Detected Orientation: {''.join(orientation_codes)}")

        # 2. Parse orientation codes to determine axis assignment and flips
        # Nibabel convention: codes indicate the direction each axis points TO
        # Flip is needed when axis points in the negative direction (L, P, I)
        rl_axis = 4
        is_axis = 4
        pa_axis = 4

        rl_flip = False
        is_flip = False
        pa_flip = False

        # Iterate through the codes (0=x, 1=y, 2=z in data array)
        for axis, code in enumerate(orientation_codes):
            # --- Right-Left Axis ---
            if code in ["R", "r"]:
                rl_axis = axis
                rl_flip = False  # Points right (positive direction)
            elif code in ["L", "l"]:
                rl_axis = axis
                rl_flip = True  # Points left (negative direction, needs flip)

            # --- Inferior-Superior Axis ---
            elif code in ["S", "s"]:
                is_axis = axis
                is_flip = False  # Points superior (positive direction)
            elif code in ["I", "i"]:
                is_axis = axis
                is_flip = True  # Points inferior (negative direction, needs flip)

            # --- Posterior-Anterior Axis ---
            elif code in ["A", "a"]:
                pa_axis = axis
                pa_flip = False  # Points anterior (positive direction)
            elif code in ["P", "p"]:
                pa_axis = axis
                pa_flip = True  # Points posterior (negative direction, needs flip)

        # Validation
        if 4 in [
            rl_axis,
            is_axis,
            pa_axis,
        ]:  # 4 is a sentinel to indicate any axis that was not set
            raise ValueError(
                f"Could not determine all axes from orientation: {''.join(orientation_codes)}"
            )

        # 3. Construct the final parameters
        permute = [rl_axis, is_axis, pa_axis]
        flips = [rl_flip, is_flip, pa_flip]

        # spacing returned in [X, Y, Z] order by mapping data spacings
        spacing_xyz = np.zeros(3, dtype=np.float32)
        for a in range(3):
            spacing_xyz[permute[a]] = spacing_ijk[a]

        return spacing_xyz, permute, flips

    def _update_running_statistics(self, hb_voxel: cp.ndarray):
        """
        Update running min/max statistics using exponential moving average.
        Initializes from first frame with valid data.

        Args:
            hb_voxel: Current voxel data (cupy array)
        """
        self.frame_count += 1

        # Compute current frame statistics from all voxels
        current_min = float(cp.min(hb_voxel))
        current_max = float(cp.max(hb_voxel))

        # Initialize on first frame
        if (
            (self.global_min is None or self.global_max is None)
            and current_min != 0
            and current_max != 0
        ):
            self.global_min = current_min
            self.global_max = current_max
            print(
                f"VoxelStreamToVolume: Initialized statistics from first frame - "
                f"min={self.global_min:.6f}, max={self.global_max:.6f}"
            )
            return

        # Use exponential moving average for smooth adaptation
        # For first few frames, use larger alpha for faster convergence
        alpha = self.stats_alpha if self.frame_count > 10 else 0.3

        # Update running statistics
        self.global_min = (1 - alpha) * self.global_min + alpha * current_min
        self.global_max = (1 - alpha) * self.global_max + alpha * current_max

        # Log statistics every 100 frames for debugging
        if self.frame_count % 10 == 0:
            print(
                f"VoxelStreamToVolume: Frame {self.frame_count} - "
                f"Running stats: min={self.global_min:.6f}, max={self.global_max:.6f} "
                f"(current: min={current_min:.6f}, max={current_max:.6f})"
            )

    def _normalize_and_process_activated_voxels(
        self, hb_voxel: np.ndarray, normalize_min_value: float, normalize_max_value: float
    ):
        """
        Normalize the volume to [normalize_min_value, normalize_max_value] while preserving 0 as baseline.
        Applies visualization scale factor to amplify small activations in white/gray matter.
        """
        # If statistics not initialized yet, return zeros (waiting for first valid frame)
        if self.global_min is None or self.global_max is None:
            print("VoxelStreamToVolume: Waiting for statistics initialization...")
            return cp.zeros_like(hb_voxel, dtype=cp.float32)

        # Step 1/2: Normalize while preserving 0 as baseline.
        hb = hb_voxel.astype(cp.float32, copy=False)
        hb_voxel_normalized = cp.zeros_like(hb, dtype=cp.float32)

        if self.global_max > 0:
            # Apply visualization scale to amplify small activations
            # Global max includes whole brain, but we visualize white/gray matter (smaller values)
            pos_scale = (
                float(normalize_max_value) / float(self.global_max)
            ) * self.visualization_scale
            pos_mask = hb >= 0
            hb_voxel_normalized[pos_mask] = hb[pos_mask] * pos_scale

        if self.global_min < 0:
            # Apply visualization scale to amplify small deactivations
            neg_scale = (
                float(abs(normalize_min_value)) / float(abs(self.global_min))
            ) * self.visualization_scale
            neg_mask = hb < 0
            hb_voxel_normalized[neg_mask] = hb[neg_mask] * neg_scale

        # Step 3: Clip in-place to ensure values stay in range.
        cp.clip(
            hb_voxel_normalized, normalize_min_value, normalize_max_value, out=hb_voxel_normalized
        )

        return hb_voxel_normalized

    def _cupy_resample(self, data_gpu, src_affine, target_affine, target_shape):
        # 1. Calculate the transform matrix (Target -> Source)
        inv_src_affine = np.linalg.inv(src_affine)
        mapping_matrix = inv_src_affine @ target_affine

        # Extract the rotation/scaling (3x3) and translation parts
        # SciPy/CuPy affine_transform expects: input_coords = matrix @ output_coords + offset
        matrix = mapping_matrix[:3, :3]
        offset = mapping_matrix[:3, 3]

        # 2. Move data to GPU
        data_gpu = cp.asarray(data_gpu, dtype=cp.float32)
        matrix_gpu = cp.asarray(matrix)
        offset_gpu = cp.asarray(offset)

        # 3. Resample
        resampled_gpu = cupyx.scipy.ndimage.affine_transform(
            data_gpu,
            matrix=matrix_gpu,
            offset=offset_gpu,
            output_shape=target_shape,
            order=1,
        )

        return resampled_gpu
