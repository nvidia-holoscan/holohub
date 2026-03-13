# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ast

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity


class Preprocessor3DOp(Operator):

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        spec.output("out_frames")
        spec.param("calibration_matrix")

    def initialize(self):
        self.calibration_matrix = cp.array(ast.literal_eval(self.calibration_matrix))
        self.calibration_matrix_inverse = cp.linalg.inv(self.calibration_matrix)
        self.idx = 0

    def compute(self, op_input, op_output, context):
        """
        Preprocess 2D tracks and visibility:
            - Rearrange tracks to [B N 2 T], visibility [B N 1 T]
            - Normalize tracks by camera calibration matrix
            - Concatenate tracks and visibility
        Input:
            - tracks (T, B, N, 2)
            - visibility (T, B, N, 1)
            - frames (T, H, W, 3) in [-1, 1]
            - B = 1 always
        Output:
            - tracks + visibility (B, N, 3, T)
            - frames (H, W, 3, T) in [-1, 1]
        """
        in_message = op_input.receive("in")
        if in_message is None:
            return
        tracks = cp.asarray(in_message.get("tracks"))
        visibility = cp.asarray(in_message.get("visible_tracks"))
        frames = cp.asarray(in_message.get("frames"))

        tracks = cp.transpose(tracks, (1, 2, 3, 0))
        visibility = cp.transpose(visibility, (1, 2, 3, 0))
        visibility = (visibility > 0).astype(cp.float32)

        tracks = normalize_tracks_by_calibration(self.calibration_matrix_inverse, tracks)
        out_tracks = cp.concatenate((tracks, visibility), axis=2)

        # cp.save(f"out_tracks_{self.idx}.npy", out_tracks)
        # self.idx += 1

        out_message = Entity(context)
        out_message.add(hs.as_tensor(out_tracks), "tracks")
        op_output.emit(out_message, "out")

        out_message = Entity(context)
        out_message.add(hs.as_tensor(frames), "frames")
        op_output.emit(out_message, "out_frames")


def transform_predictions(predictions, R, t, scale=1):
    """
    Apply rigid transform (R, t) to all relevant predictions using CuPy.

    Args:
        predictions: Tuple of CuPy arrays from network
        R: [B, 3, 3] rotation
        t: [B, 3] translation
    """
    # Unpack predictions
    (
        _,
        projections,
        projections_static,
        rotation_params,
        translation_params,
        B,
        points3D,
        points3D_static,
        depths,
        depths_static,
        _,
        basis_params,
        _,
        _,
        _,
        NR,
    ) = predictions

    # Transform camera poses
    # rotation_params: [B, F, 3, 3]
    new_rotation_params = cp.matmul(R[:, None, ...], rotation_params)

    # Transform translation: [B, F, 3] -> [B, F, 3]
    new_translation_params = (
        cp.matmul((scale * R)[:, None, ...], translation_params[..., None])[..., 0] + t[:, None, :]
    )

    # Transform 3D points [B, F, 3, P]
    # Work with [B, F, P, 3] then back
    points_reshaped = cp.transpose(points3D, (0, 1, 3, 2))
    Rt = cp.transpose(scale * R, (0, 2, 1))[:, None, ...]  # [B, 1, 3, 3]
    new_points_reshaped = cp.matmul(points_reshaped, Rt) + t[:, None, None, :]
    new_points3D = cp.transpose(new_points_reshaped, (0, 1, 3, 2))

    # Static points
    points_static_reshaped = cp.transpose(points3D_static, (0, 1, 3, 2))
    new_points_static_reshaped = cp.matmul(points_static_reshaped, Rt) + t[:, None, None, :]
    new_points3D_static = cp.transpose(new_points_static_reshaped, (0, 1, 3, 2))

    # Basis vectors (optional)
    new_B = None
    if B is not None:
        # B: [B, N, 3, K] -> [B, N, K, 3]
        B_reshaped = cp.transpose(B, (0, 1, 3, 2))
        new_B = cp.matmul(B_reshaped, Rt)  # rotate only
        new_B = cp.transpose(new_B, (0, 1, 3, 2))  # back to [B, N, 3, K]

    # Camera-space points: P_cam = R^T (P_world - t)
    b = int(new_rotation_params.shape[0])
    f = int(new_rotation_params.shape[1])
    n = int(new_points3D.shape[-1])

    R_bt = cp.transpose(new_rotation_params.reshape(b * f, 3, 3), (0, 2, 1))
    Pw = new_points3D.reshape(b * f, 3, n)
    tw = new_translation_params.reshape(b * f, 3)[:, :, None]
    new_points3D_camera = cp.matmul(R_bt, Pw - tw).reshape(b, f, 3, n)

    Pw_s = new_points3D_static.reshape(b * f, 3, n)
    new_points3D_static_camera = cp.matmul(R_bt, Pw_s - tw).reshape(b, f, 3, n)

    # Projections and depths
    new_projections = new_points3D_camera[..., :2, :] / (new_points3D_camera[..., 2:3, :] + 1e-8)
    new_depths = new_points3D_camera[..., 2, :]

    new_projections_static = new_points3D_static_camera[..., :2, :] / (
        new_points3D_static_camera[..., 2:3, :] + 1e-8
    )
    new_depths_static = new_points3D_static_camera[..., 2, :]

    return (
        None,
        new_projections,
        new_projections_static,
        new_rotation_params,
        new_translation_params,
        new_B,
        new_points3D,
        new_points3D_static,
        new_depths,
        new_depths_static,
        0,
        basis_params,
        0,
        0,
        None,
        NR,
    )


def compute_alignment_transform(target_R, target_t, source_R, source_t):
    """
    Compute alignment transform from source to target using overlapping frames (CuPy).

    Args:
        target_R: [B, L, 3, 3]
        target_t: [B, L, 3]
        source_R: [B, L, 3, 3]
        source_t: [B, L, 3]
    Returns:
        R: [B, 3, 3]
        t: [B, 3]
    """
    batch_size = int(target_R.shape[0])
    num_overlap = int(target_R.shape[1])

    # Build homogeneous 4x4 poses
    target_poses = cp.tile(cp.eye(4, dtype=cp.float32), (batch_size, num_overlap, 1, 1))
    source_poses = cp.tile(cp.eye(4, dtype=cp.float32), (batch_size, num_overlap, 1, 1))

    target_poses[..., :3, :3] = target_R
    target_poses[..., :3, 3] = target_t
    source_poses[..., :3, :3] = source_R
    source_poses[..., :3, 3] = source_t

    # Relative transform: target * inv(source)
    inv_source = cp.linalg.inv(source_poses)
    relative_poses = cp.matmul(target_poses, inv_source)

    middle_idx = relative_poses.shape[1] // 2
    R = relative_poses[:, middle_idx, :3, :3]
    t = relative_poses[:, middle_idx, :3, 3]

    return R, t


def normalize_tracks_by_calibration(calibration_matrix, tracks_formatted):
    """
    Normalize tracks by inverse calibration matrix.

    Args:
        calibration_matrix: [3, 3] - calibration matrix
        tracks_formatted: Tensor [B, N, 2, T] - tracks formatted for network input

    Returns:
        Tensor [B, N, 2, T] - normalized tracks
    """
    # Normalize tracks by inverse calibration matrix
    T = tracks_formatted.shape[3]  # number of frames
    for i in range(T):
        tracks_formatted[:, :, 0, i] *= calibration_matrix[0, 0]
        tracks_formatted[:, :, 0, i] += calibration_matrix[0, 2]
        tracks_formatted[:, :, 1, i] *= calibration_matrix[1, 1]
        tracks_formatted[:, :, 1, i] += calibration_matrix[1, 2]

    return tracks_formatted


class StitchPredictionsOp(Operator):
    """High-performance operator for stitching predictions.

    Inputs:
        - predictions: message with keys from tracks_4d out_tensor_names
        - track_ids: message with key "track_ids" -> cp.ndarray [P]

    Output:
        - out: message containing tensors aligned with the original operator, or
               an incremental slice if emit_incremental is True.
    """

    def __init__(self, *args, **kwargs):
        # Defaults
        self.blend_alpha = 0.5
        self.enable_forward_fill = True
        self.enable_backfill = False
        self.emit_incremental = False

        # Window tracking
        self.window_idx = 0
        self.num_frames = 0  # used frames
        self.num_points = 0  # used points (max index + 1)

        # Capacities (grow as needed)
        self.frame_capacity = 0
        self.point_capacity = 0
        self._growth_factor = 1.5

        # Allocate empty device buffers (lazy-allocated on first use)
        self.full_projections = None  # (1, Fcap, 2, Pcap)
        self.full_projections_static = None  # (1, Fcap, 2, Pcap)
        self.full_rotation_params = None  # (1, Fcap, 3, 3)
        self.full_translation_params = None  # (1, Fcap, 3)
        self.full_points3D = None  # (1, Fcap, 3, Pcap)
        self.full_points3D_static = None  # (1, Fcap, 3, Pcap)
        self.full_depths = None  # (1, Fcap, Pcap)
        self.full_depths_static = None  # (1, Fcap, Pcap)

        # Tracking arrays (device)
        self.last_update_frame = None  # (Pcap,) int64, -1 when unseen
        self.point_start_frame = None  # (Pcap,) int64, -1 when unseen
        self.seen_mask = None  # (Pcap,) bool

        # Previous indices for overlap blending
        self.prev_indices = None

        # GPU kernel for forward/backfill
        self._forward_fill_kernel = cp.RawKernel(
            r"""
        extern "C" __global__ void forward_fill(
            const long* __restrict__ last_update_frame,   // [Pcap] - only [:P] are valid
            const long* __restrict__ point_start_frame,   // [Pcap]
            float* __restrict__ proj,                     // [F,2,Pcap] flattened as ((f*2 + d)*Pcap + p)
            float* __restrict__ proj_static,              // [F,2,Pcap]
            float* __restrict__ p3d,                      // [F,3,Pcap]
            float* __restrict__ p3d_static,               // [F,3,Pcap]
            float* __restrict__ depths,                   // [F,Pcap] flattened as (f*Pcap + p)
            float* __restrict__ depths_static,            // [F,Pcap]
            const int F,
            const int Pcap,
            const int P,
            const int do_backfill)
        {
            int p = blockDim.x * blockIdx.x + threadIdx.x;
            if (p >= P) return;  // Only process used points
            long lf = last_update_frame[p];
            if (lf >= 0 && lf < F) {
                int fsrc = (int)lf;
                // Forward fill
                for (int f = fsrc + 1; f < F; ++f) {
                    // projections (2)
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        proj[( (f*2 + d) * Pcap ) + p] = proj[( (fsrc*2 + d) * Pcap ) + p];
                        proj_static[( (f*2 + d) * Pcap ) + p] = proj_static[( (fsrc*2 + d) * Pcap ) + p];
                    }
                    // points3D (3)
                    #pragma unroll
                    for (int d = 0; d < 3; ++d) {
                        p3d[( (f*3 + d) * Pcap ) + p] = p3d[( (fsrc*3 + d) * Pcap ) + p];
                        p3d_static[( (f*3 + d) * Pcap ) + p] = p3d_static[( (fsrc*3 + d) * Pcap ) + p];
                    }
                    depths[f*Pcap + p] = depths[fsrc*Pcap + p];
                    depths_static[f*Pcap + p] = depths_static[fsrc*Pcap + p];
                }
                // Optional backfill
                if (do_backfill) {
                    long sf = point_start_frame[p];
                    if (sf >= 0 && sf < fsrc) {
                        for (int f = (int)sf; f < fsrc; ++f) {
                            #pragma unroll
                            for (int d = 0; d < 2; ++d) {
                                proj[( (f*2 + d) * Pcap ) + p] = proj[( (fsrc*2 + d) * Pcap ) + p];
                                proj_static[( (f*2 + d) * Pcap ) + p] = proj_static[( (fsrc*2 + d) * Pcap ) + p];
                            }
                            #pragma unroll
                            for (int d = 0; d < 3; ++d) {
                                p3d[( (f*3 + d) * Pcap ) + p] = p3d[( (fsrc*3 + d) * Pcap ) + p];
                                p3d_static[( (f*3 + d) * Pcap ) + p] = p3d_static[( (fsrc*3 + d) * Pcap ) + p];
                            }
                            depths[f*Pcap + p] = depths[fsrc*Pcap + p];
                            depths_static[f*Pcap + p] = depths_static[fsrc*Pcap + p];
                        }
                    }
                }
            }
        }
        """,
            name="forward_fill",
        )
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("predictions")
        spec.input("track_ids")
        spec.output("out")

        # Window params
        spec.param("window_size")
        spec.param("overlap_size")

        # Performance/behavior toggles
        spec.param("enable_forward_fill")
        spec.param("enable_backfill")
        spec.param("emit_incremental")

    def _round_up_capacity(self, needed: int, current: int) -> int:
        if current <= 0:
            return max(16, needed)
        cap = current
        while cap < needed:
            cap = int(cap * self._growth_factor) + 1
        return cap

    def _allocate_or_grow(self, new_frame_needed: int, new_point_needed: int):
        # Determine new capacities
        new_fcap = self.frame_capacity
        new_pcap = self.point_capacity
        if new_frame_needed > self.frame_capacity:
            new_fcap = self._round_up_capacity(new_frame_needed, self.frame_capacity)
        if new_point_needed > self.point_capacity:
            new_pcap = self._round_up_capacity(new_point_needed, self.point_capacity)

        if new_fcap == self.frame_capacity and new_pcap == self.point_capacity:
            return

        # Allocate new buffers on device
        b = 1

        def alloc_like(shape, dtype, fill_zero=True):
            return cp.zeros(shape, dtype=dtype) if fill_zero else cp.empty(shape, dtype=dtype)

        new_full_projections = alloc_like((b, new_fcap, 2, new_pcap), cp.float32)
        new_full_projections_static = alloc_like((b, new_fcap, 2, new_pcap), cp.float32)
        new_full_rotation_params = alloc_like((b, new_fcap, 3, 3), cp.float32)
        new_full_translation_params = alloc_like((b, new_fcap, 3), cp.float32)
        new_full_points3D = alloc_like((b, new_fcap, 3, new_pcap), cp.float32)
        new_full_points3D_static = alloc_like((b, new_fcap, 3, new_pcap), cp.float32)
        new_full_depths = alloc_like((b, new_fcap, new_pcap), cp.float32)
        new_full_depths_static = alloc_like((b, new_fcap, new_pcap), cp.float32)

        new_last_update_frame = cp.full((new_pcap,), -1, dtype=cp.int64)
        new_point_start_frame = cp.full((new_pcap,), -1, dtype=cp.int64)
        new_seen_mask = cp.zeros((new_pcap,), dtype=cp.bool_)

        # Copy existing used region
        if (
            self.frame_capacity > 0
            and self.point_capacity > 0
            and self.num_frames > 0
            and self.num_points > 0
        ):
            f_slice = slice(0, self.num_frames)
            p_slice = slice(0, self.num_points)
            new_full_projections[:, f_slice, :, p_slice] = self.full_projections[
                :, f_slice, :, p_slice
            ]
            new_full_projections_static[:, f_slice, :, p_slice] = self.full_projections_static[
                :, f_slice, :, p_slice
            ]
            new_full_rotation_params[:, f_slice] = self.full_rotation_params[:, f_slice]
            new_full_translation_params[:, f_slice] = self.full_translation_params[:, f_slice]
            new_full_points3D[:, f_slice, :, p_slice] = self.full_points3D[:, f_slice, :, p_slice]
            new_full_points3D_static[:, f_slice, :, p_slice] = self.full_points3D_static[
                :, f_slice, :, p_slice
            ]
            new_full_depths[:, f_slice, p_slice] = self.full_depths[:, f_slice, p_slice]
            new_full_depths_static[:, f_slice, p_slice] = self.full_depths_static[
                :, f_slice, p_slice
            ]

            new_last_update_frame[: self.last_update_frame.shape[0]] = self.last_update_frame
            new_point_start_frame[: self.point_start_frame.shape[0]] = self.point_start_frame
            new_seen_mask[: self.seen_mask.shape[0]] = self.seen_mask

        # Swap in
        self.full_projections = new_full_projections
        self.full_projections_static = new_full_projections_static
        self.full_rotation_params = new_full_rotation_params
        self.full_translation_params = new_full_translation_params
        self.full_points3D = new_full_points3D
        self.full_points3D_static = new_full_points3D_static
        self.full_depths = new_full_depths
        self.full_depths_static = new_full_depths_static

        self.last_update_frame = new_last_update_frame
        self.point_start_frame = new_point_start_frame
        self.seen_mask = new_seen_mask

        self.frame_capacity = new_fcap
        self.point_capacity = new_pcap

    def _copy_predictions(self, preds, indices, frame_start, frame_end, src_start=0):
        # Determine src_end based on desired frame range, clip to actual size
        src_end = src_start + (frame_end - frame_start)
        src_end = int(min(src_end, int(preds[1].shape[1])))

        # Copy point-indexed predictions
        self.full_projections[:, frame_start:frame_end, :][..., indices] = preds[1][
            :, src_start:src_end
        ]
        self.full_projections_static[:, frame_start:frame_end, :][..., indices] = preds[2][
            :, src_start:src_end
        ]
        self.full_points3D[:, frame_start:frame_end, :][..., indices] = preds[6][
            :, src_start:src_end
        ]
        self.full_points3D_static[:, frame_start:frame_end, :][..., indices] = preds[7][
            :, src_start:src_end
        ]
        self.full_depths[:, frame_start:frame_end][..., indices] = preds[8][:, src_start:src_end]
        self.full_depths_static[:, frame_start:frame_end][..., indices] = preds[9][
            :, src_start:src_end
        ]

        # Copy camera params
        self.full_rotation_params[:, frame_start:frame_end] = preds[3][:, src_start:src_end]
        self.full_translation_params[:, frame_start:frame_end] = preds[4][:, src_start:src_end]

        # Update last_update_frame where this window has data
        has_data = cp.any(preds[8][:, src_start:src_end] != 0, axis=(0, 1))  # [Pwindow]
        if has_data.size > 0:
            points_with_data = indices[has_data]
            if points_with_data.size > 0:
                self.last_update_frame[points_with_data] = cp.maximum(
                    self.last_update_frame[points_with_data],
                    cp.asarray(frame_end - 1, dtype=cp.int64),
                )

    def _blend_camera_params(self, preds, overlap_start, overlap_end, alpha):
        self.full_rotation_params[:, overlap_start:overlap_end] = (
            self.full_rotation_params[:, overlap_start:overlap_end]
            + preds[3][:, : self.overlap_size]
        ) / 2.0

        self.full_translation_params[:, overlap_start:overlap_end] = (
            alpha * self.full_translation_params[:, overlap_start:overlap_end]
            + (1.0 - alpha) * preds[4][:, : self.overlap_size]
        )

    def _blend_overlap(self, preds, common_indices, mask, overlap_start, overlap_end, alpha):
        self.full_projections[:, overlap_start:overlap_end, ...][..., common_indices] = (
            alpha * self.full_projections[:, overlap_start:overlap_end, ...][..., common_indices]
            + (1.0 - alpha) * preds[1][:, : self.overlap_size, ...][..., mask]
        )
        self.full_projections_static[:, overlap_start:overlap_end, ...][..., common_indices] = (
            alpha
            * self.full_projections_static[:, overlap_start:overlap_end, ...][..., common_indices]
            + (1.0 - alpha) * preds[2][:, : self.overlap_size, ...][..., mask]
        )
        self.full_points3D[:, overlap_start:overlap_end, ...][..., common_indices] = (
            alpha * self.full_points3D[:, overlap_start:overlap_end, ...][..., common_indices]
            + (1.0 - alpha) * preds[6][:, : self.overlap_size, ...][..., mask]
        )
        self.full_points3D_static[:, overlap_start:overlap_end, ...][..., common_indices] = (
            alpha
            * self.full_points3D_static[:, overlap_start:overlap_end, ...][..., common_indices]
            + (1.0 - alpha) * preds[7][:, : self.overlap_size, ...][..., mask]
        )
        self.full_depths[:, overlap_start:overlap_end, ...][..., common_indices] = (
            alpha * self.full_depths[:, overlap_start:overlap_end, ...][..., common_indices]
            + (1.0 - alpha) * preds[8][:, : self.overlap_size, ...][..., mask]
        )
        self.full_depths_static[:, overlap_start:overlap_end, ...][..., common_indices] = (
            alpha * self.full_depths_static[:, overlap_start:overlap_end, ...][..., common_indices]
            + (1.0 - alpha) * preds[9][:, : self.overlap_size, ...][..., mask]
        )

    def _forward_fill_gpu(self):
        if not self.enable_forward_fill:
            return
        F = int(self.num_frames)
        P = int(self.num_points)
        Pcap = int(self.point_capacity)
        if F <= 0 or P <= 0:
            return
        # Use full capacity arrays (contiguous) - kernel will only touch [:F, :P]
        # Slicing with [:P] on the last dim creates non-contiguous views, and
        # reshape(-1) on non-contiguous arrays creates copies, causing the kernel
        # to write to temporary buffers that are discarded.
        proj = self.full_projections[0, :F, :, :]  # (F, 2, Pcap) - contiguous
        proj_static = self.full_projections_static[0, :F, :, :]
        p3d = self.full_points3D[0, :F, :, :]  # (F, 3, Pcap)
        p3d_static = self.full_points3D_static[0, :F, :, :]
        depths = self.full_depths[0, :F, :]  # (F, Pcap)
        depths_static = self.full_depths_static[0, :F, :]

        threads = 256
        blocks = (P + threads - 1) // threads  # Only iterate over used points

        self._forward_fill_kernel(
            (blocks,),
            (threads,),
            (
                self.last_update_frame,  # Full array, kernel only reads [:P]
                self.point_start_frame,
                proj.reshape(-1),  # Now contiguous, reshape returns a view
                proj_static.reshape(-1),
                p3d.reshape(-1),
                p3d_static.reshape(-1),
                depths.reshape(-1),
                depths_static.reshape(-1),
                F,
                Pcap,
                P,
                1 if self.enable_backfill else 0,
            ),
        )

    def _get_predictions(self):
        F = slice(0, self.num_frames)
        P = slice(0, self.num_points)
        indices = (
            cp.arange(self.num_points, dtype=cp.int64)
            if self.num_points > 0
            else cp.asarray([], dtype=cp.int64)
        )
        return (
            None,  # focal_params
            self.full_projections[:, F, :, P],
            self.full_projections_static[:, F, :, P],
            self.full_rotation_params[:, F],
            self.full_translation_params[:, F],
            None,  # B
            self.full_points3D[:, F, :, P],
            self.full_points3D_static[:, F, :, P],
            self.full_depths[:, F, P],
            self.full_depths_static[:, F, P],
            0,
            None,  # basis_params
            0,
            0,
            None,  # points3D_camera
            None,  # NR
            indices,
        )

    def add_window(self, predictions, indices, start_frame):
        # Actual window size from predictions
        actual_window_size = int(predictions[1].shape[1])

        # Determine required used extents
        new_max_idx = int(cp.max(indices).item()) + 1 if indices.size > 0 else int(self.num_points)
        end_frame = start_frame + actual_window_size
        new_total_frames = max(end_frame, int(self.num_frames))

        # Ensure capacity before writes
        self._allocate_or_grow(new_total_frames, new_max_idx)

        # Track first-frame for newly seen points on device
        if indices.size > 0:
            seen_before = self.seen_mask[indices]
            new_mask = ~seen_before
            if cp.any(new_mask):
                new_indices = indices[new_mask]
                self.point_start_frame[new_indices] = cp.asarray(start_frame, dtype=cp.int64)
                self.seen_mask[new_indices] = True

        # Update used extents
        self.num_frames = new_total_frames
        self.num_points = new_max_idx

        if self.window_idx == 0:
            self._copy_predictions(
                predictions, indices, frame_start=start_frame, frame_end=end_frame
            )
        else:
            overlap_start = start_frame
            overlap_end = start_frame + int(self.overlap_size)

            R, t = compute_alignment_transform(
                self.full_rotation_params[:, overlap_start:overlap_end],
                self.full_translation_params[:, overlap_start:overlap_end],
                predictions[3][:, : int(self.overlap_size)],
                predictions[4][:, : int(self.overlap_size)],
            )

            aligned_preds = transform_predictions(predictions, R, t, scale=1)

            # Copy non-overlapping slice
            self._copy_predictions(
                aligned_preds,
                indices,
                frame_start=start_frame + int(self.overlap_size),
                frame_end=end_frame,
                src_start=int(self.overlap_size),
            )

            # Blend overlapping region
            alpha = self.blend_alpha
            self._blend_camera_params(aligned_preds, overlap_start, overlap_end, alpha)

            if self.prev_indices is not None and self.prev_indices.size > 0 and indices.size > 0:
                _mask = cp.isin(indices, self.prev_indices)
                common_indices = indices[_mask]
                if common_indices.size > 0:
                    self._blend_overlap(
                        aligned_preds, common_indices, _mask, overlap_start, overlap_end, alpha
                    )

        self.prev_indices = indices
        self.window_idx += 1

        # GPU forward/backfill over used extents
        self._forward_fill_gpu()

        return self._get_predictions()

    def _emit_all_predictions(self, op_output, context, out_preds, indices):
        out_message = Entity(context)
        if out_preds[0] is not None:
            out_message.add(hs.as_tensor(out_preds[0]), "focal_params")

        out_message.add(hs.as_tensor(out_preds[1]), "projections2")
        out_message.add(hs.as_tensor(out_preds[2]), "projections2_static")
        out_message.add(hs.as_tensor(out_preds[3]), "rotation_params")
        out_message.add(hs.as_tensor(out_preds[4]), "position_params")

        if out_preds[5] is not None:
            out_message.add(hs.as_tensor(out_preds[5]), "B")

        out_message.add(hs.as_tensor(out_preds[6]), "points3D")
        out_message.add(hs.as_tensor(out_preds[7]), "points3D_static")
        out_message.add(hs.as_tensor(out_preds[8]), "depths")
        out_message.add(hs.as_tensor(out_preds[9]), "depths_static")

        out_message.add(hs.as_tensor(cp.asarray(out_preds[10])), "unused1")

        if out_preds[11] is not None:
            out_message.add(hs.as_tensor(out_preds[11]), "basis_params")

        out_message.add(hs.as_tensor(cp.asarray(out_preds[12])), "unused2")
        out_message.add(hs.as_tensor(cp.asarray(out_preds[13])), "unused3")

        if out_preds[14] is not None:
            out_message.add(hs.as_tensor(out_preds[14]), "points3D_camera")
        if out_preds[15] is not None:
            out_message.add(hs.as_tensor(out_preds[15]), "NR")

        out_message.add(hs.as_tensor(out_preds[16]), "indices")
        out_message.add(hs.as_tensor(indices), "track_ids")

        op_output.emit(out_message, "out")

    def _emit_necessary_predictions(
        self, op_output, context, out_preds, indices, start_frame, end_frame
    ):
        # Optionally emit only the incremental slice (after overlap)
        if self.emit_incremental and self.window_idx > 1:
            inc_start = start_frame + int(self.overlap_size)
            inc_end = end_frame
            depths_slice = out_preds[8][:, inc_start:inc_end]
        else:
            depths_slice = out_preds[8]

        out_message = Entity(context)
        out_message.add(hs.as_tensor(depths_slice), "depths")
        out_message.add(hs.as_tensor(indices), "track_ids")
        out_message.add(hs.as_tensor(cp.asarray(out_preds[3])), "rotation_params")
        out_message.add(hs.as_tensor(cp.asarray(out_preds[4])), "position_params")
        out_message.add(
            hs.as_tensor(cp.asarray(out_preds[7])), "points3D"
        )  # These are actually the static points3D
        op_output.emit(out_message, "out")

    def compute(self, op_input, op_output, context):
        pred_msg = op_input.receive("predictions")
        ids_msg = op_input.receive("track_ids")

        # Extract predictions
        projections2 = cp.asarray(pred_msg.get("projections2"))
        projections2_static = cp.asarray(pred_msg.get("projections2_static"))
        rotation_params = cp.asarray(pred_msg.get("rotation_params"))
        position_params = cp.asarray(pred_msg.get("position_params"))
        points3D = cp.asarray(pred_msg.get("points3D"))
        points3D_static = cp.asarray(pred_msg.get("points3D_static"))
        depths = cp.asarray(pred_msg.get("depths"))
        depths_static = cp.asarray(pred_msg.get("depths_static"))

        B = (
            cp.asarray(pred_msg.get("B"))
            if "B" in pred_msg and pred_msg.get("B") is not None
            else None
        )
        basis_params = (
            cp.asarray(pred_msg.get("basis_params"))
            if "basis_params" in pred_msg and pred_msg.get("basis_params") is not None
            else None
        )
        points3D_camera = (
            cp.asarray(pred_msg.get("points3D_camera"))
            if "points3D_camera" in pred_msg and pred_msg.get("points3D_camera") is not None
            else None
        )
        NR = (
            cp.asarray(pred_msg.get("NR"))
            if "NR" in pred_msg and pred_msg.get("NR") is not None
            else None
        )

        preds = (
            None,
            projections2,
            projections2_static,
            rotation_params,
            position_params,
            B,
            points3D,
            points3D_static,
            depths,
            depths_static,
            0,
            basis_params,
            0,
            0,
            points3D_camera,
            NR,
        )

        indices = cp.asarray(ids_msg.get("track_ids"))

        # Compute start frame from window index and window parameters
        start_frame = int(self.window_idx * (int(self.window_size) - int(self.overlap_size)))

        out_preds = self.add_window(preds, indices, start_frame)

        end_frame = start_frame + int(preds[1].shape[1])
        self._emit_necessary_predictions(
            op_output, context, out_preds, indices, start_frame, end_frame
        )


class Postprocess3DOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("predictions")
        spec.input("tracks")
        spec.input("frames")
        spec.output("out")

        # Windowing parameters provided by config (window section)
        spec.param("window_size")
        spec.param("overlap_size")
        spec.param("calibration_matrix")

    def initialize(self):
        self.window_idx = 0
        self.advance_size = int(self.window_size) - int(self.overlap_size)
        self.calibration_matrix = cp.array(ast.literal_eval(self.calibration_matrix))

    def compute(self, op_input, op_output, context):
        pred_msg = op_input.receive(
            "predictions"
        )  # ['depths', 'track_ids', 'rotation_params', 'position_params']
        depths = cp.asarray(pred_msg.get("depths"))
        indices = cp.asarray(pred_msg.get("track_ids"))
        camera_position = cp.asarray(pred_msg.get("position_params"))  # b t 3
        camera_rotation = cp.asarray(pred_msg.get("rotation_params"))  # b t 3 3
        points3D = cp.asarray(pred_msg.get("points3D"))  # b t 3 n
        pred_tracks = op_input.receive("tracks")
        tracks = cp.asarray(pred_tracks.get("tracks"))  # b n c t
        frames_msg = op_input.receive("frames")
        frames = cp.asarray(frames_msg.get("frames"))

        tracks_2d = tracks[:, :, :2]
        visibility = tracks[:, :, 2:]

        tracks_2d = normalize_tracks_by_calibration(self.calibration_matrix, tracks_2d)

        window_start = int(self.window_idx * self.advance_size)
        T_actual = tracks.shape[3]  # tracks shape is [B, N, C, T]
        window_end = window_start + T_actual

        depths_window = depths[:, window_start:window_end, indices]
        depths_reshaped = depths_window.transpose(0, 2, 1)
        depths_reshaped = depths_reshaped[:, :, None, :]  # [B, N, 1, T_window]

        tracks_with_depth = cp.concatenate(
            (tracks_2d, depths_reshaped), axis=2
        )  # [B, N, 3, T_window]

        self.window_idx += 1

        camera_position_window = camera_position[0, window_start:window_end]
        camera_rotation_window = camera_rotation[0, window_start:window_end]

        out_message = Entity(context)
        out_message.add(hs.as_tensor(tracks_with_depth), "tracks_with_depth")
        out_message.add(hs.as_tensor(visibility), "visible_tracks")
        out_message.add(hs.as_tensor(frames), "frames")
        out_message.add(hs.as_tensor(camera_position), "camera_position")
        out_message.add(hs.as_tensor(camera_position_window), "camera_position_window")
        out_message.add(hs.as_tensor(points3D), "points3D")
        out_message.add(hs.as_tensor(camera_rotation_window), "camera_rotation")
        op_output.emit(out_message, "out")
