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

import cupy as cp
import holoscan as hs
from holoscan.core import ConditionType, Operator, OperatorSpec
from holoscan.gxf import Entity


class PostprocessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("inference_result").condition(ConditionType.NONE)
        spec.input("track_ids").condition(ConditionType.NONE)
        spec.output("output")
        spec.param("overlap_size")

    def initialize(self):
        self.output_queue = []
        self.idx = 0
        # Buffers for the current batch
        self._batch_frames = None
        self._batch_point_coords = None
        self._batch_visibility = None

    def _prepare_output(self):
        # Queue stores frame indices for the current batch
        t = self.output_queue.pop(0)

        # Process frame
        frame_t = self._batch_frames[t]
        frame_t = (((frame_t + 1.0) / 2.0) * 255.0).astype(cp.uint8)

        if frame_t.ndim == 4:
            frame_t = frame_t[0]

        # Process visibility
        pts_t = self._batch_point_coords[t]
        vis_t = self._batch_visibility[t]

        pts_visible = pts_t[vis_t]

        if pts_visible.shape[0] == 0:
            # Guard against empty selection
            pts_visible = cp.zeros((1, 2), dtype=cp.float32)

        return hs.as_tensor(frame_t), hs.as_tensor(pts_visible)

    def _emit_message(self, op_output, context):
        output = self._prepare_output()
        out_message = Entity(context)
        out_message.add(output[0], "frame")
        out_message.add(output[1], "point_coords")
        op_output.emit(out_message, "output")

    def compute(self, op_input, op_output, context):
        """Unbatch the batched predictions and frames, emit one per tick with visibility filtering."""
        # If we have pending outputs, emit one and return
        if self.output_queue:
            self._emit_message(op_output, context)
            return

        inference_result = op_input.receive("inference_result")
        _ = op_input.receive("track_ids")
        if inference_result is None:
            return

        # Inputs
        tracks = cp.asarray(inference_result["tracks"])  # (T, 1, N, 2)
        visibility = cp.asarray(inference_result["visible_tracks"])  # (T, 1, N, 1), bool
        frames = cp.asarray(inference_result["frames"])  # (T, H, W, 3) in [-1, 1]

        idx_slice = -self.overlap_size - 1 if self.overlap_size > 0 and self.idx > 0 else 0
        tracks = tracks[idx_slice:]
        visibility = visibility[idx_slice:]
        frames = frames[idx_slice:]
        self.idx += 1

        # Normalize coordinates by spatial dims (prefer from frames when available)
        if frames.ndim >= 3:
            H = frames.shape[1]
            W = frames.shape[2]
        else:
            H = 256
            W = 256

        point_coords = tracks[:, 0].astype(cp.float32)  # (T, N, 2)
        point_coords[..., 0] /= float(W)
        point_coords[..., 1] /= float(H)

        # Ensure visibility is boolean of shape (T, N)
        if visibility.dtype != cp.bool_:
            visibility = visibility > 0.5
        if visibility.ndim == 4:
            visibility = visibility[:, 0, :, 0]

        T, N = visibility.shape

        # Store batch buffers for per-frame processing
        self._batch_frames = frames
        self._batch_point_coords = point_coords
        self._batch_visibility = visibility

        # Enqueue frame indices
        self.output_queue.extend(range(T))

        if self.output_queue:
            self._emit_message(op_output, context)


class Visualize3DPostprocessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in_results").condition(ConditionType.NONE)
        spec.output("output")
        spec.param("overlap_size")
        spec.param("window_size")

    def initialize(self):
        self.output_queue = []
        self.idx = 0
        # Buffers for the current batch
        self._batch_frames = None
        self._batch_point_coords = None
        self._batch_visibility = None
        self._batch_camera_position = None
        self._batch_points3D = None
        self.global_rotation = cp.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        self._batch_camera_rotation = None

    def _prepare_output(self):
        # Queue stores frame indices for the current batch
        t = self.output_queue.pop(0)

        # Process frame
        frame_t = self._batch_frames[t]
        frame_t = (((frame_t + 1.0) / 2.0) * 255.0).astype(cp.uint8)

        if frame_t.ndim == 4:
            frame_t = frame_t[0]

        # Process visibility
        pts_t = self._batch_point_coords[t]
        vis_t = self._batch_visibility[t]

        pts_visible = pts_t[vis_t, :2]

        if pts_visible.shape[0] == 0:
            # Guard against empty selection
            pts_visible = cp.zeros((1, 2), dtype=cp.float32)

        camera_position_window = self._batch_camera_position_window[t : t + 1]
        camera_rotation = self._batch_camera_rotation[t : t + 1]
        points3D = self._batch_points3D[t]  # 3, N
        points3D = cp.transpose(points3D, (1, 0))  # N, 3

        return (
            frame_t,
            pts_visible,
            self._batch_camera_position,
            camera_position_window,
            points3D,
            camera_rotation,
        )

    def _emit_message(self, op_output, context):
        output = self._prepare_output()
        out_message = Entity(context)
        out_message.add(hs.as_tensor(output[0]), "frame")
        out_message.add(hs.as_tensor(cp.ascontiguousarray(output[1])), "point_coords")
        out_message.add(hs.as_tensor(cp.ascontiguousarray(output[2])), "camera_position")
        out_message.add(hs.as_tensor(cp.ascontiguousarray(output[3])), "camera_position_window")
        out_message.add(hs.as_tensor(cp.ascontiguousarray(output[4])), "points3D")
        out_message.add(hs.as_tensor(cp.ascontiguousarray(output[5])), "camera_rotation")
        op_output.emit(out_message, "output")

    def compute(self, op_input, op_output, context):
        """Unbatch the batched predictions and frames, emit one per tick with visibility filtering."""
        # If we have pending outputs, emit one and return
        if self.output_queue:
            self._emit_message(op_output, context)
            return

        inference_result = op_input.receive("in_results")

        if inference_result is None:
            return

        # Inputs
        tracks = cp.asarray(inference_result["tracks_with_depth"])  # [B, N, 3, T_window]
        visibility = cp.asarray(inference_result["visible_tracks"])  # [B, N, 1, T_window]
        frames = cp.asarray(inference_result["frames"])  # (T, H, W, 3) in [-1, 1]
        camera_position = cp.asarray(
            inference_result["camera_position"]
        )  # (T, 3)  T being the whole trajectory! Not the window!
        camera_position_window = cp.asarray(
            inference_result["camera_position_window"]
        )  # (T, 3)  This T is only the window.
        points3D = cp.asarray(inference_result["points3D"])  # (B, T, 3, N)
        camera_rotation = cp.asarray(inference_result["camera_rotation"])  # (T, 3, 3)

        if self.idx > 0 and self.overlap_size > 0:
            slice_start = -self.overlap_size
            tracks = tracks[..., slice_start:]
            visibility = visibility[..., slice_start:]
            frames = frames[slice_start:]
            camera_position_window = camera_position_window[slice_start:]
            points3D = points3D[:, slice_start:]
            camera_rotation = camera_rotation[slice_start:]
        self.idx += 1

        # Normalize coordinates by spatial dims (prefer from frames when available)
        H = frames.shape[1] if frames.ndim >= 3 else 256
        W = frames.shape[2] if frames.ndim >= 3 else 256

        point_coords = tracks[0, :, :]  # (N, 3, T)
        point_coords = point_coords.astype(cp.float32)
        # Normalize in place or copy
        point_coords[:, 0, :] = point_coords[:, 0, :] / float(W)
        point_coords[:, 1, :] = point_coords[:, 1, :] / float(H)
        point_coords[:, 2, :] = point_coords[:, 2, :] / 100.0

        # Transpose to (T, N, 3)
        point_coords = cp.transpose(point_coords, (2, 0, 1))

        # Ensure visibility is boolean of shape (T, N)
        if visibility.dtype != cp.bool_:
            visibility = visibility > 0.5
        if visibility.ndim == 4:
            # (B, N, 1, T) -> (1, N, 1, T)
            visibility = visibility[0, :, 0, :]  # (N, T)

        visibility = cp.transpose(visibility, (1, 0))  # (T, N)

        T, N = visibility.shape

        # Store batch buffers for per-frame processing
        self._batch_frames = frames
        self._batch_point_coords = point_coords
        self._batch_visibility = visibility
        self._batch_camera_position = camera_position[: -self.window_size]
        self._batch_camera_position_window = camera_position_window
        self._batch_camera_rotation = camera_rotation
        self._batch_points3D = points3D[0]

        # Enqueue frame indices
        self.output_queue.extend(range(T))

        if self.output_queue:
            self._emit_message(op_output, context)
