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
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity


class TracksAssemblerOp(Operator):
    """Assembler with generic inputs.

    This operator simply concatenates multiple tracking branches for a single
    window into one output tensor. It does **not** try to deduplicate or fuse
    tracks; each input branch occupies a contiguous block of points.

    Inputs (all for the same temporal window of length T):
      - input0_in: message with keys
          - "tracks": [T, 1, N, 2]
          - "visible_tracks": [T, 1, N, 1]
          - "frames": [T, H, W, 3]   (optional, only one branch needs to provide it)
      - input1_in: same as input0_in, but "frames" is optional
      - input2_in: same as input0_in, but "frames" is optional

    Outputs:
      - out:
          - "tracks": [T, 1, 3*N, 2]
          - "visible_tracks": [T, 1, 3*N, 1]
          - "frames": [T, H, W, 3]  (forwarded from the first input that has it)
      - track_ids_out:
          - "track_ids": [3*N] global ids, monotonically increasing over time
    """

    def __init__(self, *args, grid_size=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.called_once = False
        self.n_points_per_op = grid_size * grid_size
        self._initialize_ids()
        self.total_points_allocated = 3 * self.n_points_per_op

    def _initialize_ids(self):
        self.last_ids_0 = list(range(0, self.n_points_per_op))
        self.last_ids_1 = list(range(self.n_points_per_op, 2 * self.n_points_per_op))
        self.last_ids_2 = list(range(2 * self.n_points_per_op, 3 * self.n_points_per_op))

    def setup(self, spec: OperatorSpec):
        spec.input("input0_in")
        spec.input("input1_in")
        spec.input("backward_in")  # This is actually the backward tracks
        spec.output("out")
        spec.output("track_ids_out")

        spec.param("window_size")
        spec.param("overlap_size")

    def compute(self, op_input, op_output, context):
        in0 = op_input.receive("input0_in")
        in1 = op_input.receive("input1_in")
        backward_in = op_input.receive("backward_in")

        # Rotate id ranges across windows and allocate fresh ids for the newest block.
        if self.called_once:
            self.last_ids_0 = self.last_ids_1
            self.last_ids_1 = self.last_ids_2
            start_new = self.total_points_allocated
            end_new = start_new + self.n_points_per_op
            self.last_ids_2 = list(range(start_new, end_new))
            self.total_points_allocated = end_new
        else:
            self.called_once = True

        # Extract tensors from inputs.
        tracks0 = cp.asarray(in0.get("tracks"))
        vis0 = cp.asarray(in0.get("visible_tracks"))
        frames0 = in0.get("frames") if "frames" in in0.keys() else None

        tracks1 = cp.asarray(in1.get("tracks"))
        vis1 = cp.asarray(in1.get("visible_tracks"))
        frames1 = in1.get("frames") if "frames" in in1.keys() else None

        # Basic sanity: all inputs must have the same temporal length.
        T = max(tracks0.shape[0], tracks1.shape[0])
        if T != int(self.window_size):
            raise RuntimeError(
                f"TracksAssemblerOp: expected T={self.window_size}, got T={T} "
                f"(tracks0={tracks0.shape[0]}, tracks1={tracks1.shape[0]})"
            )

        # Identify which tracking is full and which one is half-window
        is_full_window0 = tracks0.shape[0] == self.window_size

        tracks_full = tracks0 if is_full_window0 else tracks1
        tracks_half = tracks1 if is_full_window0 else tracks0
        vis_full = vis0 if is_full_window0 else vis1
        vis_half = vis1 if is_full_window0 else vis0
        frames_full = frames0 if is_full_window0 else frames1

        backward_tracks = cp.asarray(backward_in.get("tracks"))
        backward_visibility = cp.asarray(backward_in.get("visible_tracks"))

        if T != backward_tracks.shape[0]:
            raise RuntimeError(
                f"TracksAssemblerOp: expected T={T}, got T={backward_tracks.shape[0]} "
                f"(backward_tracks={backward_tracks.shape[0]})"
            )

        # Number of points per branch (assumed equal across branches).
        _, B, N, _ = tracks0.shape
        if B != 1:
            raise RuntimeError(
                f"TracksAssemblerOp: expected B=1, got B={B} "
                f"(tracks0={tracks0.shape[0]}, tracks1={tracks1.shape[0]})"
            )

        total_points = 3 * N
        window_tracks = cp.zeros((T, 1, total_points, 2), dtype=cp.float32)
        window_visibility = cp.zeros((T, 1, total_points, 1), dtype=cp.float32)

        # Contiguous slices per branch.
        s_full = slice(0, N)
        s_half = slice(N, 2 * N)
        s_backward = slice(2 * N, 3 * N)

        window_tracks[:, :, s_full, :] = tracks_full
        window_visibility[:, :, s_full, :] = vis_full

        T_half = tracks_half.shape[0]

        if T_half == T:
            # Full-length half tracks - use directly
            window_tracks[:, :, s_half, :] = tracks_half
            window_visibility[:, :, s_half, :] = vis_half
        else:
            # tracks_half is shorter (covers latter part of window)
            start_idx = T - T_half
            window_tracks[:start_idx, :, s_half, :] = tracks_half[0:1, :, :, :]
            window_visibility[:start_idx, :, s_half, :] = 0
            window_tracks[start_idx:, :, s_half, :] = tracks_half
            window_visibility[start_idx:, :, s_half, :] = vis_half

        window_tracks[:, :, s_backward, :] = backward_tracks
        window_visibility[:, :, s_backward, :] = backward_visibility

        frames = None
        if frames_full is not None:
            frames = cp.asarray(frames_full)

        out_message = Entity(context)
        out_message.add(hs.as_tensor(window_tracks), "tracks")
        out_message.add(hs.as_tensor(window_visibility), "visible_tracks")
        if frames is not None:
            out_message.add(hs.as_tensor(frames), "frames")
        op_output.emit(out_message, "out")

        # Track ids for the concatenated blocks, monotonically increasing over windows.
        current_track_ids = self.last_ids_0 + self.last_ids_1 + self.last_ids_2
        current_track_ids = cp.asarray(current_track_ids, dtype=cp.int32)

        out_ids = Entity(context)
        out_ids.add(hs.as_tensor(current_track_ids), "track_ids")
        op_output.emit(out_ids, "track_ids_out")
