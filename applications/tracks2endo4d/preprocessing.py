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

import logging

import cupy as cp
import holoscan as hs
from holoscan.core import ConditionType, Operator, OperatorSpec
from holoscan.gxf import Entity


class OverlapWindowCoordinatorOp(Operator):
    """Coordinator that:

    - Buffers frames into non-overlapping windows of length `window_size`.
    - Emits per-frame messages for two forward branches, where a new forward
      run is started every `overlap_size` frames.

    Inputs:
      - in: message with key "source_video" (preprocessed frame)

    Outputs:
      - fwd0_frame: per-frame messages for the first forward branch
      - fwd1_frame: per-frame messages for the second forward branch
      - batch_out: full window [T, ...] for the backward branch
    """

    def initialize(self):
        self._batch_pool = []
        self._pool_idx = 0
        self._write_idx = 0
        self._step_tensors = []

        self._frame_idx = 0
        self._slots = [
            {"active": False, "start": 0, "step": 0},
            {"active": False, "start": 0, "step": 0},
        ]

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("fwd0_frame")
        spec.output("fwd1_frame")
        spec.output("batch_out")
        spec.param("window_size")
        spec.param("overlap_size")

    def _ensure_buffers(self, frame_shape):
        if self._batch_pool:
            return

        T = int(self.window_size)
        # Allocate batch pool (3 buffers to cycle through safely)
        # Shape: (T, 1, H, W, C)
        batch_shape = (T, 1) + frame_shape
        self._batch_pool = [cp.zeros(batch_shape, dtype=cp.float32) for _ in range(3)]

        # Allocate step scalars (0 to T)
        self._step_tensors = [cp.array([i], dtype=cp.int32) for i in range(T + 1)]

    def _maybe_start_run(self):
        """Start a new forward run every overlap_size frames, reusing slots."""
        if self._frame_idx % int(self.overlap_size) != 0:
            return

        # Pick an available slot (inactive or finished run).
        for slot in self._slots:
            if not slot["active"]:
                slot["active"] = True
                slot["start"] = self._frame_idx
                slot["step"] = 0
                return

        # If both are active, we skip starting a new run to keep at most 2 in parallel.

    def _emit_forward_frame(self, context, op_output, port_name, frame_view, slot):
        out_message = Entity(context)
        # Emit as 'video' [1, H, W, C]
        out_message.add(hs.as_tensor(frame_view[None]), "video")

        step = slot["step"]
        # Fallback if step exceeds pre-allocated range (shouldn't happen with correct logic)
        if step < len(self._step_tensors):
            s_tensor = self._step_tensors[step]
        else:
            s_tensor = cp.array([step], dtype=cp.int32)
        out_message.add(hs.as_tensor(s_tensor), "step")

        op_output.emit(out_message, port_name)

    def _emit_batch_if_full(self, context, op_output):
        T = int(self.window_size)
        stride = int(self.overlap_size)

        if self._write_idx == T:
            # Emit current batch
            current_batch = self._batch_pool[self._pool_idx]

            out_message = Entity(context)
            # Use "frames" (plural) or "frame"?
            # We updated BatchSplitterVideoOp to accept "frames" or "frame".
            # Original used "frames". Let's use "frame" to be standard.
            out_message.add(hs.as_tensor(current_batch), "frame")

            op_output.emit(out_message, "batch_out")

            # Prepare next batch by copying overlap region
            next_pool_idx = (self._pool_idx + 1) % 3
            next_batch = self._batch_pool[next_pool_idx]

            # Keep the last (T - stride) frames
            keep_count = T - stride
            if keep_count > 0:
                # Copy from end of current to start of next
                cp.copyto(next_batch[:keep_count], current_batch[stride:])

            self._pool_idx = next_pool_idx
            self._write_idx = keep_count

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        frame = msg.get("source_video")

        # Lazy initialization of buffers
        if not self._batch_pool:
            shape = frame.shape
            self._ensure_buffers(shape)

        # Get current batch buffer
        batch = self._batch_pool[self._pool_idx]

        # Write frame to buffer
        batch[self._write_idx, 0] = frame

        # Get view for forward emission (1, H, W, C)
        frame_view = batch[self._write_idx]

        # Maybe start a new forward run on this frame
        self._maybe_start_run()

        # Emit per-frame messages for all active forward slots
        for idx, slot in enumerate(self._slots):
            if not slot["active"]:
                continue
            port_name = "fwd0_frame" if idx == 0 else "fwd1_frame"
            self._emit_forward_frame(context, op_output, port_name, frame_view, slot)
            slot["step"] += 1
            if slot["step"] >= int(self.window_size):
                slot["active"] = False

        self._frame_idx += 1
        self._write_idx += 1

        # If a full window is buffered, emit for backward branch
        self._emit_batch_if_full(context, op_output)


class BatchSplitterVideoOp(Operator):
    """Splits a batch tensor into individual frames"""

    def __init__(self, fragment, *args, max_frames=0, **kwargs):
        self.max_frames = max_frames
        super().__init__(fragment, *args, **kwargs)

    def initialize(self):
        self.output_queue = []
        self.emitted_frames = 0
        self.logger: logging.Logger = logging.getLogger(__name__)

    def setup(self, spec: OperatorSpec):
        spec.input("in").condition(ConditionType.NONE)  # batched input tensor
        spec.output("out")
        spec.param("grid_query_frame")

    def _emit_message(self, context, op_output):
        if len(self.output_queue) == 0:
            return
        out_message = Entity(context)
        m = self.output_queue.pop(0)
        # Emit as 'video' to match TapNextInferenceOp expectation
        out_message.add(hs.as_tensor(m[0][None]), "video")
        out_message.add(hs.as_tensor(m[1]), "step")
        op_output.emit(out_message, "out")
        self.emitted_frames += 1

    def compute(self, op_input, op_output, context):
        message = op_input.receive("in")
        if self.max_frames > 0 and self.emitted_frames == self.max_frames:
            self.logger.warning(
                f"No more frames. Stopping app... {self.emitted_frames=} {self.max_frames=}"
            )
            self.fragment.stop_execution()
            return

        if message is None:
            self._emit_message(context, op_output)
            return

        if "frame" in message:
            batch = cp.asarray(message["frame"])
        elif "frames" in message:
            batch = cp.asarray(message["frames"])
        else:
            raise RuntimeError(
                "BatchSplitterVideoOp: Input message missing 'frame' or 'frames' tensor."
            )

        for i in range(self.grid_query_frame, batch.shape[0]):
            frame = cp.ascontiguousarray(batch[i])
            self.output_queue.append([frame, cp.array(i, dtype=cp.int32)])

        if self.output_queue:
            # emit each frame as a new message
            self._emit_message(context, op_output)
