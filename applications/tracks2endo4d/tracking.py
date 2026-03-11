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

import copy
import os

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity


def get_model_path(args, data_path, name):
    args_return = copy.deepcopy(args)
    args_return["model_path_map"] = {name: os.path.join(data_path, args["model_file"])}
    del args_return["model_file"]
    return args_return


class BatchMergerOp(Operator):
    """Accumulates incoming messages until batch_size is reached, then emits merged batch."""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._buffer_frames = []
        self._buffer_tracks = []
        self._buffer_visibility = []
        self._buffer_window_num = []

    def setup(self, spec: OperatorSpec):
        spec.input("predictions_in")
        spec.input("frame_in")
        spec.output("out")
        spec.param("window_size")
        spec.param("suffix")

    def compute(self, op_input, op_output, context):
        """Collect individual frames and emit when batch is full."""
        frame_message = op_input.receive("frame_in")
        preds = op_input.receive("predictions_in")
        frame = cp.asarray(frame_message.get("video"))  # "frame" / "video"
        if frame.ndim == 5:
            frame = frame[0]
        tracks_name = [key for key in preds.keys() if "tracks" in key][0]
        # visibility_name = [key for key in preds.keys() if "visible_tracks" in key][0]
        visibility_name = "visible_" + tracks_name

        # We copy the arrays to free GPU memory
        tracks = cp.array(preds.get(tracks_name), copy=True)
        visibility = cp.array(preds.get(visibility_name), copy=True)

        self._buffer_frames.append(frame)
        self._buffer_tracks.append(tracks)
        self._buffer_visibility.append(visibility)

        assert len(self._buffer_frames) == len(self._buffer_tracks) == len(self._buffer_visibility)

        # print("Buffer frames length", len(self._buffer_frames), "calling from", self.name)

        # Once we have a full batch, merge and emit
        if len(self._buffer_frames) == self.window_size:
            # Merge into one batch tensor: shape [N, ...]
            frames_merged = cp.concatenate(self._buffer_frames, axis=0)
            tracks_merged = cp.concatenate(self._buffer_tracks, axis=0)
            visibility_merged = cp.concatenate(self._buffer_visibility, axis=0)

            # Reset buffer for next batch
            self._buffer_frames.clear()
            self._buffer_tracks.clear()
            self._buffer_visibility.clear()

            out_message = Entity(context)
            out_message.add(hs.as_tensor(frames_merged), "frames")
            out_message.add(hs.as_tensor(tracks_merged), "tracks")
            out_message.add(hs.as_tensor(visibility_merged), "visible_tracks")
            op_output.emit(out_message, "out")


class BatchMergerSchedulingOp(Operator):
    """Accumulates incoming messages until a condition is reached, then emits merged batch.

    Buffer clearing is done by the schedule_clear condition.

    Args:
        schedule_clear (lambda function): Condition to clear the buffer and emit the batch.
    """

    def __init__(self, fragment, schedule_emission, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.schedule_emission = schedule_emission

    def initialize(self):
        self._buffer_frames = []
        self._buffer_tracks = []
        self._buffer_visibility = []
        self.num_frames = 0

    def setup(self, spec: OperatorSpec):
        spec.input("predictions_in")
        spec.input("frame_in")
        spec.output("out")
        spec.param("window_size")
        spec.param("overlap_size")
        spec.param("suffix")

    def compute(self, op_input, op_output, context):
        """Collect individual frames and emit when batch is full."""
        frame_message = op_input.receive("frame_in")
        preds = op_input.receive("predictions_in")
        if "frame" in frame_message.keys():
            frame = cp.asarray(frame_message.get("frame"))
        else:
            frame = cp.asarray(frame_message.get("video"))
        if frame.ndim == 5:
            frame = frame[0]
        tracks_name = [key for key in preds.keys() if "tracks" in key][0]
        # visibility_name = [key for key in preds.keys() if "visible_tracks" in key][0]
        visibility_name = "visible_" + tracks_name

        # We copy the arrays to free GPU memory
        tracks = cp.array(preds.get(tracks_name), copy=True)
        visibility = cp.array(preds.get(visibility_name), copy=True)

        self._buffer_frames.append(frame)
        self._buffer_tracks.append(tracks)
        self._buffer_visibility.append(visibility)

        assert len(self._buffer_frames) == len(self._buffer_tracks) == len(self._buffer_visibility)
        self.num_frames += 1

        # print("Buffer frames length", len(self._buffer_frames), "calling from", self.name)

        # Once we have a full batch, merge and emit
        if self.schedule_emission(self.num_frames):
            # Merge into one batch tensor: shape [N, ...]
            frames_merged = cp.concatenate(self._buffer_frames, axis=0)
            tracks_merged = cp.concatenate(self._buffer_tracks, axis=0)
            visibility_merged = cp.concatenate(self._buffer_visibility, axis=0)

            out_message = Entity(context)
            out_message.add(hs.as_tensor(frames_merged), "frames")
            out_message.add(hs.as_tensor(tracks_merged), "tracks")
            out_message.add(hs.as_tensor(visibility_merged), "visible_tracks")
            op_output.emit(out_message, "out")

        if len(self._buffer_frames) == self.window_size:
            # Reset buffer for next batch
            self._buffer_frames.clear()
            self._buffer_tracks.clear()
            self._buffer_visibility.clear()


class ReverseBatch(Operator):
    """Reverses the order of a batch of frames only if do_backwards is True. Otherwise, it just passes the batch through."""

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        spec.param("do_backwards")
        spec.param("axis")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        out_message = Entity(context)

        if self.do_backwards:
            for key in in_message.keys():
                cp_tensor = cp.asarray(in_message.get(key))
                cp_tensor = cp.flip(cp_tensor, axis=self.axis)
                cp_tensor = cp.ascontiguousarray(cp_tensor)
                out_message.add(hs.as_tensor(cp_tensor), key)
        else:
            out_message = in_message
        op_output.emit(out_message, "out")
