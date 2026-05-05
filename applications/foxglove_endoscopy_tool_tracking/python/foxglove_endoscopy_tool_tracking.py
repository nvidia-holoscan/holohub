# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import time
from argparse import ArgumentParser

import cupy as cp
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType

from holohub.foxglove import (
    FoxgloveBatch,
    FoxgloveBox2D,
    FoxgloveImageAnnotations,
    FoxglovePoint2D,
    FoxglovePointsAnnotation,
    FoxglovePublisherOp,
    FoxgloveTensorAdapterOp,
    FoxgloveText,
    PointsAnnotationType,
)
from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp

ENDOSCOPY_WIDTH = 854
ENDOSCOPY_HEIGHT = 480
TOOL_MASK_WIDTH = 107
TOOL_MASK_HEIGHT = 60


def image_coordinates(x, y):
    if not math.isfinite(x) or not math.isfinite(y) or x < 0.0 or y < 0.0:
        return None
    if x <= 1.0 and y <= 1.0:
        return x * ENDOSCOPY_WIDTH, y * ENDOSCOPY_HEIGHT
    if x <= TOOL_MASK_WIDTH and y <= TOOL_MASK_HEIGHT:
        return x * ENDOSCOPY_WIDTH / TOOL_MASK_WIDTH, y * ENDOSCOPY_HEIGHT / TOOL_MASK_HEIGHT
    if x <= ENDOSCOPY_WIDTH and y <= ENDOSCOPY_HEIGHT:
        return x, y
    return None


class ToolTrackingFoxgloveAdapterOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("messages")
        spec.param("annotation_topic", "/detections")
        spec.param(
            "labels",
            ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "Spec.Bag"],
        )

    def compute(self, op_input, op_output, context):
        entity = op_input.receive("input")
        tensor = entity.get("scaled_coords")
        if tensor is None:
            raise RuntimeError("Tool tracking output is missing tensor 'scaled_coords'")

        coords = cp.asarray(tensor).get()
        if coords.size % 3 != 0:
            raise RuntimeError("Tool tracking tensor 'scaled_coords' must be Nx3")
        coords = coords.reshape((-1, 3))

        annotations = FoxgloveImageAnnotations()
        annotations.topic = self.annotation_topic
        annotations.timestamp_ns = time.time_ns()

        boxes = []
        point_sets = []
        texts = []
        for index, (raw_x, raw_y, raw_marker_size) in enumerate(coords):
            maybe_point = image_coordinates(float(raw_x), float(raw_y))
            if maybe_point is None:
                continue

            marker_size = float(raw_marker_size)
            if not math.isfinite(marker_size) or marker_size <= 0.0:
                continue

            x, y = maybe_point
            label = self.labels[index] if index < len(self.labels) else f"tool_{index}"
            box_extent = (
                marker_size * min(ENDOSCOPY_WIDTH, ENDOSCOPY_HEIGHT)
                if marker_size <= 1.0
                else marker_size
            )

            box = FoxgloveBox2D()
            box.x = max(0.0, min(x - box_extent * 0.5, ENDOSCOPY_WIDTH - 1.0))
            box.y = max(0.0, min(y - box_extent * 0.5, ENDOSCOPY_HEIGHT - 1.0))
            box.width = min(box_extent, ENDOSCOPY_WIDTH - box.x)
            box.height = min(box_extent, ENDOSCOPY_HEIGHT - box.y)
            box.label = label
            boxes.append(box)

            point = FoxglovePoint2D()
            point.x = x
            point.y = y
            point.label = label

            point_set = FoxglovePointsAnnotation()
            point_set.type = PointsAnnotationType.POINTS
            point_set.label = label
            point_set.thickness = max(4.0, box_extent * 0.4)
            point_set.points = [point]
            point_sets.append(point_set)

            text = FoxgloveText()
            text.x = x + 6.0
            text.y = y - 6.0
            text.text = label
            text.font_size = 14.0
            texts.append(text)

        annotations.boxes = boxes
        annotations.point_sets = point_sets
        annotations.texts = texts

        batch = FoxgloveBatch()
        batch.annotations = [annotations]
        op_output.emit(batch, "messages")


class FoxgloveEndoscopyToolTrackingApp(Application):
    def __init__(self, data_path):
        super().__init__()
        self.name = "Foxglove endoscopy tool tracking"
        self.data_path = data_path
        self.enable_metadata(True)

    def compose(self):
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        width = 854
        height = 480
        source_block_size = width * height * 3 * 4
        replayer = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=self.data_path,
            **self.kwargs("replayer"),
        )
        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=source_block_size,
                num_blocks=2,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("format_converter"),
        )

        lstm_inferer = LSTMTensorRTInferenceOp(
            self,
            name="lstm_inferer",
            pool=BlockMemoryPool(
                self,
                name="lstm_pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=107 * 60 * 7 * 4,
                num_blocks=2 + 5 * 2,
            ),
            cuda_stream_pool=cuda_stream_pool,
            model_file_path=os.path.join(self.data_path, "tool_loc_convlstm.onnx"),
            engine_cache_dir=os.path.join(self.data_path, "engines"),
            **self.kwargs("lstm_inference"),
        )
        postprocessor = ToolTrackingPostprocessorOp(
            self,
            name="tool_tracking_postprocessor",
            device_allocator=BlockMemoryPool(
                self,
                name="postprocessor_pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=107 * 60 * 7 * 4,
                num_blocks=4,
            ),
        )
        mask_adapter = FoxgloveTensorAdapterOp(
            self,
            name="mask_to_foxglove",
            **self.kwargs("mask_adapter"),
        )
        tracking_adapter = ToolTrackingFoxgloveAdapterOp(
            self,
            name="tool_tracking_foxglove",
            **self.kwargs("tool_tracking_foxglove"),
        )
        foxglove = FoxglovePublisherOp(
            self,
            name="foxglove",
            **self.kwargs("foxglove"),
        )

        self.add_flow(replayer, foxglove, {("output", "image")})
        self.add_flow(replayer, format_converter, {("output", "source_video")})
        self.add_flow(format_converter, lstm_inferer)
        self.add_flow(lstm_inferer, postprocessor, {("tensor", "in")})
        self.add_flow(postprocessor, mask_adapter, {("out", "input")})
        self.add_flow(postprocessor, tracking_adapter, {("out", "input")})
        self.add_flow(mask_adapter, foxglove, {("messages", "messages")})
        self.add_flow(tracking_adapter, foxglove, {("messages", "messages")})


def main():
    parser = ArgumentParser(description="Foxglove endoscopy tool tracking")
    parser.add_argument(
        "-d",
        "--data",
        default=os.environ.get("HOLOSCAN_INPUT_PATH", f"{os.getcwd()}/data/endoscopy"),
        help="Path to the HoloHub endoscopy sample data.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "foxglove_endoscopy_tool_tracking.yaml"),
        help="Path to the application YAML config.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        raise ValueError(f"Data path '{args.data}' does not exist.")

    app = FoxgloveEndoscopyToolTrackingApp(args.data)
    app.config(args.config)
    app.run()


if __name__ == "__main__":
    main()
