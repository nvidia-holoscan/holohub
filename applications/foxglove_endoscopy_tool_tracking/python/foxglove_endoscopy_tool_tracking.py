# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.operators import FormatConverterOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType

from holohub.foxglove import FoxglovePublisherOp, FoxgloveTensorAdapterOp
from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp


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
                block_size=107 * 60 * 7 * 4 * 4,
                num_blocks=4,
            ),
        )
        mask_adapter = FoxgloveTensorAdapterOp(
            self,
            name="mask_to_foxglove",
            **self.kwargs("mask_adapter"),
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
        self.add_flow(mask_adapter, foxglove, {("messages", "messages")})


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
