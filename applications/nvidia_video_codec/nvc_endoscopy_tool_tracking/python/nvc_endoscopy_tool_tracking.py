# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import os
from argparse import ArgumentParser

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application
from holoscan.operators import FormatConverterOp, HolovizOp
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    RMMAllocator,
    UnboundedAllocator,
)

from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp
from holohub.nv_video_decoder import NvVideoDecoderOp
from holohub.nv_video_encoder import NvVideoEncoderOp
from holohub.nv_video_reader import NvVideoReaderOp
from holohub.tensor_to_file import TensorToFileOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp


class EndoscopyApp(Application):
    def __init__(self, data):
        """Initialize the endoscopy tool tracking application"""
        super().__init__()

        # set name
        self.name = "Endoscopy App"

        if (data is None) or (data == "none"):
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

    def compose(self):
        width = 854
        height = 480
        source_block_size = width * height * 3 * 4
        source_num_blocks = 2
        fps = self.kwargs("holoviz")["framerate"]

        h264_file_reader = NvVideoReaderOp(
            self,
            CountCondition(self, 683),
            PeriodicCondition(self, name="periodic_condition", recess_period=1 / fps),
            name="h264_file_reader",
            directory=self.sample_data_path,
            allocator=UnboundedAllocator(self, name="video_reader_pool"),
            **self.kwargs("reader"),
        )

        decoder = NvVideoDecoderOp(
            self,
            name="nv_decoder",
            allocator=UnboundedAllocator(self, name="video_decoder_pool"),
            **self.kwargs("decoder"),
        )

        decoder_output_format_converter = FormatConverterOp(
            self,
            name="decoder_output_format_converter",
            pool=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=source_block_size,
                num_blocks=source_num_blocks,
            ),
            **self.kwargs("decoder_output_format_converter"),
        )

        rgb_float_format_converter = FormatConverterOp(
            self,
            name="rgb_float_format_converter",
            pool=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=source_block_size,
                num_blocks=source_num_blocks,
            ),
            **self.kwargs("rgb_float_format_converter"),
        )

        model_file_path = self.sample_data_path + "/tool_loc_convlstm.onnx"
        engine_cache_dir = self.sample_data_path + "/engines"

        lstm_inferer = LSTMTensorRTInferenceOp(
            self,
            name="lstm_inferer",
            model_file_path=model_file_path,
            engine_cache_dir=engine_cache_dir,
            pool=UnboundedAllocator(self, name="pool"),
            cuda_stream_pool=CudaStreamPool(
                self,
                name="cuda_stream",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=5,
            ),
            **self.kwargs("lstm_inference"),
        )

        tool_tracking_postprocessor = ToolTrackingPostprocessorOp(
            self,
            name="tool_tracking_postprocessor",
            device_allocator=UnboundedAllocator(self, "device_allocator"),
            **self.kwargs("tool_tracking_postprocessor"),
        )

        record_output = bool(self.from_config("record_output"))

        visualizer = HolovizOp(
            self,
            name="visualizer",
            allocator=BlockMemoryPool(
                self,
                name="allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=source_block_size,
                num_blocks=source_num_blocks,
            ),
            enable_render_buffer_input=False,
            enable_render_buffer_output=record_output,
            **self.kwargs("holoviz"),
        )
        self.add_flow(h264_file_reader, decoder, {("output", "input")})

        self.add_flow(
            decoder,
            decoder_output_format_converter,
            {("output", "source_video")},
        )
        self.add_flow(decoder_output_format_converter, visualizer, {("tensor", "receivers")})
        self.add_flow(
            decoder_output_format_converter,
            rgb_float_format_converter,
            {("tensor", "source_video")},
        )
        self.add_flow(rgb_float_format_converter, lstm_inferer)
        self.add_flow(lstm_inferer, tool_tracking_postprocessor, {("tensor", "in")})
        self.add_flow(tool_tracking_postprocessor, visualizer, {("out", "receivers")})

        if record_output:
            holoviz_output_format_converter = FormatConverterOp(
                self,
                name="holoviz_output_format_converter",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                **self.kwargs("holoviz_output_format_converter"),
            )

            encoder = NvVideoEncoderOp(
                self,
                name="nv_encoder",
                width=width,
                height=height,
                allocator=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.HOST,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                **self.kwargs("encoder"),
            )

            writer = TensorToFileOp(
                self,
                name="nv_writer",
                allocator=RMMAllocator(self, name="video_writer_allocator"),
                **self.kwargs("writer"),
            )

            self.add_flow(
                visualizer,
                holoviz_output_format_converter,
                {("render_buffer_output", "source_video")},
            )
            self.add_flow(holoviz_output_format_converter, encoder, {("tensor", "input")})
            self.add_flow(encoder, writer, {("output", "input")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Endoscopy tool tracking demo application.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "nvc_endoscopy_tool_tracking.yaml")
    else:
        config_file = args.config

    app = EndoscopyApp(data=args.data)

    app.config(config_file)
    app.run()
