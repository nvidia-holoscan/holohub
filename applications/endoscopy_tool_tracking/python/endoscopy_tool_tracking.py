# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)

from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp
from holohub.qcap_source import QCAPSourceOp


class EndoscopyApp(Application):
    def __init__(self, data, record_type=None, source="replayer"):
        """Initialize the endoscopy tool tracking application

        Parameters
        ----------
        record_type : {None, "input", "visualizer"}, optional
            Set to "input" if you want to record the input video stream, or
            "visualizer" if you want to record the visualizer output.
        source : {"replayer", "aja", "qcap"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            or YUAN capture card is used.
        """
        super().__init__()

        # set name
        self.name = "Endoscopy App"

        # Optional parameters affecting the graph created by compose.
        self.record_type = record_type
        if record_type is not None:
            if record_type not in ("input", "visualizer"):
                raise ValueError("record_type must be either ('input' or 'visualizer')")
        self.source = source

        if data == "none":
            data = os.environ.get("HOLOSCAN_DATA_PATH", "../data")

        self.sample_data_path = data

    def compose(self):
        is_aja = self.source.lower() == "aja"
        is_qcap = self.source.lower() == "qcap"
        record_type = self.record_type
        is_aja_overlay_enabled = False

        if is_aja:
            aja_kwargs = self.kwargs("aja")
            source = AJASourceOp(self, name="aja", **aja_kwargs)

            # 4 bytes/channel, 4 channels
            width = aja_kwargs["width"]
            height = aja_kwargs["height"]
            is_rdma = aja_kwargs["rdma"]
            is_aja_overlay_enabled = is_aja and aja_kwargs["enable_overlay"]
            source_block_size = width * height * 4 * 4
            source_num_blocks = 3 if is_rdma else 4
        elif is_qcap:
            qcap_kwargs = self.kwargs("qcap")
            source = QCAPSourceOp(self, name="qcap", **qcap_kwargs)

            # 4 bytes/channel, 4 channels
            width = qcap_kwargs["width"]
            height = qcap_kwargs["height"]
            is_rdma = qcap_kwargs["rdma"]
            source_block_size = width * height * 4 * 4
            source_num_blocks = 3 if is_rdma else 4
        else:
            width = 854
            height = 480
            video_dir = self.sample_data_path
            if not os.path.exists(video_dir):
                raise ValueError(f"Could not find video data: {video_dir=}")
            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=video_dir,
                **self.kwargs("replayer"),
            )
            # 4 bytes/channel, 3 channels
            source_block_size = width * height * 3 * 4
            source_num_blocks = 2

        source_pool_kwargs = dict(
            storage_type=MemoryStorageType.DEVICE,
            block_size=source_block_size,
            num_blocks=source_num_blocks,
        )
        if record_type is not None:
            if ((record_type == "input") and is_aja) or (record_type == "visualizer"):
                recorder_format_converter = FormatConverterOp(
                    self,
                    name="recorder_format_converter",
                    pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
                    **self.kwargs("recorder_format_converter"),
                )
            recorder = VideoStreamRecorderOp(
                name="recorder", fragment=self, **self.kwargs("recorder")
            )

        if is_aja:
            config_key_name = "format_converter_aja"
        elif is_qcap:
            config_key_name = "format_converter_qcap"
        else:
            config_key_name = "format_converter_replayer"

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs(config_key_name),
        )

        lstm_inferer_block_size = 107 * 60 * 7 * 4
        lstm_inferer_num_blocks = 2 + 5 * 2
        model_file_path = os.path.join(self.sample_data_path, "tool_loc_convlstm.onnx")
        engine_cache_dir = os.path.join(self.sample_data_path, "engines")
        lstm_inferer = LSTMTensorRTInferenceOp(
            self,
            name="lstm_inferer",
            pool=BlockMemoryPool(
                self,
                name="device_allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=lstm_inferer_block_size,
                num_blocks=lstm_inferer_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            model_file_path=model_file_path,
            engine_cache_dir=engine_cache_dir,
            **self.kwargs("lstm_inference"),
        )

        tool_tracking_postprocessor_block_size = 107 * 60 * 7 * 4
        tool_tracking_postprocessor_num_blocks = 2
        tool_tracking_postprocessor = ToolTrackingPostprocessorOp(
            self,
            name="tool_tracking_postprocessor",
            device_allocator=BlockMemoryPool(
                self,
                name="device_allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=tool_tracking_postprocessor_block_size,
                num_blocks=tool_tracking_postprocessor_num_blocks,
            ),
            host_allocator=UnboundedAllocator(self, name="host_allocator"),
        )

        if (record_type == "visualizer") and not is_aja:
            visualizer_allocator = BlockMemoryPool(self, name="allocator", **source_pool_kwargs)
        else:
            visualizer_allocator = None

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            enable_render_buffer_input=is_aja_overlay_enabled,
            enable_render_buffer_output=is_aja_overlay_enabled or record_type == "visualizer",
            allocator=visualizer_allocator,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("holoviz_overlay" if is_aja_overlay_enabled else "holoviz"),
        )

        # Flow definition
        self.add_flow(lstm_inferer, tool_tracking_postprocessor, {("tensor", "in")})
        self.add_flow(tool_tracking_postprocessor, visualizer, {("out", "receivers")})
        self.add_flow(
            source,
            format_converter,
            {("video_buffer_output" if is_aja or is_qcap else "output", "source_video")},
        )
        self.add_flow(format_converter, lstm_inferer)
        if is_aja_overlay_enabled:
            # Overlay buffer flow between AJA source and visualizer
            self.add_flow(source, visualizer, {("overlay_buffer_output", "render_buffer_input")})
            self.add_flow(visualizer, source, {("render_buffer_output", "overlay_buffer_input")})
        else:
            self.add_flow(
                source, visualizer, {("video_buffer_output" if is_aja or is_qcap else "output", "receivers")}
            )
        if record_type == "input":
            if is_aja:
                self.add_flow(
                    source,
                    recorder_format_converter,
                    {("video_buffer_output", "source_video")},
                )
                self.add_flow(recorder_format_converter, recorder)
            else:
                self.add_flow(source, recorder)
        elif record_type == "visualizer":
            self.add_flow(
                visualizer,
                recorder_format_converter,
                {("render_buffer_output", "source_video")},
            )
            self.add_flow(recorder_format_converter, recorder)


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Endoscopy tool tracking demo application.")
    parser.add_argument(
        "-r",
        "--record_type",
        choices=["none", "input", "visualizer"],
        default="none",
        help="The video stream to record (default: %(default)s).",
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja", "qcap"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. If 'aja' use an AJA "
            "capture card as the source. If 'qcap' use an YUAN capture card "
            " as the source (default: %(default)s)."
        ),
    )
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
    record_type = args.record_type
    if record_type == "none":
        record_type = None

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "endoscopy_tool_tracking.yaml")
    else:
        config_file = args.config

    app = EndoscopyApp(record_type=record_type, source=args.source, data=args.data)
    app.config(config_file)
    app.run()
