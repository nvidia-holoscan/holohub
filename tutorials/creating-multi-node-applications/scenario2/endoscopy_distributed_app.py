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

from holoscan.core import Application, Fragment
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    InferenceProcessorOp,
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

# Enable this line for Yuam capture card
# from holohub.qcap_source import QCAPSourceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp


class Fragment1(Fragment):
    def __init__(self, app, name, sample_data_path, source, record_type):
        super().__init__(app, name)

        self.source = source
        self.sample_data_path = sample_data_path
        self.record_type = record_type

    def compose(self):
        rdma = False
        record_type = self.record_type
        is_overlay_enabled = False

        if self.source.lower() == "aja":
            aja_kwargs = self.kwargs("aja")
            source = AJASourceOp(self, name="aja", **aja_kwargs)

            # 4 bytes/channel, 4 channels
            width = aja_kwargs["width"]
            height = aja_kwargs["height"]
            rdma = aja_kwargs["rdma"]
            is_overlay_enabled = aja_kwargs["enable_overlay"]
            source_block_size = width * height * 4 * 4
            source_num_blocks = 3 if rdma else 4
        elif self.source.lower() == "yuan":
            yuan_kwargs = self.kwargs("yuan")
            # Uncomment to enable QCap
            # source = QCAPSourceOp(self, name="yuan", **yuan_kwargs)

            # 4 bytes/channel, 4 channels
            width = yuan_kwargs["width"]
            height = yuan_kwargs["height"]
            rdma = yuan_kwargs["rdma"]
            source_block_size = width * height * 4 * 4
            source_num_blocks = 3 if rdma else 4
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
            if ((record_type == "input") and (self.source != "replayer")) or (
                record_type == "visualizer"
            ):
                recorder_format_converter = FormatConverterOp(
                    self,
                    name="recorder_format_converter",
                    pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
                    **self.kwargs("recorder_format_converter"),
                )
            recorder = VideoStreamRecorderOp(
                name="recorder", fragment=self, **self.kwargs("recorder")
            )

        config_key_name = "format_converter_" + self.source.lower()

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

        if (record_type == "visualizer") and (self.source == "replayer"):
            visualizer_allocator = BlockMemoryPool(self, name="allocator", **source_pool_kwargs)
        else:
            visualizer_allocator = None

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            enable_render_buffer_input=is_overlay_enabled,
            enable_render_buffer_output=is_overlay_enabled or record_type == "visualizer",
            allocator=visualizer_allocator,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("holoviz_overlay" if is_overlay_enabled else "holoviz"),
        )

        # Flow definition
        self.add_flow(lstm_inferer, tool_tracking_postprocessor, {("tensor", "in")})
        self.add_flow(
            tool_tracking_postprocessor,
            visualizer,
            {("out_coords", "receivers"), ("out_mask", "receivers")},
        )
        self.add_flow(
            source,
            format_converter,
            {("video_buffer_output" if self.source != "replayer" else "output", "source_video")},
        )
        self.add_flow(format_converter, lstm_inferer)
        if is_overlay_enabled:
            # Overlay buffer flow between AJA source and visualizer
            self.add_flow(source, visualizer, {("overlay_buffer_output", "render_buffer_input")})
            self.add_flow(visualizer, source, {("render_buffer_output", "overlay_buffer_input")})
        else:
            self.add_flow(
                source,
                visualizer,
                {("video_buffer_output" if self.source != "replayer" else "output", "receivers")},
            )
        if record_type == "input":
            if self.source != "replayer":
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


class Fragment2(Fragment):
    def __init__(self, app, name, source, model_path, record_type):
        super().__init__(app, name)

        self.source = source
        self.record_type = record_type
        self.model_path = model_path

    def compose(self):

        is_aja = self.source.lower() == "aja"

        pool = UnboundedAllocator(self, name="fragment2_pool")
        in_dtype = "rgba8888" if is_aja else "rgb888"

        out_of_body_preprocessor = FormatConverterOp(
            self,
            name="out_of_body_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("out_of_body_preprocessor"),
        )

        model_path_map = {
            "out_of_body": os.path.join(self.model_path, "out_of_body_detection.onnx")
        }
        for k, v in model_path_map.items():
            if not os.path.exists(v):
                raise RuntimeError(f"Could not find model file: {v}")
        inference_kwargs = self.kwargs("out_of_body_inference")
        inference_kwargs["model_path_map"] = model_path_map
        out_of_body_inference = InferenceOp(
            self,
            name="out_of_body_inference",
            allocator=pool,
            **inference_kwargs,
        )
        out_of_body_postprocessor = InferenceProcessorOp(
            self,
            name="out_of_body_postprocessor",
            allocator=pool,
            disable_transmitter=True,
            **self.kwargs("out_of_body_postprocessor"),
        )

        self.add_flow(out_of_body_preprocessor, out_of_body_inference, {("", "receivers")})
        self.add_flow(
            out_of_body_inference, out_of_body_postprocessor, {("transmitter", "receivers")}
        )


class EndoscopyDistributedApp(Application):
    def __init__(self, data, record_type=None, source="replayer"):
        """Initialize the endoscopy distributed application containing the
           endoscopy_tool_tracking and endoscopy_out_of_body apps

        Parameters
        ----------
        record_type : {None, "input", "visualizer"}, optional
            Set to "input" if you want to record the input video stream, or
            "visualizer" if you want to record the visualizer output.
        source : {"replayer", "aja"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            capture card is used.
        """
        super().__init__()

        # set name
        self.name = "Endoscopy Distributed App"

        # Optional parameters affecting the graph created by compose.
        self.record_type = record_type
        if record_type is not None:
            if record_type not in ("input", "visualizer"):
                raise ValueError("record_type must be either ('input' or 'visualizer')")
        self.source = source

        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

    def compose(self):
        is_aja = self.source.lower() == "aja"

        fragment1 = Fragment1(
            self,
            name="fragment1",
            source=self.source,
            sample_data_path=os.path.join(self.sample_data_path, "endoscopy"),
            record_type=self.record_type,
        )
        fragment2 = Fragment2(
            self,
            name="fragment2",
            source=self.source,
            model_path=os.path.join(self.sample_data_path, "endoscopy_out_of_body_detection"),
            record_type=self.record_type,
        )

        self.add_flow(
            fragment1,
            fragment2,
            {
                (
                    "aja.video_buffer_output" if is_aja else "replayer.output",
                    "out_of_body_preprocessor",
                )
            },
        )


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
        choices=["replayer", "aja", "yuan"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. Otherwise use a "
            "capture card as the source (default: %(default)s)."
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
    apps_argv = Application().argv
    args = parser.parse_args(apps_argv[1:])
    record_type = args.record_type
    if record_type == "none":
        record_type = None

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "endoscopy_distributed_app.yaml")
    else:
        config_file = args.config

    app = EndoscopyDistributedApp(record_type=record_type, source=args.source, data=args.data)
    app.config(config_file)
    app.run()
