# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    FormatConverterOp,
    HolovizOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
)

# Uncomment UnboundedAllocator to enable DELTACAST capture card (linter issue)
from holoscan.resources import (  # UnboundedAllocator,
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
)

from holohub.aja_source import AJASourceOp
from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp

# Enable this line for Yuam capture card
# from holohub.qcap_source import QCAPSourceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp

# Uncomment to enable DELTACAST capture card
# from holohub.videomaster import VideoMasterSourceOp, VideoMasterTransmitterOp


# Enable this line for vtk rendering
# from holohub.vtk_renderer import VtkRendererOp


class EndoscopyApp(Application):
    def __init__(self, data, record_type=None, source="replayer"):
        """Initialize the endoscopy tool tracking application

        Parameters
        ----------
        record_type : {None, "input", "visualizer"}, optional
            Set to "input" if you want to record the input video stream, or
            "visualizer" if you want to record the visualizer output.
        source : {"replayer", "aja", "deltacast", "yuan"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA or Yuan
            capture card is used.
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
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

    def compose(self):
        rdma = False
        record_type = self.record_type
        is_overlay_enabled = False
        renderer = self.kwargs("visualizer")["visualizer"]
        input_video_signal = "receivers" if renderer == "holoviz" else "videostream"
        input_annotations_signal = "receivers" if renderer == "holoviz" else "annotations"
        source_name = self.source.lower()

        if source_name == "aja":
            aja_kwargs = self.kwargs("aja")
            source = AJASourceOp(self, name="aja", **aja_kwargs)

            # 4 bytes/channel, 4 channels
            width = aja_kwargs["width"]
            height = aja_kwargs["height"]
            rdma = aja_kwargs["rdma"]
            is_overlay_enabled = aja_kwargs["enable_overlay"]
            source_block_size = width * height * 4 * 4
            source_num_blocks = 3 if rdma else 4
        elif source_name == "deltacast":
            deltacast_kwargs = self.kwargs("deltacast")

            width = deltacast_kwargs["width"]
            height = deltacast_kwargs["height"]
            rdma = deltacast_kwargs["rdma"]
            is_overlay_enabled = deltacast_kwargs["enable_overlay"]

            source_block_size = width * height * 4 * 4
            source_num_blocks = 3 if rdma else 4
            # Uncomment to enable DELTACAST capture card (linter issue)
            # source = VideoMasterSourceOp(
            #     self,
            #     name="deltacast",
            #     pool=UnboundedAllocator(self, name="pool"),
            #     rdma=rdma,
            #     board=deltacast_kwargs["board"],
            #     input=deltacast_kwargs["input"],
            #     width=width,
            #     height=height,
            #     progressive=deltacast_kwargs.get("progressive", True),
            #     framerate=deltacast_kwargs.get("framerate", 60),
            # )
        elif source_name == "yuan":
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
            # the RMMAllocator supported since v2.6 is much faster than the default UnboundAllocator
            try:
                from holoscan.resources import RMMAllocator

                source.add_arg(allocator=RMMAllocator(self, name="video_replayer_allocator"))
            except Exception:
                pass
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

        config_key_name = "format_converter_" + source_name

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

        # the tool tracking post process outputs
        # - a RGBA float32 color mask
        # - coordinates with x,y and size in float32
        bytes_per_float32 = 4
        tool_tracking_postprocessor_block_size = max(
            107 * 60 * 7 * 4 * bytes_per_float32, 7 * 3 * bytes_per_float32
        )
        tool_tracking_postprocessor_num_blocks = 2 * 2
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
        )

        visualizer_allocator = None
        should_use_allocator = record_type == "visualizer" and self.source == "replayer"
        if source_name == "deltacast":
            should_use_allocator = should_use_allocator or is_overlay_enabled

        if should_use_allocator:
            visualizer_allocator = BlockMemoryPool(self, name="allocator", **source_pool_kwargs)

        if renderer == "holoviz":
            visualizer = HolovizOp(
                self,
                name="holoviz",
                width=width,
                height=height,
                enable_render_buffer_input=(
                    is_overlay_enabled if source_name != "deltacast" else None
                ),
                enable_render_buffer_output=is_overlay_enabled or record_type == "visualizer",
                allocator=visualizer_allocator,
                cuda_stream_pool=cuda_stream_pool,
                **self.kwargs("holoviz_overlay" if is_overlay_enabled else "holoviz"),
            )
        # Uncomment the following lines to use VTK renderer
        # else:
        #     visualizer = VtkRendererOp(
        #         self,
        #         name="vtk",
        #         width=width,
        #         height=height,
        #         window_name="VTK (Kitware) Python",
        #         **self.kwargs("vtk_op"),
        #     )

        # Flow definition
        self.add_flow(lstm_inferer, tool_tracking_postprocessor, {("tensor", "in")})

        self.add_flow(
            tool_tracking_postprocessor,
            visualizer,
            {("out", input_annotations_signal)},
        )

        output_signal = "output" if self.source == "replayer" else "video_buffer_output"
        if source_name == "deltacast":
            output_signal = "signal"

        self.add_flow(
            source,
            format_converter,
            {(output_signal, "source_video")},
        )
        self.add_flow(format_converter, lstm_inferer)

        if source_name == "deltacast":
            if is_overlay_enabled:
                # Uncomment to enable DELTACAST capture card (linter issue)
                # overlayer = VideoMasterTransmitterOp(
                #     self,
                #     name="videomaster",
                #     pool=UnboundedAllocator(self, name="pool"),
                #     rdma=deltacast_kwargs.get("rdma", False),
                #     board=deltacast_kwargs.get("board", 0),
                #     width=width,
                #     height=height,
                #     output=deltacast_kwargs.get("output", 0),
                #     progressive=deltacast_kwargs.get("progressive", True),
                #     framerate=deltacast_kwargs.get("framerate", 60),
                #     enable_overlay=deltacast_kwargs.get("enable_overlay", False),
                # )
                overlay_format_converter = FormatConverterOp(
                    self,
                    name="overlay_format_converter",
                    pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
                    **self.kwargs("deltacast_overlay_format_converter"),
                )
                self.add_flow(visualizer, overlay_format_converter, {("render_buffer_output", "")})
                # Uncomment to enable DELTACAST capture card (linter issue)
                # self.add_flow(overlay_format_converter, overlayer)
            else:
                visualizer_format_converter_videomaster = FormatConverterOp(
                    self,
                    name="visualizer_format_converter",
                    pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
                    **self.kwargs("deltacast_visualizer_format_converter"),
                )
                drop_alpha_channel_converter = FormatConverterOp(
                    self,
                    name="drop_alpha_channel_converter",
                    pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
                    **self.kwargs("deltacast_drop_alpha_channel_converter"),
                )
                self.add_flow(source, drop_alpha_channel_converter)
                self.add_flow(drop_alpha_channel_converter, visualizer_format_converter_videomaster)
                self.add_flow(
                    visualizer_format_converter_videomaster, visualizer, {("", "receivers")}
                )
        else:
            if is_overlay_enabled:
                # Overlay buffer flow between AJA source and visualizer
                self.add_flow(
                    source, visualizer, {("overlay_buffer_output", "render_buffer_input")}
                )
                self.add_flow(
                    visualizer, source, {("render_buffer_output", "overlay_buffer_input")}
                )
            else:
                self.add_flow(
                    source,
                    visualizer,
                    {
                        (
                            "video_buffer_output" if self.source != "replayer" else "output",
                            input_video_signal,
                        )
                    },
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


if __name__ == "__main__":
    default_data_path = f"{os.getcwd()}/data/endoscopy"
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
        choices=[
            "replayer",
            "aja",
            "deltacast",
            "yuan",
        ],
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
        default=os.environ.get("HOLOSCAN_INPUT_PATH", default_data_path),
        help=("Set the data path (default: %(default)s)."),
    )
    args = parser.parse_args()
    record_type = args.record_type
    if record_type == "none":
        record_type = None

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "endoscopy_tool_tracking.yaml")
    else:
        config_file = args.config

    # handle case where HOLOSCAN_INPUT_PATH is set with no value
    if len(args.data) == 0:
        args.data = default_data_path

    if not os.path.isdir(args.data):
        raise ValueError(
            f"Data path '{args.data}' does not exist. Use --data or set HOLOSCAN_INPUT_PATH environment variable."
        )

    app = EndoscopyApp(record_type=record_type, source=args.source, data=args.data)
    app.config(config_file)
    app.run()
