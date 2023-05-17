# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from holoscan.logger import load_env_log_level
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    SegmentationPostprocessorOp,
    TensorRTInferenceOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)


class EndoToolSegApp(Application):
    def __init__(self, source="replayer"):
        """Initialize the endoscopy tool segmentation application

        Parameters
        ----------
        source : {"replayer", "aja"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            capture card is used.
        """

        super().__init__()

        # set name
        self.name = "Endoscopy Tool Segmentation App"

        # Optional parameters affecting the graph created by compose.
        self.source = source

    def compose(self):
        n_channels = 4  # RGBA
        bpp = 4  # bytes per pixel

        is_aja = self.source.lower() == "aja"
        if is_aja:
            source = AJASourceOp(self, name="aja", **self.kwargs("aja"))
            drop_alpha_block_size = 1920 * 1080 * n_channels * bpp
            drop_alpha_num_blocks = 2
            drop_alpha_channel = FormatConverterOp(
                self,
                name="drop_alpha_channel",
                pool=BlockMemoryPool(
                    self,
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=drop_alpha_block_size,
                    num_blocks=drop_alpha_num_blocks,
                ),
                **self.kwargs("drop_alpha_channel"),
            )
        else:
            source = VideoStreamReplayerOp(self, name="replayer", **self.kwargs("replayer"))

        width_preprocessor = 1264
        height_preprocessor = 1080
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 2
        segmentation_preprocessor = FormatConverterOp(
            self,
            name="segmentation_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            **self.kwargs("segmentation_preprocessor"),
        )

        tensorrt_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        segmentation_inference = TensorRTInferenceOp(
            self,
            name="segmentation_inference",
            pool=UnboundedAllocator(self, name="pool"),
            cuda_stream_pool=tensorrt_cuda_stream_pool,
            **self.kwargs("segmentation_inference"),
        )

        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("segmentation_postprocessor"),
        )

        segmentation_visualizer = HolovizOp(
            self,
            name="segmentation_visualizer",
            **self.kwargs("segmentation_visualizer"),
        )

        if is_aja:
            self.add_flow(source, segmentation_visualizer, {("video_buffer_output", "receivers")})
            self.add_flow(source, drop_alpha_channel, {("video_buffer_output", "")})
            self.add_flow(drop_alpha_channel, segmentation_preprocessor)
        else:
            self.add_flow(source, segmentation_visualizer, {("", "receivers")})
            self.add_flow(source, segmentation_preprocessor)
        self.add_flow(segmentation_preprocessor, segmentation_inference)
        self.add_flow(segmentation_inference, segmentation_postprocessor)
        self.add_flow(
            segmentation_postprocessor,
            segmentation_visualizer,
            {("", "receivers")},
        )


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Endoscopy tool segmentation demo application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. If 'aja' use an AJA "
            "capture card as the source (default: %(default)s)."
        ),
    )
    args = parser.parse_args()

    load_env_log_level()

    config_file = os.path.join(os.path.dirname(__file__), "tool_segmentation.yaml")

    app = EndoToolSegApp(source=args.source)
    app.config(config_file)
    app.run()
