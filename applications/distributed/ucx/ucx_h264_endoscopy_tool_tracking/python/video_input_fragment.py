# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    from holoscan.conditions import AsynchronousCondition, CountCondition, PeriodicCondition
except ImportError as e:
    raise ImportError(
        "This example requires Holoscan SDK >= 2.1.0 so AsynchronousCondition is available."
    ) from e

import os

from gxf_imports import (
    VideoDecoderContext,
    VideoDecoderRequestOp,
    VideoDecoderResponseOp,
    VideoReadBitstreamOp,
)
from holoscan.core import Fragment
from holoscan.operators import FormatConverterOp
from holoscan.resources import UnboundedAllocator


class VideoInputFragment(Fragment):
    def __init__(self, app, name, video_dir):
        super().__init__(app, name)
        self.video_dir = video_dir

        if not os.path.exists(self.video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

    def compose(self):
        bitstream_reader = VideoReadBitstreamOp(
            self,
            CountCondition(self, 750),
            PeriodicCondition(self, name="periodic-condition", recess_period=0.04),
            name="bitstream_reader",
            input_file_path=f"{self.video_dir}/surgical_video.264",
            pool=UnboundedAllocator(self, name="pool"),
            **self.kwargs("bitstream_reader"),
        )

        response_condition = AsynchronousCondition(self, name="response_condition")
        video_decoder_context = VideoDecoderContext(
            self, name="decoder-context", async_scheduling_term=response_condition
        )

        request_condition = AsynchronousCondition(self, name="request_condition")
        video_decoder_request = VideoDecoderRequestOp(
            self,
            request_condition,
            name="video_decoder_request",
            async_scheduling_term=request_condition,
            videodecoder_context=video_decoder_context,
            **self.kwargs("video_decoder_request"),
        )

        video_decoder_response = VideoDecoderResponseOp(
            self,
            response_condition,
            name="video_decoder_response",
            pool=UnboundedAllocator(self, name="pool"),
            videodecoder_context=video_decoder_context,
            **self.kwargs("video_decoder_response"),
        )

        decoder_output_format_converter = FormatConverterOp(
            self,
            name="decoder_output_format_converter",
            pool=UnboundedAllocator(self, name="pool"),
            **self.kwargs("decoder_output_format_converter"),
        )

        self.add_flow(
            bitstream_reader, video_decoder_request, {("output_transmitter", "input_frame")}
        )
        self.add_flow(
            video_decoder_response,
            decoder_output_format_converter,
            {("output_transmitter", "source_video")},
        )
