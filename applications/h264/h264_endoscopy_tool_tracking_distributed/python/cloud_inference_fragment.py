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
    from holoscan.conditions import AsynchronousCondition
except ImportError as e:
    raise ImportError(
        "This example requires Holoscan SDK >= 2.1.0 so AsynchronousCondition is available."
    ) from e


from gxf_imports import VideoDecoderContext, VideoDecoderRequestOp, VideoDecoderResponseOp
from holoscan.core import Fragment
from holoscan.operators import FormatConverterOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator

from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp


class CloudInferenceFragment(Fragment):
    def __init__(
        self,
        app,
        name,
        model_dir,
    ):
        super().__init__(app, name)
        self.model_dir = model_dir

    def compose(self):
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

        rgb_float_format_converter = FormatConverterOp(
            self,
            name="rgb_float_format_converter",
            pool=UnboundedAllocator(self, name="pool"),
            **self.kwargs("rgb_float_format_converter"),
        )

        model_file_path = self.model_dir + "/tool_loc_convlstm.onnx"
        engine_cache_dir = self.model_dir + "/engines"

        lstm_inferer = LSTMTensorRTInferenceOp(
            self,
            name="lstm_inferer",
            model_file_path=model_file_path,
            engine_cache_dir=engine_cache_dir,
            pool=UnboundedAllocator(self, name="pool"),
            cuda_stream_pool=CudaStreamPool(self, 0, 0, 0, 1, 5, name="cuda_stream"),
            **self.kwargs("lstm_inference"),
        )

        tool_tracking_postprocessor = ToolTrackingPostprocessorOp(
            self,
            name="tool_tracking_postprocessor",
            device_allocator=UnboundedAllocator(self, name="device_allocator"),
            host_allocator=UnboundedAllocator(self, name="host_allocator"),
            **self.kwargs("tool_tracking_postprocessor"),
        )

        self.add_operator(video_decoder_request)
        self.add_flow(
            video_decoder_response,
            decoder_output_format_converter,
            {("output_transmitter", "source_video")},
        )
        self.add_flow(
            decoder_output_format_converter,
            rgb_float_format_converter,
            {("tensor", "source_video")},
        )
        self.add_flow(rgb_float_format_converter, lstm_inferer)
        self.add_flow(lstm_inferer, tool_tracking_postprocessor, {("tensor", "in")})
