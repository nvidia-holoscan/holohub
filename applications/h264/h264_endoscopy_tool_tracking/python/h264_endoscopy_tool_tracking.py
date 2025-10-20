# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    from holoscan.conditions import AsynchronousCondition, CountCondition, PeriodicCondition
except ImportError as e:
    raise ImportError(
        "This example requires Holoscan SDK >= 2.1.0 so AsynchronousCondition is available."
    ) from e
from holoscan.core import Application
from holoscan.gxf import load_extensions
from holoscan.operators import FormatConverterOp, GXFCodeletOp, HolovizOp
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    GXFComponentResource,
    MemoryStorageType,
    UnboundedAllocator,
)

from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp
from holohub.tensor_to_video_buffer import TensorToVideoBufferOp

# Enable this line for Yuam capture card
# from holohub.qcap_source import QCAPSourceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp

# Import h.264 GXF codelets and components as Holoscan operators and resources
# Starting with Holoscan SDK v2.1.0, importing GXF codelets/components as Holoscan operators/
# resources can be done by extending the GXFCodeletOp class and the GXFComponentResource class.
# This new feature allows GXF codelets and components in Holoscan applications without writing
# custom class wrappers in C++ and Python wrappers for each GXF codelet and component.


# The VideoDecoderResponseOp implements nvidia::gxf::VideoDecoderResponse and handles the output
# of the decoded H264 bit stream.
# Parameters:
# - pool (Allocator): Memory pool for allocating output data.
# - outbuf_storage_type (int): Output Buffer Storage(memory) type used by this allocator.
#   Can be 0: kHost, 1: kDevice.
# - videodecoder_context (VideoDecoderContext): Decoder context
#   Handle.
class VideoDecoderResponseOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoDecoderResponse", *args, **kwargs)


# The VideoDecoderRequestOp implements nvidia::gxf::VideoDecoderRequest and handles the input
# for the H264 bit stream decode.
# Parameters:
# - inbuf_storage_type (int): Input Buffer storage type, 0:kHost, 1:kDevice.
# - async_scheduling_term (AsynchronousCondition): Asynchronous scheduling condition.
# - videodecoder_context (VideoDecoderContext): Decoder context Handle.
# - codec (int): Video codec to use, 0:H264, only H264 supported. Default:0.
# - disableDPB (int): Enable low latency decode, works only for IPPP case.
# - output_format (str): VidOutput frame video format, nv12pl and yuv420planar are supported.
class VideoDecoderRequestOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoDecoderRequest", *args, **kwargs)


# The VideoDecoderContext implements nvidia::gxf::VideoDecoderContext and holds common variables
# and underlying context.
# Parameters:
# - async_scheduling_term (AsynchronousCondition): Asynchronous scheduling condition required to get/set event state.
class VideoDecoderContext(GXFComponentResource):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoDecoderContext", *args, **kwargs)


# The VideoReadBitstreamOp implements nvidia::gxf::VideoReadBitStream and reads h.264 video files
# from the disk at the specified input file path.
# Parameters:
# - input_file_path (str): Path to image file
# - pool (Allocator): Memory pool for allocating output data
# - outbuf_storage_type (int): Output Buffer storage type, 0:kHost, 1:kDevice
class VideoReadBitstreamOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoReadBitStream", *args, **kwargs)


# The VideoWriteBitstreamOp implements nvidia::gxf::VideoWriteBitstream and writes bit stream to
# the disk at specified output path.
# Parameters:
# - output_video_path (str): The file path of the output video
# - frame_width (int): The width of the output video
# - frame_height (int): The height of the output video
# - inbuf_storage_type (int): Input Buffer storage type, 0:kHost, 1:kDevice
class VideoWriteBitstreamOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoWriteBitstream", *args, **kwargs)


# The VideoEncoderResponseOp implements nvidia::gxf::VideoEncoderResponse and handles the output
# of the encoded YUV frames.
# Parameters:
# - pool (Allocator): Memory pool for allocating output data.
# - videoencoder_context (VideoEncoderContext): Encoder context handle.
# - outbuf_storage_type (int): Output Buffer Storage(memory) type used by this allocator.
#   Can be 0: kHost, 1: kDevice. Default: 1.
class VideoEncoderResponseOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoEncoderResponse", *args, **kwargs)


# The VideoEncoderContext implements nvidia::gxf::VideoEncoderContext and holds common variables
# and underlying context.
# Parameters:
# - async_scheduling_term (AsynchronousCondition): Asynchronous scheduling condition required to get/set event state.
class VideoEncoderContext(GXFComponentResource):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoEncoderContext", *args, **kwargs)


# The VideoEncoderRequestOp implements nvidia::gxf::VideoEncoderRequest and handles the input for
# encoding YUV frames to H264 bit stream.
# Refer to operators/video_encoder/video_encoder_request/README.md for details
class VideoEncoderRequestOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoEncoderRequest", *args, **kwargs)


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

        bitstream_reader = VideoReadBitstreamOp(
            self,
            CountCondition(self, 750),
            PeriodicCondition(self, name="periodic-condition", recess_period=0.04),
            name="bitstream_reader",
            input_file_path=f"{self.sample_data_path}/surgical_video.264",
            pool=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.HOST,
                block_size=source_block_size,
                num_blocks=source_num_blocks,
            ),
            **self.kwargs("bitstream_reader"),
        )

        response_condition = AsynchronousCondition(self, "response_condition")
        video_decoder_context = VideoDecoderContext(self, async_scheduling_term=response_condition)

        request_condition = AsynchronousCondition(self, "request_condition")
        video_decoder_request = VideoDecoderRequestOp(
            self,
            name="video_decoder_request",
            async_scheduling_term=request_condition,
            videodecoder_context=video_decoder_context,
            **self.kwargs("video_decoder_request"),
        )

        video_decoder_response = VideoDecoderResponseOp(
            self,
            name="video_decoder_response",
            pool=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=source_block_size,
                num_blocks=source_num_blocks,
            ),
            videodecoder_context=video_decoder_context,
            **self.kwargs("video_decoder_response"),
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

        visualizer_allocator = BlockMemoryPool(
            self,
            name="allocator",
            storage_type=MemoryStorageType.DEVICE,
            block_size=source_block_size,
            num_blocks=source_num_blocks,
        )
        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            enable_render_buffer_input=False,
            enable_render_buffer_output=record_output == True,
            allocator=visualizer_allocator,
            **self.kwargs("holoviz"),
        )

        self.add_flow(
            bitstream_reader, video_decoder_request, {("output_transmitter", "input_frame")}
        )
        self.add_flow(
            video_decoder_response,
            decoder_output_format_converter,
            {("output_transmitter", "source_video")},
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
            encoder_async_condition = AsynchronousCondition(self, "encoder_async_condition")
            video_encoder_context = VideoEncoderContext(
                self, scheduling_term=encoder_async_condition
            )

            video_encoder_request = VideoEncoderRequestOp(
                self,
                name="video_encoder_request",
                videoencoder_context=video_encoder_context,
                **self.kwargs("video_encoder_request"),
            )
            video_encoder_response = VideoEncoderResponseOp(
                self,
                name="video_encoder_response",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                videoencoder_context=video_encoder_context,
                **self.kwargs("video_encoder_response"),
            )

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

            encoder_input_format_converter = FormatConverterOp(
                self,
                name="encoder_input_format_converter",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                **self.kwargs("encoder_input_format_converter"),
            )

            tensor_to_video_buffer = TensorToVideoBufferOp(
                self, name="tensor_to_video_buffer", **self.kwargs("tensor_to_video_buffer")
            )

            bitstream_writer = VideoWriteBitstreamOp(
                self,
                name="bitstream_writer",
                output_video_path="{self.sample_data_path}/surgical_video_output.264",
                input_crc_file_path="{self.sample_data_path}/surgical_video_output.txt",
                **self.kwargs("bitstream_writer"),
            )

            self.add_flow(
                visualizer,
                holoviz_output_format_converter,
                {("render_buffer_output", "source_video")},
            )
            self.add_flow(
                holoviz_output_format_converter,
                encoder_input_format_converter,
                {("tensor", "source_video")},
            )
            self.add_flow(
                encoder_input_format_converter, tensor_to_video_buffer, {("tensor", "in_tensor")}
            )
            self.add_flow(
                tensor_to_video_buffer, video_encoder_request, {("out_video_buffer", "input_frame")}
            )
            self.add_flow(
                video_encoder_response, bitstream_writer, {("output_transmitter", "data_receiver")}
            )


def main():
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
        config_file = os.path.join(os.path.dirname(__file__), "h264_endoscopy_tool_tracking.yaml")
    else:
        config_file = args.config

    app = EndoscopyApp(data=args.data)

    context = app.executor.context_uint64
    exts = [
        "libgxf_videodecoder.so",
        "libgxf_videodecoderio.so",
        "libgxf_videoencoder.so",
        "libgxf_videoencoderio.so",
    ]
    load_extensions(context, exts)

    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
