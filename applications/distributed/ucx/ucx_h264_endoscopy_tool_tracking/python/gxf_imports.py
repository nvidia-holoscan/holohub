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

from holoscan.operators import GXFCodeletOp
from holoscan.resources import GXFComponentResource

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
