/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GXF_IMPORTS_CC
#define GXF_IMPORTS_CC

#include <holoscan/core/resources/gxf/gxf_component_resource.hpp>
#include <holoscan/operators/gxf_codelet/gxf_codelet.hpp>

// Import h.264 GXF codelets and components as Holoscan operators and resources
// Starting with Holoscan SDK v2.1.0, importing GXF codelets/components as Holoscan operators/
// resources can be done using the HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR and
// HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE macros. This new feature allows using GXF codelets
// and components in Holoscan applications without writing custom class wrappers (for C++) and
// Python wrappers (for Python) for each GXF codelet and component.
// For the VideoEncoderRequestOp class, since it needs to override the setup() to provide custom
// parameters and override the initialize() to register custom converters, it requires a custom
// class that extends the holoscan::ops::GXFCodeletOp class.

// The VideoDecoderResponseOp implements nvidia::gxf::VideoDecoderResponse and handles the output
// of the decoded H264 bit stream.
// Parameters:
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data.
// - outbuf_storage_type (uint32_t): Output Buffer Storage(memory) type used by this allocator.
//   Can be 0: kHost, 1: kDevice.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder context
//   Handle.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderResponseOp, "nvidia::gxf::VideoDecoderResponse")

// The VideoDecoderRequestOp implements nvidia::gxf::VideoDecoderRequest and handles the input
// for the H264 bit stream decode.
// Parameters:
// - inbuf_storage_type (uint32_t): Input Buffer storage type, 0:kHost, 1:kDevice.
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder
//   context Handle.
// - codec (uint32_t): Video codec to use, 0:H264, only H264 supported. Default:0.
// - disableDPB (uint32_t): Enable low latency decode, works only for IPPP case.
// - output_format (std::string): VidOutput frame video format, nv12pl and yuv420planar are
//   supported.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderRequestOp, "nvidia::gxf::VideoDecoderRequest")

// The VideoDecoderContext implements nvidia::gxf::VideoDecoderContext and holds common variables
// and underlying context.
// Parameters:
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition required to get/set event state.
HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(VideoDecoderContext, "nvidia::gxf::VideoDecoderContext")

// The VideoReadBitstreamOp implements nvidia::gxf::VideoReadBitStream and reads h.264 video files
// from the disk at the specified input file path.
// Parameters:
// - input_file_path (std::string): Path to image file
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data
// - outbuf_storage_type (int32_t): Output Buffer storage type, 0:kHost, 1:kDevice
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoReadBitstreamOp, "nvidia::gxf::VideoReadBitStream")

// The VideoWriteBitstreamOp implements nvidia::gxf::VideoWriteBitstream and writes bit stream to
// the disk at specified output path.
// Parameters:
// - output_video_path (std::string): The file path of the output video
// - frame_width (int): The width of the output video
// - frame_height (int): The height of the output video
// - inbuf_storage_type (int): Input Buffer storage type, 0:kHost, 1:kDevice
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoWriteBitstreamOp, "nvidia::gxf::VideoWriteBitstream")

// The VideoEncoderResponseOp implements nvidia::gxf::VideoEncoderResponse and handles the output
// of the encoded YUV frames.
// Parameters:
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data.
// - videoencoder_context (std::shared_ptr<holoscan::ops::VideoEncoderContext>): Encoder context
//   handle.
// - outbuf_storage_type (uint32_t): Output Buffer Storage(memory) type used by this allocator.
//   Can be 0: kHost, 1: kDevice. Default: 1.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoEncoderResponseOp, "nvidia::gxf::VideoEncoderResponse")

// The VideoEncoderContext implements nvidia::gxf::VideoEncoderContext and holds common variables
// and underlying context.
// Parameters:
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition required to get/set event state.
HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(VideoEncoderContext, "nvidia::gxf::VideoEncoderContext")

#endif