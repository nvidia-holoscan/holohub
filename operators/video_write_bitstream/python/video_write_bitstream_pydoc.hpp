/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#ifndef PYHOLOHUB_OPERATORS_VIDEO_WRITE_BITSTREAM_OP_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_VIDEO_WRITE_BITSTREAM_OP_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace VideoWriteBitstreamOp {

PYDOC(VideoWriteBitstreamOp, R"doc(
Operator class to write h.264 video file into bitstream.
)doc")

// PyVideoWriteBitstreamOp Constructor
PYDOC(VideoWriteBitstreamOp_python, R"doc(
Operator class to write h.264 video file into bitstream.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
output_video_path: str
    The path to save the video.
frame_width: int
    The width of the output video
frame_height: int
    The height of the output video
inbuf_storage_type: int
    Input buffer storage type.
data_receiver: holoscan.IOSpec
    Data receiver to get data.
input_crc_file_path: str
    File for CRC verification
name : str, optional
    The name of the operator.
)doc")

}  // namespace VideoWriteBitstreamOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_VIDEO_DECODER_RESPONSE_PYDOC_HPP
