/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOHUB_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace V4L2VideoCaptureOp {

// Constructor
PYDOC(V4L2VideoCaptureOp, R"doc(
Operator to get a video stream from a V4L2 source.
)doc")

// V4L2VideoCaptureOp Constructor
PYDOC(V4L2VideoCaptureOp_python, R"doc(
Operator to get a video stream from a V4L2 source. Built-in HDMI capture card or USB cam.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
allocator : ``holoscan.resources.Allocator``
    Memory allocator to use for the output.
device : str
    The device to target (e.g. "/dev/video0" for device 0)
width : int, optional
    Width of the video stream.
height : int, optional
    Height of the video stream.
num_buffers : int, optional
    Number of V4L2 buffers to use.
pixel_format : str
    Video stream pixel format ("YUYV" or "RGBA32")
name : str, optional
    The name of the operator.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

}  // namespace V4L2VideoCaptureOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP
