/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, TECNALIA. All rights reserved.
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

#ifndef PYHOLOHUB_OPERATORS_GSTREAMER_GST_VIDEO_RECORDER_OP_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_GSTREAMER_GST_VIDEO_RECORDER_OP_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace GstVideoRecorderOp {

// PyGstVideoRecorderOp Constructor
PYDOC(GstVideoRecorderOp, R"doc(

Operator for recording video streams to file using GStreamer.

**==Named Inputs==**

input : TensorMap
Video frames to encode and write to file.
Width, height, and storage type are automatically detected from the first frame.

Parameters
----------

fragment : holoscan.core.Fragment
    Fragment that owns the operator.
encoder : str
    Encoder base name (e.g. ``"nvh264"``, ``"nvh265"``, ``"x264"``, or ``"x265"``).
    The "enc" suffix is automatically appended to form the element name.
    Default value is ``"nvh264"``.
format : str
    Pixel format for video data (e.g. ``"RGBA"``, ``"RGB"``, ``"BGRA"``, ``"BGR"``, and ``"GRAY8"``).
    Default value is ``"RGBA"``.
framerate : str
    Video framerate as a fraction or decimal, for example ``"30/1"``,
    ``"30000/1001"``, ``"29.97"``, or ``"60"``.
    Special value ``"0/1"`` enables live mode with no framerate control.
    Default value is ``"30/1"``
max_buffers : int
    Maximum number of buffers to queue (0 = unlimited).
    Default value is ``10``.
block : bool
    Whether ``push_buffer()`` blocks when the internal queue is full (``True`` = block, ``False`` = non-blocking, may drop/timeout).
    Default value is ``True``.
filename : str
    Output video filename.
    If no extension is provided, ``".mp4"`` is automatically appended.
    Default value is ``"output.mp4"``.
properties : dict of str to str
    Map of encoder-specific properties.
    Examples include ``{"bitrate": "8000", "preset": "1", "gop-size": "30"}``.
    Default value is an empty dictionary.
name : str
    Operator name.
    Default value is ``"gst_video_recorder"``.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

}  // namespace GstVideoRecorderOp
}  // namespace holoscan::doc
#endif  // PYHOLOHUB_OPERATORS_GSTREAMER_GST_VIDEO_RECORDER_OP_PYDOC_HPP
