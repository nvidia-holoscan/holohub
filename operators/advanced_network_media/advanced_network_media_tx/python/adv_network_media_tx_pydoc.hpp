/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOHUB_OPERATORS_ADV_NET_MEDIA_TX_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_ADV_NET_MEDIA_TX_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace AdvNetworkMediaTxOp {

PYDOC(AdvNetworkMediaTxOp, R"doc(
Advanced Networking Media Transmitter operator.

This operator processes video frames from GXF entities (either VideoBuffer or Tensor)
and transmits them over Rivermax-enabled network infrastructure.
)doc")

// PyAdvNetworkMediaTxOp Constructor
PYDOC(AdvNetworkMediaTxOp_python, R"doc(
Advanced Networking Media Transmitter operator.

This operator processes video frames from GXF entities (either VideoBuffer or Tensor)
and transmits them over Rivermax-enabled network infrastructure.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
interface_name : str, optional
    Name of the network interface to use for transmission.
queue_id : int, optional
    Queue ID for the network interface (default: 0).
video_format : str, optional
    Video format for transmission (default: "RGB888").
bit_depth : int, optional
    Bit depth of the video data (default: 8).
frame_width : int, optional
    Width of the video frame in pixels (default: 1920).
frame_height : int, optional
    Height of the video frame in pixels (default: 1080).
name : str, optional
    The name of the operator (default: "advanced_network_media_tx").
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
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

}  // namespace AdvNetworkMediaTxOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_ADV_NET_MEDIA_TX_PYDOC_HPP
