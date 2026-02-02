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

#ifndef PYHOLOHUB_OPERATORS_ADV_NET_MEDIA_RX_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_ADV_NET_MEDIA_RX_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace AdvNetworkMediaRxOp {

PYDOC(AdvNetworkMediaRxOp, R"doc(
Advanced Networking Media Receiver operator.

This operator receives video frames over Rivermax-enabled network infrastructure
and outputs them as GXF VideoBuffer entities.
)doc")

// PyAdvNetworkMediaRxOp Constructor
PYDOC(AdvNetworkMediaRxOp_python, R"doc(
Advanced Networking Media Receiver operator.

This operator receives video frames over Rivermax-enabled network infrastructure
and outputs them as GXF VideoBuffer entities.

Note: Advanced network initialization must be done in the application before creating
this operator using: adv_network_common.adv_net_init(config)

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
interface_name : str, optional
    Name of the network interface to use for reception.
queue_id : int, optional
    Queue ID for the network interface (default: 0).
frame_width : int, optional
    Width of the video frame in pixels (default: 1920).
frame_height : int, optional
    Height of the video frame in pixels (default: 1080).
bit_depth : int, optional
    Bit depth of the video data (default: 8).
video_format : str, optional
    Video format for reception (default: "RGB888").
hds : bool, optional
    Header Data Split setting (default: True).
output_format : str, optional
    Output format for the frames (default: "video_buffer").
memory_location : str, optional
    Memory location for frame storage (default: "device").
name : str, optional
    The name of the operator (default: "advanced_network_media_rx").
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

}  // namespace AdvNetworkMediaRxOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_ADV_NET_MEDIA_RX_PYDOC_HPP
