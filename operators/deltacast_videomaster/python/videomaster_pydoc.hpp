/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 DELTACAST.TV. All rights
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

#ifndef PYHOLOSCAN_OPERATORS_VIDEOMASTER_SOURCE_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_VIDEOMASTER_SOURCE_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace VideoMasterSourceOp {

// PyVideoMasterSourceOp Constructor
PYDOC(VideoMasterSourceOp, R"doc(
 Operator to get a video stream from a Deltacast capture card.)doc")

PYDOC(VideoMasterSourceOp_python, R"doc(
     Operator to get a video stream from a Deltacast capture card.

     Parameters
     ----------
    fragment : Fragment
        The fragment that the operator belongs to.
    rdma : bool, optional
        Boolean indicating whether RDMA is enabled.
    board : int, optional
        The board to target (e.g., "0" for board 0). Default value is ``0``.
    input : int, optional
        The RX channel of the baords (e.g., "1" for input 1). Default value is ``0``.
    width : int, optional
        Width of the video stream. Default value is ``1920``.
    height : int, optional
        Height of the video stream. Default value is ``1080``.
    progressive : bool, optional
        Whether or not the video is an interlaced format. Default value is ``True``.
    framerate : int, optional
        Frame rate of the video stream. Default value is ``60``.
    pool : Allocator of type UnboundedAllocator
        The pool to use for memory allocation.
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
}  // namespace VideoMasterSourceOp

namespace VideoMasterTransmitterOp {

// PyVideoMasterTransmitterOp Constructor
PYDOC(VideoMasterTransmitterOp, R"doc(
 Operator to stream video from a Deltacast capture card.
 )doc")

PYDOC(VideoMasterTransmitterOp_python, R"doc(
 Operator to stream video from a Deltacast capture card.
 
 Parameters
     ----------
    fragment : Fragment
        The fragment that the operator belongs to.
    rdma : bool, optional
        Boolean indicating whether RDMA is enabled.
    board : int, optional
        The board to target (e.g., "0" for board 0). Default value is ``0``.
    output : int, optional
        The TX channel of the baords (e.g., "1" for input 1). Default value is ``0``.
    width : int, optional
        Width of the video stream. Default value is ``1920``.
    height : int, optional
        Height of the video stream. Default value is ``1080``.
    progressive : bool, optional
        Whether or not the video is an interlaced format. Default value is ``True``.
    framerate : int, optional
        Frame rate of the video stream. Default value is ``60``.
    pool : Allocator of type UnboundedAllocator
        The pool to use for memory allocation.
    enable_overlay : bool, optional
        Boolean indicating whether a overlay processing is done by the board or not. Default value is ``False``.
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

}  // namespace VideoMasterTransmitterOp

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_OPERATORS_AJA_SOURCE_PYDOC_HPP */
