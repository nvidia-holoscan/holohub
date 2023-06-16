/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights reserved.
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

#pragma once

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace QCAPSourceOp {

// Constructor
PYDOC(QCAPSourceOp, R"doc(
Operator to get a video stream from an YUAN High-Tech capture card.
)doc")

// PyQCAPSourceOp Constructor
PYDOC(QCAPSourceOp_python, R"doc(
Operator to get a video stream from an YUAN High-Tech capture card.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
width : int, optional
    Width of the video stream.
height : int, optional
    Height of the video stream.
framerate : int, optional
    Frame rate of the video stream.
rdma : bool, optional
    Boolean indicating whether RDMA is enabled.
pixel_format : str, optional
    The pixel format of the video stream.
input_type : str, optional
    The input type of the video stream.
mst_mode : int, optional
    The mst mode of the video stream.
sdi2g_mode : int, optional
    The SDI 12G mode of the video stream.
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

}  // namespace QCAPSourceOp
}  // namespace holoscan::doc
