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

#ifndef PYHOLOHUB_OPERATORS_OPENIGTLINK_RX_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_OPENIGTLINK_RX_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace OpenIGTLinkRxOp {

PYDOC(OpenIGTLinkRxOp, R"doc(
Operator class to send data using the OpenIGTLink protocol.
)doc")

// PyOpenIGTLinkRxOp Constructor
PYDOC(OpenIGTLinkRxOp_python, R"doc(
Operator class to send data using the OpenIGTLink protocol.

Named outputs:
    out_tensor: nvidia::gxf::Tensor
        Emits a message containing a tensor named "out_tensor" that contains
        the OpenIGTLink Image message converted to nvidia::gxf::Tensor.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
port : integer, optional
    Port number of server.
out_tensor_name : str, optional
    Name of output tensor.
flip_width_height : bool, optional
    Flip width and height (necessary for receiving from 3D Slicer).

name : str, optional
    The name of the operator.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

}  // namespace OpenIGTLinkRxOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_OPENIGTLINK_RX_PYDOC_HPP
