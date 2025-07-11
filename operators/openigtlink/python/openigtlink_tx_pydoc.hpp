/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOHUB_OPERATORS_OPENIGTLINK_TX_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_OPENIGTLINK_TX_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace OpenIGTLinkTxOp {

PYDOC(OpenIGTLinkTxOp, R"doc(
Operator class to transmit data using the OpenIGTLink protocol.
)doc")

// PyOpenIGTLinkTxOp Constructor
PYDOC(OpenIGTLinkTxOp_python, R"doc(
Operator class to transmit data using the OpenIGTLink protocol.

Named inputs:
    receivers: multi-receiver accepting nvidia::gxf::Tensor and/or nvidia::gxf::VideoBuffer
        The inputs are converted to igtl::ImageMessage and sent out over the network with
        the OpenIGTLink protocol.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
device_name : str, optional
    OpenIGTLink device name.
input_names : std::vector<std::string>, optional.
    Names of input messages.
host_name : str, optional.
    Host name.
port : integer, optional
    Port number of server.

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

}  // namespace OpenIGTLinkTxOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_OPENIGTLINK_TX_PYDOC_HPP
