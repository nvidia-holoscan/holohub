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

#pragma once

#include <string>

#include "macros.hpp"

namespace holoscan::doc::DDSVideoSubscriberOp {

PYDOC(DDSVideoSubscriberOp, R"doc(
DDS Video Subscriber operator.
)doc")

// PyDDSVideoSubscriberOp Constructor
PYDOC(DDSVideoSubscriberOp_python, R"doc(
DDS Video Subscriber operator.

Parameters
----------
allocator : holoscan.resources.Allocator
    Memory pool allocator used by the operator.
qos_provider: str, optional
    URI for the QoS Provider
participant_qos: str, optional
    QoS profile for the domain participant
domain_id : int, optional
    DDS domain to use.
reader_qos: str, optional
    QoS profile for the data reader
stream_id : int, optional
    Stream ID of the video stream.
name : str, optional
    The name of the operator.
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
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace holoscan::doc::DDSVideoSubscriberOp
