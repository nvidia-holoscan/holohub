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

#ifndef PYHOLOHUB_OPERATORS_VIDEO_DECODER_CONTEXT_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_VIDEO_DECODER_CONTEXT_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace VideoDecoderContext {

PYDOC(VideoDecoderContext, R"doc(
Default serializer for GXF entities.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
response_scheduling_term: holoscan.AsynchronousCondition
    Async Scheduling Term required to get/set event state.
device_id: int
    CUDA device ID.
name : str, optional
    The name of the serializer.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the resource.
)doc")

}  // namespace VideoDecoderContext

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_VIDEO_ENCODER_CONTEXT_PYDOC_HPP
