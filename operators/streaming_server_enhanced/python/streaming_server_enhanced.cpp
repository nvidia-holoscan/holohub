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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../streaming_server_resource.hpp"
#include "../streaming_server_upstream_op.hpp"
#include "../streaming_server_downstream_op.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

PYBIND11_MODULE(_streaming_server_enhanced, m) {
  m.doc() = R"pbdoc(
      Holoscan SDK Streaming Server 04_80 Tensor Python Bindings
      -------------------------------------------------------
      .. currentmodule:: _streaming_server_enhanced
  )pbdoc";

  // StreamingServerResource
  py::class_<StreamingServerResource, holoscan::Resource, std::shared_ptr<StreamingServerResource>>(
      m, "StreamingServerResource", R"doc(
Resource that manages a streaming server for video streaming in Holoscan.

This resource provides centralized management for streaming server operations,
handling both upstream (receiving) and downstream (sending) connections.

Parameters
----------
width : uint32_t, optional
    Width of the video frames in pixels. Default is 854.
height : uint32_t, optional
    Height of the video frames in pixels. Default is 480.
fps : uint32_t, optional
    Frame rate of the video. Default is 30.
port : uint16_t, optional
    Port used for streaming server. Default is 48010.
)doc")
      .def(py::init<>(), R"doc(Create a streaming server resource.)doc")
      .def("initialize", &StreamingServerResource::initialize, "Initialize the resource")
      .def("setup", &StreamingServerResource::setup, "spec"_a, "Setup the resource");

  // StreamingServerUpstreamOp
  py::class_<StreamingServerUpstreamOp, holoscan::Operator, std::shared_ptr<StreamingServerUpstreamOp>>(
      m, "StreamingServerUpstreamOp", R"doc(
Operator that handles upstream (receiving) video streaming from clients.

This operator receives frames from streaming clients and emits holoscan::Tensor
objects to the Holoscan pipeline.

Parameters
----------
width : uint32_t, optional
    Width of the video frames in pixels. Default is 854.
height : uint32_t, optional
    Height of the video frames in pixels. Default is 480.
fps : uint32_t, optional
    Frame rate of the video. Default is 30.
streaming_server_resource : StreamingServerResource
    The streaming server resource to use.
allocator : holoscan.resources.Allocator, optional
    Memory allocator for frame data.
)doc")
      .def(py::init<>(), R"doc(Create a streaming server upstream operator.)doc")
      .def("initialize", &StreamingServerUpstreamOp::initialize, "Initialize the operator")
      .def("setup", &StreamingServerUpstreamOp::setup, "spec"_a, "Setup the operator");

  // StreamingServerDownstreamOp
  py::class_<StreamingServerDownstreamOp, holoscan::Operator, std::shared_ptr<StreamingServerDownstreamOp>>(
      m, "StreamingServerDownstreamOp", R"doc(
Operator that handles downstream (sending) video streaming to clients.

This operator receives holoscan::Tensor objects, processes them, and sends
the processed frames to connected streaming clients.

Parameters
----------
width : uint32_t, optional
    Width of the video frames in pixels. Default is 854.
height : uint32_t, optional
    Height of the video frames in pixels. Default is 480.
fps : uint32_t, optional
    Frame rate of the video. Default is 30.
streaming_server_resource : StreamingServerResource
    The streaming server resource to use.
)doc")
      .def(py::init<>(), R"doc(Create a streaming server downstream operator.)doc")
      .def("initialize", &StreamingServerDownstreamOp::initialize, "Initialize the operator")
      .def("setup", &StreamingServerDownstreamOp::setup, "spec"_a, "Setup the operator");
}

}  // namespace holoscan::ops
