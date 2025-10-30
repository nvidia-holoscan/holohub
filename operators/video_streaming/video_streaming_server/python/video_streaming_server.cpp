/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "../video_streaming_server_resource.hpp"
#include "../video_streaming_server_upstream_op.hpp"
#include "../video_streaming_server_downstream_op.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include "../../operator_util.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline classes for handling Python kwargs */

class PyStreamingServerUpstreamOp : public StreamingServerUpstreamOp {
 public:
  using StreamingServerUpstreamOp::StreamingServerUpstreamOp;

  explicit PyStreamingServerUpstreamOp(Fragment* fragment,
                                       uint32_t width = 854,
                                       uint32_t height = 480,
                                       uint32_t fps = 30,
                                       std::shared_ptr<Allocator> allocator = nullptr,
                                       std::shared_ptr<StreamingServerResource>
                                           streaming_server_resource = nullptr,
                                       const std::string& name =
                                           "streaming_server_upstream"s,
                                       const py::args& args = py::args())
      : StreamingServerUpstreamOp(ArgList{Arg{"width", width},
                                          Arg{"height", height},
                                          Arg{"fps", fps}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    add_positional_condition_and_resource_args(this, args);
    if (allocator) {
      this->add_arg(Arg("allocator", allocator));
    }
    if (streaming_server_resource) {
      this->add_arg(Arg("streaming_server_resource", streaming_server_resource));
    }
    setup(*spec_);
  }
};

class PyStreamingServerDownstreamOp : public StreamingServerDownstreamOp {
 public:
  using StreamingServerDownstreamOp::StreamingServerDownstreamOp;

  explicit PyStreamingServerDownstreamOp(Fragment* fragment,
                                         uint32_t width = 854,
                                         uint32_t height = 480,
                                        uint32_t fps = 30,
                                        bool enable_processing = false,
                                        const std::string& processing_type =
                                            "none"s,
                                        std::shared_ptr<Allocator> allocator = nullptr,
                                        std::shared_ptr<StreamingServerResource>
                                            streaming_server_resource = nullptr,
                                        const std::string& name =
                                            "streaming_server_downstream"s,
                                        const py::args& args = py::args())
      : StreamingServerDownstreamOp(ArgList{Arg{"width", width},
                                            Arg{"height", height},
                                            Arg{"fps", fps},
                                            Arg{"enable_processing", enable_processing},
                                            Arg{"processing_type", processing_type}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    add_positional_condition_and_resource_args(this, args);
    if (allocator) {
      this->add_arg(Arg("allocator", allocator));
    }
    if (streaming_server_resource) {
      this->add_arg(Arg("streaming_server_resource", streaming_server_resource));
    }
    setup(*spec_);
  }
};

class PyStreamingServerResource : public StreamingServerResource {
 public:
  using StreamingServerResource::StreamingServerResource;

  explicit PyStreamingServerResource(Fragment* fragment,
                                     uint16_t port = 48010,
                                     uint32_t width = 854,
                                     uint32_t height = 480,
                                     uint16_t fps = 30,
                                     const std::string& server_name = "HoloscanStreamingServer"s,
                                     bool enable_upstream = true,
                                     bool enable_downstream = true,
                                     bool is_multi_instance = false,
                                     const std::string& name = "streaming_server_resource"s)
      : StreamingServerResource(ArgList{Arg{"port", port},
                                        Arg{"width", width},
                                        Arg{"height", height},
                                        Arg{"fps", fps},
                                        Arg{"server_name", server_name},
                                        Arg{"enable_upstream", enable_upstream},
                                        Arg{"enable_downstream", enable_downstream},
                                        Arg{"is_multi_instance", is_multi_instance}}) {
    name_ = name;
    fragment_ = fragment;

    // Create and setup the ComponentSpec to avoid "No component spec" warning
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

PYBIND11_MODULE(_video_streaming_server, m) {
  m.doc() = R"pbdoc(
      Holoscan SDK Streaming Server Enhanced Python Bindings
      ---------------------------------------------------
      .. currentmodule:: holohub.video_streaming_server
  )pbdoc";

  // StreamingServerResource
  py::class_<StreamingServerResource, PyStreamingServerResource, holoscan::Resource,
             std::shared_ptr<StreamingServerResource>>(
      m, "StreamingServerResource", R"doc(
Resource that manages a streaming server for video streaming in Holoscan.

This resource provides centralized management for streaming server operations,
handling both upstream (receiving) and downstream (sending) connections.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the resource belongs to.
port : int, optional
    Port number for the streaming server. Default value is ``48010``.
width : int, optional
    Frame width. Default value is ``854``.
height : int, optional
    Frame height. Default value is ``480``.
fps : int, optional
    Frames per second. Default value is ``30``.
server_name : str, optional
    Name of the streaming server. Default value is ``"HoloscanStreamingServer"``.
enable_upstream : bool, optional
    Enable upstream (receiving) functionality. Default value is ``True``.
enable_downstream : bool, optional
    Enable downstream (sending) functionality. Default value is ``True``.
is_multi_instance : bool, optional
    Enable multi-instance mode. Default value is ``False``.
name : str, optional (constructor only)
    The name of the resource. Default value is ``"streaming_server_resource"``.
)doc")
      .def(py::init<Fragment*,
                    uint16_t,
                    uint32_t,
                    uint32_t,
                    uint16_t,
                    const std::string&,
                    bool,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "port"_a = 48010,
           "width"_a = 854,
           "height"_a = 480,
           "fps"_a = 30,
           "server_name"_a = "HoloscanStreamingServer"s,
           "enable_upstream"_a = true,
           "enable_downstream"_a = true,
           "is_multi_instance"_a = false,
           "name"_a = "streaming_server_resource"s);

  // StreamingServerUpstreamOp
  py::class_<StreamingServerUpstreamOp, PyStreamingServerUpstreamOp, Operator,
             std::shared_ptr<StreamingServerUpstreamOp>>(
      m, "StreamingServerUpstreamOp", R"doc(
Operator that handles upstream (receiving) video streaming from clients.

This operator receives video frames from streaming clients and outputs them
for further processing in the Holoscan pipeline.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
streaming_server_resource : holohub.video_streaming_server.StreamingServerResource
    The shared StreamingServerResource for managing server connection.
width : int, optional
    Frame width. Default value is ``854``.
height : int, optional
    Frame height. Default value is ``480``.
fps : int, optional
    Frames per second. Default value is ``30``.
allocator : holoscan.resources.Allocator, optional
    Memory allocator to use. Default value is ``None``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"streaming_server_upstream"``.
)doc")
      .def(py::init<Fragment*,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    std::shared_ptr<Allocator>,
                    std::shared_ptr<StreamingServerResource>,
                    const std::string&,
                    const py::args&>(),
           "fragment"_a,
           "width"_a = 854,
           "height"_a = 480,
           "fps"_a = 30,
           "allocator"_a = py::none(),
           "streaming_server_resource"_a = py::none(),
           "name"_a = "streaming_server_upstream"s);

  // StreamingServerDownstreamOp
  py::class_<StreamingServerDownstreamOp, PyStreamingServerDownstreamOp, Operator,
             std::shared_ptr<StreamingServerDownstreamOp>>(
      m, "StreamingServerDownstreamOp", R"doc(
Operator that handles downstream (sending) video streaming to clients.

This operator receives processed video frames from the Holoscan pipeline
and sends them to connected streaming clients.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
streaming_server_resource : holohub.video_streaming_server.StreamingServerResource
    The shared StreamingServerResource for managing server connection.
width : int, optional
    Frame width. Default value is ``854``.
height : int, optional
    Frame height. Default value is ``480``.
fps : int, optional
    Frames per second. Default value is ``30``.
enable_processing : bool, optional
    Enable frame processing. Default value is ``False``.
processing_type : str, optional
    Type of processing to apply. Default value is ``"none"``.
allocator : holoscan.resources.Allocator, optional
    Memory allocator to use. Default value is ``None``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"streaming_server_downstream"``.
)doc")
      .def(py::init<Fragment*,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
                    const std::string&,
                    std::shared_ptr<Allocator>,
                    std::shared_ptr<StreamingServerResource>,
                    const std::string&,
                    const py::args&>(),
           "fragment"_a,
           "width"_a = 854,
           "height"_a = 480,
           "fps"_a = 30,
           "enable_processing"_a = false,
           "processing_type"_a = "none"s,
           "allocator"_a = py::none(),
           "streaming_server_resource"_a = py::none(),
           "name"_a = "streaming_server_downstream"s);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
