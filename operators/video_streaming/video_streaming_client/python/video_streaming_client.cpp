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

#include "../video_streaming_client.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/arg.hpp>
#include "../../operator_util.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs */
class PyStreamingClientOp : public VideoStreamingClientOp {
 public:
  /* Inherit the constructors */
  using VideoStreamingClientOp::VideoStreamingClientOp;

  /* Constructor for Python with proper fragment setup */
  explicit PyStreamingClientOp(Fragment* fragment,
                               const py::args& args,
                               uint32_t width = 854,
                               uint32_t height = 480,
                               uint32_t fps = 30,
                               const std::string& server_ip = "127.0.0.1",
                               uint16_t signaling_port = 48010,
                               bool receive_frames = true,
                               bool send_frames = true,
                               uint32_t min_non_zero_bytes = 100,
                               std::shared_ptr<Allocator> allocator = nullptr,
                               const std::string& name = "streaming_client_enhanced"s)
      : VideoStreamingClientOp(ArgList{Arg{"width", width},
                                  Arg{"height", height},
                                  Arg{"fps", fps},
                                  Arg{"server_ip", server_ip},
                                  Arg{"signaling_port", signaling_port},
                                  Arg{"receive_frames", receive_frames},
                                  Arg{"send_frames", send_frames},
                                  Arg{"min_non_zero_bytes", min_non_zero_bytes}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    add_positional_condition_and_resource_args(this, args);
    if (allocator) {
      this->add_arg(Arg("allocator", allocator));
    }
    setup(*spec_);
  }
};

/* The python module */
PYBIND11_MODULE(_streaming_client_enhanced, m) {
  m.doc() = R"pbdoc(
      Holoscan SDK Streaming Client Enhanced Python Bindings
      ---------------------------------------------------
      .. currentmodule:: holohub.streaming_client_enhanced
  )pbdoc";

  py::class_<VideoStreamingClientOp, PyStreamingClientOp, Operator, std::shared_ptr<VideoStreamingClientOp>>(
      m, "VideoStreamingClientOp", R"doc(
Operator that wraps the StreamingClient for video streaming in Holoscan.

This operator provides client-side video streaming capabilities, allowing
applications to send and receive video frames to/from streaming servers.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
width : int, optional
    Width of the video stream. Default value is ``854``.
height : int, optional
    Height of the video stream. Default value is ``480``.
fps : int, optional
    Frames per second. Default value is ``30``.
server_ip : str, optional
    IP address of the streaming server. Default value is ``"127.0.0.1"``.
signaling_port : int, optional
    Signaling port for the streaming server. Default value is ``48010``.
receive_frames : bool, optional
    Boolean indicating whether to receive frames from the server. Default value is ``True``.
send_frames : bool, optional
    Boolean indicating whether to send frames to the server. Default value is ``True``.
min_non_zero_bytes : int, optional
    Minimum non-zero bytes required in frame data to consider the frame valid. Default value is ``100``.
allocator : holoscan.resources.Allocator, optional
    Memory allocator for output buffer allocation. Default value is ``None``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"streaming_client_enhanced"``.
)doc")
      .def(py::init<Fragment*,
                    const py::args&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    const std::string&,
                    uint16_t,
                    bool,
                    bool,
                    uint32_t,
                    std::shared_ptr<Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "width"_a = 854,
           "height"_a = 480,
           "fps"_a = 30,
           "server_ip"_a = "127.0.0.1"s,
           "signaling_port"_a = 48010,
           "receive_frames"_a = true,
           "send_frames"_a = true,
           "min_non_zero_bytes"_a = 100,
           "allocator"_a = py::none(),
           "name"_a = "streaming_client_enhanced"s,
           R"doc(
Constructor for VideoStreamingClientOp.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
width : int, optional
    Width of the video stream.
height : int, optional
    Height of the video stream.
fps : int, optional
    Frames per second.
server_ip : str, optional
    IP address of the streaming server.
signaling_port : int, optional
    Signaling port for the streaming server.
receive_frames : bool, optional
    Boolean indicating whether to receive frames from the server.
send_frames : bool, optional
    Boolean indicating whether to send frames to the server.
min_non_zero_bytes : int, optional
    Minimum non-zero bytes required in frame data.
name : str, optional
    The name of the operator.
)doc");
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
