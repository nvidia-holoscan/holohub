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

#include "../streaming_client.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/arg.hpp>
#include "../operator_util.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs */
class PyStreamingClientOp : public StreamingClientOp {
 public:
  /* Inherit the constructors */
  using StreamingClientOp::StreamingClientOp;

  /* Constructor for Python with proper fragment setup */
  explicit PyStreamingClientOp(Fragment* fragment,
                               const py::args& args,
                               uint32_t width = 640,
                               uint32_t height = 480,
                               uint32_t fps = 30,
                               const std::string& server_ip = "127.0.0.1",
                               uint16_t signaling_port = 48010,
                               bool receive_frames = false,
                               bool send_frames = false,
                               uint32_t min_non_zero_bytes = 100,
                               const std::string& name = "streaming_client")
      : StreamingClientOp(ArgList{Arg{"width", width},
                                  Arg{"height", height},
                                  Arg{"fps", fps},
                                  Arg{"server_ip", server_ip},
                                  Arg{"signaling_port", signaling_port},
                                  Arg{"receive_frames", receive_frames},
                                  Arg{"send_frames", send_frames},
                                  Arg{"min_non_zero_bytes", min_non_zero_bytes}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_streaming_client, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK StreamingClient Python Bindings
        ---------------------------------------
        .. currentmodule:: _streaming_client
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<StreamingClientOp, PyStreamingClientOp, Operator, std::shared_ptr<StreamingClientOp>>(
      m, "StreamingClientOp", R"doc(
Operator that wraps the StreamingClient for video streaming in Holoscan.

This operator provides integration with the StreamingClient library,
allowing Holoscan applications to send and receive video streams.

Parameters
----------
width : uint32_t, optional
    Width of the video frames in pixels. Default is 640.
height : uint32_t, optional
    Height of the video frames in pixels. Default is 480.
fps : uint32_t, optional
    Frame rate of the video. Default is 30.
server_ip : str, optional
    IP address of the streaming server. Default is "127.0.0.1".
signaling_port : uint16_t, optional
    Port used for signaling with the server. Default is 48010.
receive_frames : bool, optional
    Whether to receive frames from the server. Default is False.
send_frames : bool, optional
    Whether to send frames to the server. Default is False.
min_non_zero_bytes : uint32_t, optional
    Minimum non-zero bytes required in frame data for validation. Default is 100.
name : str, optional
    The name of the operator. Default is "streaming_client".
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
                    const std::string&>(),
           "fragment"_a,
           "width"_a = 640,
           "height"_a = 480,
           "fps"_a = 30,
           "server_ip"_a = "127.0.0.1"s,
           "signaling_port"_a = 48010,
           "receive_frames"_a = false,
           "send_frames"_a = false,
           "min_non_zero_bytes"_a = 100,
           "name"_a = "streaming_client"s,
           R"doc(Create a streaming client operator.

Parameters
----------
fragment : Fragment
    The fragment that contains this operator.
width : uint32_t, optional
    Width of the video frames in pixels. Default is 640.
height : uint32_t, optional
    Height of the video frames in pixels. Default is 480.
fps : uint32_t, optional
    Frame rate of the video. Default is 30.
server_ip : str, optional
    IP address of the streaming server. Default is "127.0.0.1".
signaling_port : uint16_t, optional
    Port used for signaling with the server. Default is 48010.
receive_frames : bool, optional
    Whether to receive frames from the server. Default is False.
send_frames : bool, optional
    Whether to send frames to the server. Default is False.
min_non_zero_bytes : uint32_t, optional
    Minimum non-zero bytes required in frame data for validation. Default is 100.
name : str, optional
    The name of the operator. Default is "streaming_client".
)doc")
      .def("initialize", &StreamingClientOp::initialize, "Initialize the operator")
      .def("setup", &StreamingClientOp::setup, "spec"_a, "Setup the operator");
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
