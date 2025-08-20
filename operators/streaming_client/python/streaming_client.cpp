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

#include "../streaming_client.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

#include "../../operator_util.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs */
class PyStreamingClientOp : public StreamingClientOp {
 public:
  /* Inherit the constructors */
  using StreamingClientOp::StreamingClientOp;

  /* Constructor that takes Fragment and args like other operators */
  explicit PyStreamingClientOp(Fragment* fragment, const py::args& args,
                              const std::string& name = "streaming_client")
      : StreamingClientOp() {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_streaming_client_operator, m) {
  py::class_<StreamingClientOp, PyStreamingClientOp, Operator, std::shared_ptr<StreamingClientOp>>(
      m, "StreamingClientOp")
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           "fragment"_a, "name"_a = "streaming_client"s)
      .def("initialize", &StreamingClientOp::initialize)
      .def("setup", &StreamingClientOp::setup, "spec"_a);
}

}  // namespace holoscan::ops
