# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#include <pybind11/pybind11.h>

#include "../streaming_client.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <pybind11/stl.h>

using std::string_literals::operator""s;
namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs */
class PyStreamingClientOp : public StreamingClientOp {
 public:
  /* Inherit the constructors */
  using StreamingClientOp::StreamingClientOp;

  /* For handling kwargs in Python */
  PyStreamingClientOp(const py::kwargs& kwargs) : StreamingClientOp() {
    // Set the name from Python if provided
    if (kwargs.contains("name"s)) {
      name_ = kwargs["name"s].cast<std::string>();
    }
  }
};

PYBIND11_MODULE(streaming_client, m) {
  py::class_<StreamingClientOp, PyStreamingClientOp, Operator, std::shared_ptr<StreamingClientOp>>(
      m, "StreamingClientOp")
      .def(py::init<>())
      .def(py::init<const py::kwargs&>());
}

}  // namespace holoscan::ops
