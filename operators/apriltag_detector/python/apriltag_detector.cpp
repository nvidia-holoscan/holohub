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

#include "../apriltag_detector.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include <holoscan/python/core/emitter_receiver_registry.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */
class PyApriltagDetectorOp : public ApriltagDetectorOp {
 public:
  /* Inherit the constructors */
  using ApriltagDetectorOp::ApriltagDetectorOp;

  // Define a constructor that fully initializes the object.
  PyApriltagDetectorOp(holoscan::Fragment* fragment, int width, int height, int number_of_tags,
                       const std::string& name = "apriltag_detector")
      : ApriltagDetectorOp(holoscan::ArgList{holoscan::Arg{"width", width},
                                             holoscan::Arg{"height", height},
                                             holoscan::Arg{"number_of_tags", number_of_tags}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

// Custom function to convert the array of `float2` to a Python list
py::list corners_to_list(const ApriltagDetectorOp::output_corners& oc) {
  py::list corners_list;
  for (const auto& corner : oc.corners) { corners_list.append(py::make_tuple(corner.x, corner.y)); }
  return corners_list;
}

// Custom function to set the array of `float2` from a Python list
void list_to_corners(py::list list, ApriltagDetectorOp::output_corners& oc) {
  for (size_t i = 0; i < 4; ++i) {
    py::tuple item = list[i].cast<py::tuple>();
    oc.corners[i].x = item[0].cast<float>();
    oc.corners[i].y = item[1].cast<float>();
  }
}

PYBIND11_MODULE(_apriltag_detector, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK ApriltagDetectionOp Python Bindings
        ----------------------------------------------
        .. currentmodule:: _apriltag_detection
    )pbdoc";

  auto op = py::class_<ApriltagDetectorOp,
                       PyApriltagDetectorOp,
                       holoscan::Operator,
                       std::shared_ptr<ApriltagDetectorOp>>(m, "ApriltagDetectorOp")
                .def(py::init<holoscan::Fragment*, int, int, int, const std::string&>(),
                     "fragment"_a,
                     "width"_a,
                     "height"_a,
                     "number_of_tags"_a,
                     "name"_a = "apriltag_detector"s)
                .def("setup", &ApriltagDetectorOp::setup, "spec"_a);

  // register the CUDA float2 and output_corners type with Python
  py::class_<float2>(m, "float2")
      .def(py::init<float, float>(), "x"_a = 0.0, "y"_a = 0.0)
      .def_readwrite("x", &float2::x)
      .def_readwrite("y", &float2::y)
      .def(
          "__repr__",
          [](const float2& f) { return fmt::format("({}, {})", f.x, f.y); },
          R"doc(Return repr(self).)doc");

  py::class_<ApriltagDetectorOp::output_corners>(m, "output_corners")
      .def(py::init<>())  // Default constructor
      .def_property(
          "corners",
          [](const ApriltagDetectorOp::output_corners& oc) { return corners_to_list(oc); },
          [](ApriltagDetectorOp::output_corners& oc, py::list list) { list_to_corners(list, oc); })
      .def_readwrite("id", &ApriltagDetectorOp::output_corners::id)  // Expose the 'id' member
      .def(
          "__repr__",
          [](const ApriltagDetectorOp::output_corners& o) {
            std::string repr = "<example.output_corners corners=[";
            for (size_t i = 0; i < 4; ++i) {
              repr += "(" + std::to_string(o.corners[i].x) + ", " + std::to_string(o.corners[i].y) +
                      ")";
              if (i < 3) repr += ", ";
            }
            repr += "] id=" + std::to_string(o.id) + ">";
            return repr;
          },
          R"doc(Return repr(self).)doc");

  // Import the emitter/receiver registry from holoscan.core and pass it to this function to
  // register this new C++ type with the SDK.
  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<std::vector<ApriltagDetectorOp::output_corners>>(
        "std::vector<ApriltagDetectorOp::output_corners>"s);
  });
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
