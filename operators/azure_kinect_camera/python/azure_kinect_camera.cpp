#pragma clang diagnostic push
#pragma ide diagnostic ignored "VirtualCallInCtorOrDtor"
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../../operator_util.hpp"

#include "../azure_kinect_camera.hpp"
#include "./azure_kinect_camera_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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
class PyAzureKinectCameraOp : public AzureKinectCameraOp {
 public:
  /* Inherit the constructors */
  using AzureKinectCameraOp::AzureKinectCameraOp;

  // Define a constructor that fully initializes the object.
  PyAzureKinectCameraOp(Fragment* fragment, const py::args& args,
                   const std::shared_ptr<Allocator>& allocator,
                   const std::string& device_serial = "ANY",
                   const unsigned int capture_timeout_ms = 30,
                   const std::string& name = "azure_kinect_camera")
      : AzureKinectCameraOp(ArgList{Arg{"allocator", allocator},
                                    Arg{"device_serial", device_serial},
                                    Arg{"capture_timeout_ms", capture_timeout_ms}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_azure_kinect_camera, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _azure_kinect_camera
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<AzureKinectCameraOp, PyAzureKinectCameraOp, Operator, std::shared_ptr<AzureKinectCameraOp>>(
      m, "AzureKinectCameraOp", doc::AzureKinectCameraOp::doc_AzureKinectCameraOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::shared_ptr<Allocator>&,
                    const std::string&,
                    unsigned int,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "device_serial"_a = "ANY"s,
           "capture_timeout_ms"_a = 30,
           "name"_a = "azure_kinect_camera"s,
           doc::AzureKinectCameraOp::doc_AzureKinectCameraOp_python)
      .def("setup", &AzureKinectCameraOp::setup, "spec"_a, doc::AzureKinectCameraOp::doc_setup);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops

#pragma clang diagnostic pop