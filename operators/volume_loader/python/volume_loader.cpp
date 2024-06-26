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

#include "../volume_loader.hpp"
#include "./volume_loader_pydoc.hpp"

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
class PyVolumeLoaderOp : public VolumeLoaderOp {
 public:
  /* Inherit the constructors */
  using VolumeLoaderOp::VolumeLoaderOp;

  // Define a constructor that fully initializes the object.
  PyVolumeLoaderOp(Fragment* fragment, const py::args& args,
                   const std::shared_ptr<Allocator>& allocator, const std::string& file_name,
                   const std::string& name = "volume_loader")
      : VolumeLoaderOp(ArgList{Arg{"allocator", allocator}, Arg{"file_name", file_name}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_volume_loader, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _volume_loader
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

  py::class_<VolumeLoaderOp, PyVolumeLoaderOp, Operator, std::shared_ptr<VolumeLoaderOp>>(
      m, "VolumeLoaderOp", doc::VolumeLoaderOp::doc_VolumeLoaderOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::shared_ptr<Allocator>&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "file_name"_a = "",
           "name"_a = "volume_loader"s,
           doc::VolumeLoaderOp::doc_VolumeLoaderOp_python)
      .def("setup", &VolumeLoaderOp::setup, "spec"_a, doc::VolumeLoaderOp::doc_setup);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
