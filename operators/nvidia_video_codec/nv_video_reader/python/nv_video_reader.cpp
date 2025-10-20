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

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "../nv_video_reader.hpp"
#include "./nv_video_reader_pydoc.hpp"

#include "../../../operator_util.hpp"
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyNvVideoReaderOp : public NvVideoReaderOp {
 public:
  /* Inherit the constructors */
  using NvVideoReaderOp::NvVideoReaderOp;

  // Define a constructor that fully initializes the object.
  PyNvVideoReaderOp(Fragment* fragment, const py::args& args, const std::string& directory,
                    const std::string& filename, std::shared_ptr<::holoscan::Allocator> allocator,
                    bool loop = false, bool verbose = false,
                    const std::string& name = "nv_video_reader")
      : NvVideoReaderOp(ArgList{Arg{"directory", directory},
                                Arg{"filename", filename},
                                Arg{"allocator", allocator},
                                Arg{"loop", loop},
                                Arg{"verbose", verbose}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_nv_video_reader, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _nv_video_reader
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<NvVideoReaderOp, PyNvVideoReaderOp, Operator, std::shared_ptr<NvVideoReaderOp>>(
      m, "NvVideoReaderOp", doc::NvVideoReaderOp::doc_NvVideoReaderOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    std::shared_ptr<::holoscan::Allocator>,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "directory"_a,
           "filename"_a,
           "allocator"_a,
           "loop"_a = false,
           "verbose"_a = false,
           "name"_a = "nv_video_reader"s,
           doc::NvVideoReaderOp::doc_NvVideoReaderOp)
      .def("initialize", &NvVideoReaderOp::initialize, doc::NvVideoReaderOp::doc_initialize)
      .def("setup", &NvVideoReaderOp::setup, "spec"_a, doc::NvVideoReaderOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
