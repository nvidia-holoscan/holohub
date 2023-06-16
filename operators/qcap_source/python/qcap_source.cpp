/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights reserved.
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

#include "../qcap_source.hpp"
#include "./qcap_source_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>
#include "holoscan/core/gxf/gxf_operator.hpp"

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

class PyQCAPSourceOp : public QCAPSourceOp {
 public:
  /* Inherit the constructors */
  using QCAPSourceOp::QCAPSourceOp;

  // Define a constructor that fully initializes the object.
  PyQCAPSourceOp(Fragment* fragment, const std::string& device = "SC0710 PCI"s,
                 uint32_t channel = 0, uint32_t width = 3840, uint32_t height = 2160,
                 uint32_t framerate = 60, bool rdma = true,
                 const std::string & pixel_format = "bgr24"s,
                 const std::string & input_type = "auto"s,
                 uint32_t mst_mode = 0, uint32_t sdi12g_mode = 0,
                 const std::string& name = "qcap_source")
      : QCAPSourceOp(ArgList{Arg{"device", device},
                             Arg{"channel", channel},
                             Arg{"width", width},
                             Arg{"height", height},
                             Arg{"framerate", framerate},
                             Arg{"rdma", rdma},
                             Arg{"pixel_format", pixel_format},
                             Arg{"input_type", input_type},
                             Arg{"mst_mode", mst_mode},
                             Arg{"sdi12g_mode", sdi12g_mode}
                             }) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

PYBIND11_MODULE(_qcap_source, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _qcap_source
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

  py::class_<QCAPSourceOp, PyQCAPSourceOp, GXFOperator, std::shared_ptr<QCAPSourceOp>>(
      m, "QCAPSourceOp", doc::QCAPSourceOp::doc_QCAPSourceOp)
      .def(py::init<Fragment*,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    const std::string&>(),
           "fragment"_a,
           "device"_a = "SC0710 PCI"s,
           "channel"_a = 0,
           "width"_a = 3840,
           "height"_a = 2160,
           "framerate"_a = 60,
           "rdma"_a = true,
           "pixel_format"_a = "bgr24"s,
           "input_type"_a = "auto"s,
           "mst_mode"_a = 0,
           "sdi12g_mode"_a = 0,
           "name"_a = "qcap_source"s,
           doc::QCAPSourceOp::doc_QCAPSourceOp_python)
      .def_property_readonly(
          "gxf_typename", &QCAPSourceOp::gxf_typename, doc::QCAPSourceOp::doc_gxf_typename)
      .def("initialize", &QCAPSourceOp::initialize, doc::QCAPSourceOp::doc_initialize)
      .def("setup", &QCAPSourceOp::setup, "spec"_a, doc::QCAPSourceOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
