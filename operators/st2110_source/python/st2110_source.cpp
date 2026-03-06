/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, XRlabs. All rights reserved.
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

#include "../st2110_source.hpp"
#include "./st2110_source_pydoc.hpp"

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

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs
 *
 * This adds a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments provide a Pythonic kwarg-based
 * interface with appropriate default values matching the operator's default parameters
 * in the C++ API `setup` method.
 */

class PyST2110SourceOp : public ST2110SourceOp {
 public:
  /* Inherit the constructors */
  using ST2110SourceOp::ST2110SourceOp;

  // Define a constructor that fully initializes the object.
  PyST2110SourceOp(Fragment* fragment,
                    const py::args& args,
                    const std::string& multicast_address = "239.0.0.1",
                    uint16_t port = 5004,
                    const std::string& interface_name = "eth0",
                    uint32_t width = 1920,
                    uint32_t height = 1080,
                    uint32_t framerate = 60,
                    const std::string& stream_format = "YCbCr-4:2:2-10bit",
                    bool enable_rgba_output = false,
                    bool enable_nv12_output = false,
                    uint32_t batch_size = 1000,
                    uint16_t max_packet_size = 1514,
                    uint16_t header_size = 42,
                    uint16_t rtp_header_size = 12,
                    bool enable_reorder_kernel = true,
                    const std::string& name = "st2110_source")
      : ST2110SourceOp(ArgList{Arg{"multicast_address", multicast_address},
                                Arg{"port", port},
                                Arg{"interface_name", interface_name},
                                Arg{"width", width},
                                Arg{"height", height},
                                Arg{"framerate", framerate},
                                Arg{"stream_format", stream_format},
                                Arg{"enable_rgba_output", enable_rgba_output},
                                Arg{"enable_nv12_output", enable_nv12_output},
                                Arg{"batch_size", batch_size},
                                Arg{"max_packet_size", max_packet_size},
                                Arg{"header_size", header_size},
                                Arg{"rtp_header_size", rtp_header_size},
                                Arg{"enable_reorder_kernel", enable_reorder_kernel}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_st2110_source, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _st2110_source
        .. autosummary::
           :toctree: _generate
           ST2110SourceOp
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<ST2110SourceOp,
             PyST2110SourceOp,
             Operator,
             std::shared_ptr<ST2110SourceOp>>(
      m,
      "ST2110SourceOp",
      doc::ST2110SourceOp::doc_ST2110SourceOp_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    uint16_t,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    const std::string&,
                    bool,
                    bool,
                    uint32_t,
                    uint16_t,
                    uint16_t,
                    uint16_t,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "multicast_address"_a = "239.0.0.1"s,
           "port"_a = 5004,
           "interface_name"_a = "eth0"s,
           "width"_a = 1920,
           "height"_a = 1080,
           "framerate"_a = 60,
           "stream_format"_a = "YCbCr-4:2:2-10bit"s,
           "enable_rgba_output"_a = false,
           "enable_nv12_output"_a = false,
           "batch_size"_a = 1000,
           "max_packet_size"_a = 1514,
           "header_size"_a = 42,
           "rtp_header_size"_a = 12,
           "enable_reorder_kernel"_a = true,
           "name"_a = "st2110_source"s,
           doc::ST2110SourceOp::doc_ST2110SourceOp_python);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
