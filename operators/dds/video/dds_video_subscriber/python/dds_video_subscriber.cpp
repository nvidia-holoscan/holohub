/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <memory>
#include <string>

#include "../dds_video_subscriber.hpp"
#include "./dds_video_subscriber_pydoc.hpp"

#include "../../../../operator_util.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

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

class PyDDSVideoSubscriberOp : public DDSVideoSubscriberOp {
 public:
  /* Inherit the constructors */
  using DDSVideoSubscriberOp::DDSVideoSubscriberOp;

  // Define a constructor that fully initializes the object.
  PyDDSVideoSubscriberOp(Fragment* fragment, const py::args& args,
                         std::shared_ptr<::holoscan::Allocator> allocator,
                         const std::string& qos_provider = "",
                         const std::string& participant_qos = "",
                         uint32_t domain_id = 0,
                         const std::string& reader_qos = "",
                         uint32_t stream_id = 0,
                         const std::string& name = "dds_video_subscriber")
      : DDSVideoSubscriberOp(ArgList{Arg{"allocator", allocator},
                                     Arg{"qos_provider", qos_provider},
                                     Arg{"participant_qos", participant_qos},
                                     Arg{"domain_id", domain_id},
                                     Arg{"reader_qos", reader_qos},
                                     Arg{"stream_id", stream_id}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_dds_video_subscriber, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _dds_video_subscriber
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

  py::class_<DDSVideoSubscriberOp,
             PyDDSVideoSubscriberOp,
             Operator,
             std::shared_ptr<DDSVideoSubscriberOp>>(
      m, "DDSVideoSubscriberOp", doc::DDSVideoSubscriberOp::doc_DDSVideoSubscriberOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    const std::string&,
                    uint32_t,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "qos_provider"_a = ""s,
           "participant_qos"_a = ""s,
           "domain_id"_a = 0,
           "reader_qos"_a = ""s,
           "stream_id"_a = 0,
           "name"_a = "dds_video_subscriber"s,
           doc::DDSVideoSubscriberOp::doc_DDSVideoSubscriberOp)
      .def("initialize", &DDSVideoSubscriberOp::initialize,
           doc::DDSVideoSubscriberOp::doc_initialize)
      .def("setup", &DDSVideoSubscriberOp::setup, "spec"_a,
           doc::DDSVideoSubscriberOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
