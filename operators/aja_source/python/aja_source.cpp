/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>

#include "../operator_util.hpp"
#include "./aja_source_pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "../aja_source.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

namespace {

// using constexpr constructor instead of unordered_map here to make clang-tidy happy
// (avoids warning of type: fuchsia-statically-constructed-objects)
constexpr std::array<std::pair<std::string_view, NTV2Channel>, 8> NTV2ChannelMapping = {
    {{"NTV2_CHANNEL1", NTV2Channel::NTV2_CHANNEL1},
     {"NTV2_CHANNEL2", NTV2Channel::NTV2_CHANNEL2},
     {"NTV2_CHANNEL3", NTV2Channel::NTV2_CHANNEL3},
     {"NTV2_CHANNEL4", NTV2Channel::NTV2_CHANNEL4},
     {"NTV2_CHANNEL5", NTV2Channel::NTV2_CHANNEL5},
     {"NTV2_CHANNEL6", NTV2Channel::NTV2_CHANNEL6},
     {"NTV2_CHANNEL7", NTV2Channel::NTV2_CHANNEL7},
     {"NTV2_CHANNEL8", NTV2Channel::NTV2_CHANNEL8}}};

constexpr NTV2Channel ToNTV2Channel(std::string_view value) {
  for (const auto& [name, channel] : NTV2ChannelMapping) {
    if (name == value) { return channel; }
  }
  return NTV2Channel::NTV2_CHANNEL_INVALID;
}

}  // namespace

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyAJASourceOp : public AJASourceOp {
 public:
  /* Inherit the constructors */
  using AJASourceOp::AJASourceOp;

  // Define a constructor that fully initializes the object.
  PyAJASourceOp(
      Fragment* fragment, const py::args& args, const std::string& device = "0"s,
      const std::variant<std::string, NTV2Channel>& channel = NTV2Channel::NTV2_CHANNEL1,
      uint32_t width = 1920, uint32_t height = 1080, uint32_t framerate = 60,
      bool interlaced = false, bool rdma = false, bool enable_overlay = false,
      const std::variant<std::string, NTV2Channel>& overlay_channel = NTV2Channel::NTV2_CHANNEL2,
      bool overlay_rdma = true, const std::string& name = "aja_source")
      : AJASourceOp(ArgList{Arg{"device", device},
                            Arg{"width", width},
                            Arg{"height", height},
                            Arg{"framerate", framerate},
                            Arg{"interlaced", interlaced},
                            Arg{"rdma", rdma},
                            Arg{"enable_overlay", enable_overlay},
                            Arg{"overlay_rdma", overlay_rdma}}) {
    add_positional_condition_and_resource_args(this, args);
    if (std::holds_alternative<std::string>(channel)) {
      this->add_arg(Arg("channel", ToNTV2Channel(std::get<std::string>(channel))));
    } else {
      this->add_arg(Arg("channel", std::get<NTV2Channel>(channel)));
    }
    if (std::holds_alternative<std::string>(overlay_channel)) {
      this->add_arg(Arg("overlay_channel", ToNTV2Channel(std::get<std::string>(overlay_channel))));
    } else {
      this->add_arg(Arg("overlay_channel", std::get<NTV2Channel>(overlay_channel)));
    }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

/* The python module */

PYBIND11_MODULE(_aja_source, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK AJASourceOp Python Bindings
        ---------------------------------------
        .. currentmodule:: _aja_source
    )pbdoc";

  py::enum_<NTV2Channel>(m, "NTV2Channel")
      .value("NTV2_CHANNEL1", NTV2Channel::NTV2_CHANNEL1)
      .value("NTV2_CHANNEL2", NTV2Channel::NTV2_CHANNEL2)
      .value("NTV2_CHANNEL3", NTV2Channel::NTV2_CHANNEL3)
      .value("NTV2_CHANNEL4", NTV2Channel::NTV2_CHANNEL4)
      .value("NTV2_CHANNEL5", NTV2Channel::NTV2_CHANNEL5)
      .value("NTV2_CHANNEL6", NTV2Channel::NTV2_CHANNEL6)
      .value("NTV2_CHANNEL7", NTV2Channel::NTV2_CHANNEL7)
      .value("NTV2_CHANNEL8", NTV2Channel::NTV2_CHANNEL8)
      .value("NTV2_MAX_NUM_CHANNELS", NTV2Channel::NTV2_MAX_NUM_CHANNELS)
      .value("NTV2_CHANNEL_INVALID", NTV2Channel::NTV2_CHANNEL_INVALID);

  py::class_<AJASourceOp, PyAJASourceOp, Operator, std::shared_ptr<AJASourceOp>>(
      m, "AJASourceOp", doc::AJASourceOp::doc_AJASourceOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::variant<std::string, NTV2Channel>,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
                    bool,
                    bool,
                    const std::variant<std::string, NTV2Channel>,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "device"_a = "0"s,
           "channel"_a = NTV2Channel::NTV2_CHANNEL1,
           "width"_a = 1920,
           "height"_a = 1080,
           "framerate"_a = 60,
           "interlaced"_a = false,
           "rdma"_a = false,
           "enable_overlay"_a = false,
           "overlay_channel"_a = NTV2Channel::NTV2_CHANNEL2,
           "overlay_rdma"_a = true,
           "name"_a = "aja_source"s,
           doc::AJASourceOp::doc_AJASourceOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
