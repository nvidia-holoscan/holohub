/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Analog Devices, Inc. All rights reserved.
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

#include <cstring>
#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/python/core/emitter_receiver_registry.hpp>
#include "../../operator_util.hpp"

#include "../cpp/iio_attribute_read.hpp"
#include "../cpp/iio_attribute_write.hpp"
#include "../cpp/iio_buffer_read.hpp"
#include "../cpp/iio_buffer_write.hpp"
#include "../cpp/iio_params.hpp"

#include "iio_configurator.hpp"
#include "iio_controller_pydoc.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

// Emitter/receiver implementation for iio_buffer_info_t
template <>
struct emitter_receiver<std::shared_ptr<iio_buffer_info_t>> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    // Convert Python IIOBufferInfo to std::shared_ptr<iio_buffer_info_t>
    auto py_buffer_info = data.cast<iio_buffer_info_t>();
    auto buffer_info = std::make_shared<iio_buffer_info_t>(py_buffer_info);
    py::gil_scoped_release release;
    op_output.emit<std::shared_ptr<iio_buffer_info_t>>(buffer_info, name.c_str(), acq_timestamp);
    return;
  }

  static py::object receive(std::any result, const std::string&, PyInputContext&) {
    HOLOSCAN_LOG_DEBUG("py_receive: std::shared_ptr<iio_buffer_info_t> case");
    auto buffer_info = std::any_cast<std::shared_ptr<iio_buffer_info_t>>(result);
    py::object py_buffer_info = py::cast(*buffer_info);
    return py_buffer_info;
  }
};

template <>
struct emitter_receiver<iio_buffer_info_t> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    auto buffer_info = data.cast<iio_buffer_info_t>();
    py::gil_scoped_release release;
    op_output.emit<iio_buffer_info_t>(buffer_info, name.c_str(), acq_timestamp);
    return;
  }

  static py::object receive(std::any result, const std::string&, PyInputContext&) {
    HOLOSCAN_LOG_DEBUG("py_receive: iio_buffer_info_t case");
    auto buffer_info = std::any_cast<iio_buffer_info_t>(result);
    py::object py_buffer_info = py::cast(buffer_info);
    return py_buffer_info;
  }
};

}  // namespace holoscan

namespace holoscan::ops {

class PyIIOAttributeRead : public IIOAttributeRead {
 public:
  using IIOAttributeRead::IIOAttributeRead;

  PyIIOAttributeRead(holoscan::Fragment* fragment, const py::args& args, std::string ctx,
                     std::string attr_name, std::string dev = "", std::string chan = "",
                     bool channel_is_output = false, const std::string& name = "iio_attribute_read")
      : IIOAttributeRead(ArgList{Arg("ctx", ctx),
                                 Arg("attr_name", attr_name),
                                 Arg("dev", dev),
                                 Arg("chan", chan),
                                 Arg("channel_is_output", channel_is_output)}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyIIOAttributeWrite : public IIOAttributeWrite {
 public:
  using IIOAttributeWrite::IIOAttributeWrite;

  PyIIOAttributeWrite(holoscan::Fragment* fragment, const py::args& args, std::string ctx,
                      std::string attr_name, std::string dev = "", std::string chan = "",
                      bool channel_is_output = false,
                      const std::string& name = "iio_attribute_read")
      : IIOAttributeWrite(ArgList{Arg("ctx", ctx),
                                  Arg("attr_name", attr_name),
                                  Arg("dev", dev),
                                  Arg("chan", chan),
                                  Arg("channel_is_output", channel_is_output)}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyIIOBufferRead : public IIOBufferRead {
 public:
  using IIOBufferRead::IIOBufferRead;

  PyIIOBufferRead(holoscan::Fragment* fragment, const py::args& args, std::string ctx,
                  std::string dev, bool is_cyclic, size_t samples_count,
                  std::vector<std::string> enabled_channel_names,
                  std::vector<bool> enabled_channel_output,
                  const std::string& name = "iio_buffer_read")
      : IIOBufferRead(ArgList{Arg("ctx", ctx),
                              Arg("dev", dev),
                              Arg("is_cyclic", is_cyclic),
                              Arg("samples_count", samples_count),
                              Arg("enabled_channel_names", enabled_channel_names),
                              Arg("enabled_channel_output", enabled_channel_output)}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyIIOBufferWrite : public IIOBufferWrite {
 public:
  using IIOBufferWrite::IIOBufferWrite;

  PyIIOBufferWrite(holoscan::Fragment* fragment, const py::args& args, std::string ctx,
                   std::string dev, bool is_cyclic, std::vector<std::string> enabled_channel_names,
                   std::vector<bool> enabled_channel_output,
                   const std::string& name = "iio_buffer_write")
      : IIOBufferWrite(ArgList{Arg("ctx", ctx),
                               Arg("dev", dev),
                               Arg("is_cyclic", is_cyclic),
                               Arg("enabled_channel_names", enabled_channel_names),
                               Arg("enabled_channel_output", enabled_channel_output)}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyIIOConfigurator : public IIOConfigurator {
 public:
  using IIOConfigurator::IIOConfigurator;

  PyIIOConfigurator(holoscan::Fragment* fragment, const py::args& args, std::string cfg,
                    const std::string& name = "iio_configurator")
      : IIOConfigurator(ArgList{Arg("cfg", cfg)}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_iio_controller, m) {
  m.doc() = R"pbdoc(
    IIO Controller Python Bindings
    -----------------------------------
    .. currentmodule:: _iio_controller
  )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<iio_data_format>(m, "IIODataFormat")
      .def(py::init<>())
      .def(py::init([](py::object pyiio_format) -> iio_data_format {
             // Constructor that accepts a pyiio data format object
             iio_data_format fmt;
             fmt.length = pyiio_format.attr("length").cast<unsigned int>();
             fmt.bits = pyiio_format.attr("bits").cast<unsigned int>();
             fmt.shift = pyiio_format.attr("shift").cast<unsigned int>();
             fmt.is_signed = pyiio_format.attr("is_signed").cast<bool>();
             fmt.is_fully_defined = pyiio_format.attr("is_fully_defined").cast<bool>();
             fmt.is_be = pyiio_format.attr("is_be").cast<bool>();
             fmt.with_scale = pyiio_format.attr("with_scale").cast<bool>();
             fmt.scale = pyiio_format.attr("scale").cast<double>();
             fmt.repeat = pyiio_format.attr("repeat").cast<unsigned int>();
             return fmt;
           }),
           "Create IIODataFormat from pyiio data format object",
           py::arg("pyiio_format"))
      .def_readwrite("length", &iio_data_format::length)
      .def_readwrite("bits", &iio_data_format::bits)
      .def_readwrite("shift", &iio_data_format::shift)
      .def_readwrite("is_signed", &iio_data_format::is_signed)
      .def_readwrite("is_fully_defined", &iio_data_format::is_fully_defined)
      .def_readwrite("is_be", &iio_data_format::is_be)
      .def_readwrite("with_scale", &iio_data_format::with_scale)
      .def_readwrite("scale", &iio_data_format::scale)
      .def_readwrite("repeat", &iio_data_format::repeat);

  py::class_<iio_channel_info_t>(m, "IIOChannelInfo")
      .def(py::init<>())
      .def_readwrite("name", &iio_channel_info_t::name)
      .def_readwrite("is_output", &iio_channel_info_t::is_output)
      .def_readwrite("index", &iio_channel_info_t::index)
      .def_readwrite("format", &iio_channel_info_t::format);

  py::class_<iio_buffer_info_t>(m, "IIOBufferInfo")
      .def(py::init<>())
      .def_readwrite("samples_count", &iio_buffer_info_t::samples_count)
      .def_readwrite("is_cyclic", &iio_buffer_info_t::is_cyclic)
      .def_readwrite("device_name", &iio_buffer_info_t::device_name)
      .def_readwrite("enabled_channels", &iio_buffer_info_t::enabled_channels)
      .def_property(
          "buffer",
          [](const iio_buffer_info_t& self) -> py::bytes {
            if (self.buffer == nullptr)
              return py::bytes();
            return py::bytes(static_cast<char*>(self.buffer), self.samples_count * sizeof(int16_t));
          },
          [](iio_buffer_info_t& self, py::bytes data) {
            std::string str_data = data;
            self.buffer = malloc(str_data.size());
            std::memcpy(self.buffer, str_data.data(), str_data.size());
          });

  py::class_<IIOAttributeRead,
             PyIIOAttributeRead,
             holoscan::Operator,
             std::shared_ptr<IIOAttributeRead>>(
      m, "IIOAttributeRead", holoscan::doc::IIOAttributeRead::doc_IIOAttributeRead_python)
      .def(py::init<holoscan::Fragment*,
                    const py::args&,
                    std::string,
                    std::string,
                    std::string,
                    std::string,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "ctx"_a,
           "attr_name"_a,
           "dev"_a = ""s,
           "chan"_a = ""s,
           "channel_is_output"_a = false,
           "name"_a = "iio_attribute_read"s,
           holoscan::doc::IIOAttributeRead::doc_IIOAttributeRead_python)
      .def("initialize",
           &IIOAttributeRead::initialize,
           holoscan::doc::IIOAttributeRead::doc_initialize);

  py::class_<IIOAttributeWrite,
             PyIIOAttributeWrite,
             holoscan::Operator,
             std::shared_ptr<IIOAttributeWrite>>(
      m, "IIOAttributeWrite", holoscan::doc::IIOAttributeWrite::doc_IIOAttributeWrite_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::string,
                    std::string,
                    std::string,
                    std::string,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "ctx"_a,
           "attr_name"_a,
           "dev"_a = ""s,
           "chan"_a = ""s,
           "channel_is_output"_a = false,
           "name"_a = "iio_attribute_write"s,
           holoscan::doc::IIOAttributeWrite::doc_IIOAttributeWrite_python)
      .def("initialize",
           &IIOAttributeWrite::initialize,
           holoscan::doc::IIOAttributeWrite::doc_initialize);

  py::class_<IIOBufferWrite, PyIIOBufferWrite, holoscan::Operator, std::shared_ptr<IIOBufferWrite>>(
      m, "IIOBufferWrite", holoscan::doc::IIOBufferWrite::doc_IIOBufferWrite_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::string,
                    std::string,
                    bool,
                    std::vector<std::string>,
                    std::vector<bool>,
                    const std::string&>(),
           "fragment"_a,
           "ctx"_a,
           "dev"_a,
           "is_cyclic"_a,
           "enabled_channel_names"_a,
           "enabled_channel_output"_a,
           "name"_a = "iio_buffer_write"s,
           holoscan::doc::IIOBufferWrite::doc_IIOBufferWrite_python)
      .def(
          "initialize", &IIOBufferWrite::initialize, holoscan::doc::IIOBufferWrite::doc_initialize);

  py::class_<IIOBufferRead, PyIIOBufferRead, holoscan::Operator, std::shared_ptr<IIOBufferRead>>(
      m, "IIOBufferRead", holoscan::doc::IIOBufferRead::doc_IIOBufferRead_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::string,
                    std::string,
                    bool,
                    size_t,
                    std::vector<std::string>,
                    std::vector<bool>,
                    const std::string&>(),
           "fragment"_a,
           "ctx"_a,
           "dev"_a,
           "is_cyclic"_a,
           "samples_count"_a,
           "enabled_channel_names"_a,
           "enabled_channel_input"_a,
           "name"_a = "iio_buffer_read"s,
           holoscan::doc::IIOBufferRead::doc_IIOBufferRead_python)
      .def("initialize", &IIOBufferRead::initialize, holoscan::doc::IIOBufferRead::doc_initialize);

  py::class_<IIOConfigurator,
             PyIIOConfigurator,
             holoscan::Operator,
             std::shared_ptr<IIOConfigurator>>(
      m, "IIOConfigurator", doc::IIOConfigurator::doc_IIOConfigurator_python)
      .def(py::init<Fragment*, const py::args&, std::string, const std::string&>(),
           "fragment"_a,
           "cfg"_a,
           "name"_a = "iio_configurator"s,
           holoscan::doc::IIOConfigurator::doc_IIOConfigurator_python);

  // Helper function to create channel info from IIO channel
  m.def(
      "create_channel_info_from_iio_channel",
      [](py::capsule channel_capsule) -> iio_channel_info_t {
        // Extract the iio_channel pointer from the capsule
        auto* channel = static_cast<const struct iio_channel*>(channel_capsule.get_pointer());
        return create_channel_info_from_iio_channel(channel);
      },
      "Create an IIOChannelInfo structure from an IIO channel pointer",
      py::arg("channel"));

  // Register custom types with the emitter/receiver registry
  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<iio_buffer_info_t>("iio_buffer_info_t"s);
    registry.add_emitter_receiver<std::shared_ptr<iio_buffer_info_t>>(
        "std::shared_ptr<iio_buffer_info_t>"s);
  });
}  // PYBIND11_MODULE
}  // namespace holoscan::ops
