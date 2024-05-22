/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "../video_write_bitstream.hpp"
#include "./video_write_bitstream_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include "../../operator_util.hpp"
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

class PyVideoWriteBitstreamOp : public VideoWriteBitstreamOp {
 public:
  /* Inherit the constructors */
  using VideoWriteBitstreamOp::VideoWriteBitstreamOp;

  // Define a constructor that fully initializes the object.
  PyVideoWriteBitstreamOp(Fragment* fragment, const py::args& args,
                         const std::string& output_video_path,
                         const int frame_width, const int frame_height,
                         const int inbuf_storage_type,
                         const std::string& input_crc_file_path = std::string(),
                         const std::string& name = "video_read_bit_stream")
      : VideoWriteBitstreamOp(ArgList{Arg{"output_video_path", output_video_path},
                                     Arg{"frame_width", frame_width},
                                     Arg{"frame_height", frame_height},
                                     Arg{"inbuf_storage_type", inbuf_storage_type},
                                     Arg{"input_crc_file_path", input_crc_file_path}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_video_write_bitstream, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _video_write_bitstream
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

  py::class_<VideoWriteBitstreamOp,
             PyVideoWriteBitstreamOp,
             GXFOperator,
             std::shared_ptr<VideoWriteBitstreamOp>>(
      m, "VideoWriteBitstreamOp", doc::VideoWriteBitstreamOp::doc_VideoWriteBitstreamOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    int,
                    int,
                    int,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "output_video_path"_a,
           "frame_width"_a,
           "frame_height"_a,
           "inbuf_storage_type"_a,
           "input_crc_file_path"_a = std::string(),
           "name"_a = "video_write_bitstream"s,
           doc::VideoWriteBitstreamOp::doc_VideoWriteBitstreamOp)
      .def_property_readonly("gxf_typename",
                             &VideoWriteBitstreamOp::gxf_typename,
                             doc::VideoWriteBitstreamOp::doc_gxf_typename)
      .def("initialize",
           &VideoWriteBitstreamOp::initialize,
           doc::VideoWriteBitstreamOp::doc_initialize)
      .def("setup", &VideoWriteBitstreamOp::setup, "spec"_a, doc::VideoWriteBitstreamOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
