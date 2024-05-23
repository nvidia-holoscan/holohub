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

#include "../video_read_bitstream.hpp"
#include "./video_read_bitstream_pydoc.hpp"

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

class PyVideoReadBitstreamOp : public VideoReadBitstreamOp {
 public:
  /* Inherit the constructors */
  using VideoReadBitstreamOp::VideoReadBitstreamOp;

  // Define a constructor that fully initializes the object.
  PyVideoReadBitstreamOp(Fragment* fragment, const py::args& args,
                         const std::string& input_file_path, const std::shared_ptr<Allocator> pool,
                         const int32_t outbuf_storage_type = 0, const int32_t aud_nal_present = 0,
                         const std::string& name = "video_read_bitstream")
      : VideoReadBitstreamOp(ArgList{Arg{"input_file_path", input_file_path},
                                     Arg{"pool", pool},
                                     Arg{"outbuf_storage_type", outbuf_storage_type},
                                     Arg{"aud_nal_present", aud_nal_present}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_video_read_bitstream, m) {
  m.doc() = R"pbdoc(
        VideoReadBitstreamOp Python Bindings
        ------------------------------------
        .. currentmodule:: _video_read_bitstream
    )pbdoc";

  py::class_<VideoReadBitstreamOp,
             PyVideoReadBitstreamOp,
             GXFOperator,
             std::shared_ptr<VideoReadBitstreamOp>>(
      m, "VideoReadBitstreamOp", doc::VideoReadBitstreamOp::doc_VideoReadBitstreamOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    std::shared_ptr<Allocator>,
                    int32_t,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "input_file_path"_a,
           "pool"_a,
           "outbuf_storage_type"_a,
           "aud_nal_present"_a,
           "name"_a = "video_read_bitstream"s,
           doc::VideoReadBitstreamOp::doc_VideoReadBitstreamOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
