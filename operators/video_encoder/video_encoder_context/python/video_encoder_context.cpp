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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include "../video_encoder_context.hpp"
#include "./video_encoder_context_pydoc.hpp"

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

// PyVideoEncoderContext trampoline class: provides override for virtual function is_available

class PyVideoEncoderContext : public VideoEncoderContext {
 public:
  /* Inherit the constructors */
  using VideoEncoderContext::VideoEncoderContext;

  // Define a constructor that fully initializes the object.
  PyVideoEncoderContext(Fragment* fragment,
                        std::shared_ptr<holoscan::AsynchronousCondition>& scheduling_term,
                        const int32_t device_id = 0,
                        const std::string& name = "video_encoder_context")
      : VideoEncoderContext(ArgList{Arg{"scheduling_term", scheduling_term},
                                    Arg{"device_id", device_id}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_video_encoder_context, m) {
  m.doc() = R"pbdoc(
        VideoEncoderContext Python Bindings
        -----------------------------------
        .. currentmodule:: _video_encoder_context
    )pbdoc";

  py::class_<VideoEncoderContext,
             PyVideoEncoderContext,
             gxf::GXFResource,
             std::shared_ptr<VideoEncoderContext>>(
      m, "VideoEncoderContext", doc::VideoEncoderContext::doc_VideoEncoderContext)
      .def(py::init<Fragment*,
                    std::shared_ptr<holoscan::AsynchronousCondition>&,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "scheduling_term"_a,
           "device_id"_a = 0,
           "name"_a = "video_encoder_context"s,
           doc::VideoEncoderContext::doc_VideoEncoderContext);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
