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

#include "../video_decoder_context.hpp"
#include "./video_decoder_context_pydoc.hpp"

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

// PyVideoDecoderContext trampoline class: provides override for virtual function is_available

class PyVideoDecoderContext : public VideoDecoderContext {
 public:
  /* Inherit the constructors */
  using VideoDecoderContext::VideoDecoderContext;

  // Define a constructor that fully initializes the object.
  PyVideoDecoderContext(Fragment* fragment,
                        std::shared_ptr<holoscan::AsynchronousCondition>& async_scheduling_term,
                        const int32_t device_id = 0,
                        const std::string& name = "video_decoder_context")
      : VideoDecoderContext(ArgList{Arg{"async_scheduling_term", async_scheduling_term},
                                    Arg{"device_id", device_id}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_video_decoder_context, m) {
  m.doc() = R"pbdoc(
        VideoDecoderContext Python Bindings
        -----------------------------------
        .. currentmodule:: _video_decoder_context
    )pbdoc";

  py::class_<VideoDecoderContext,
             PyVideoDecoderContext,
             gxf::GXFResource,
             std::shared_ptr<VideoDecoderContext>>(
      m, "VideoDecoderContext", doc::VideoDecoderContext::doc_VideoDecoderContext)
      .def(py::init<Fragment*,
                    std::shared_ptr<holoscan::AsynchronousCondition>&,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "async_scheduling_term"_a,
           "device_id"_a = 0,
           "name"_a = "video_decoder_context"s,
           doc::VideoDecoderContext::doc_VideoDecoderContext);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
