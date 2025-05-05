/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../volume_renderer.hpp"
#include "./volume_renderer_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
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
class PyVolumeRendererOp : public VolumeRendererOp {
 public:
  /* Inherit the constructors */
  using VolumeRendererOp::VolumeRendererOp;

  // Define a constructor that fully initializes the object.
  PyVolumeRendererOp(Fragment* fragment, const py::args& args, const std::string& config_file,
                     const std::string& write_config_file,
                     const std::shared_ptr<Allocator>& allocator, uint32_t alloc_width,
                     uint32_t alloc_height, std::optional<float> density_min,
                     std::optional<float> density_max,
                     const std::shared_ptr<holoscan::CudaStreamPool>& cuda_stream_pool,
                     const std::string& name = "volume_renderer")
      : VolumeRendererOp(ArgList{Arg{"config_file", config_file},
                                 Arg{"write_config_file", write_config_file},
                                 Arg{"allocator", allocator},
                                 Arg{"alloc_width", alloc_width},
                                 Arg{"alloc_height", alloc_height}}) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    if (density_min.has_value()) { this->add_arg(Arg{"density_min", density_min.value()}); }
    if (density_max.has_value()) { this->add_arg(Arg{"density_max", density_max.value()}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_volume_renderer, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _volume_renderer
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

  py::class_<VolumeRendererOp, PyVolumeRendererOp, Operator, std::shared_ptr<VolumeRendererOp>>(
      m, "VolumeRendererOp", doc::VolumeRendererOp::doc_VolumeRendererOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::shared_ptr<Allocator>&,
                    uint32_t,
                    uint32_t,
                    std::optional<float>,
                    std::optional<float>,
                    const std::shared_ptr<holoscan::CudaStreamPool>&,
                    const std::string&>(),
           "fragment"_a,
           "config_file"_a = "",
           "write_config_file"_a = "",
           "allocator"_a = std::shared_ptr<Allocator>(),
           "alloc_width"_a = 1024u,
           "alloc_height"_a = 768u,
           "density_min"_a = py::none(),
           "density_max"_a = py::none(),
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "volume_renderer"s,
           doc::VolumeRendererOp::doc_VolumeRendererOp_python)
      .def("setup", &VolumeRendererOp::setup, "spec"_a, doc::VolumeRendererOp::doc_setup);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
