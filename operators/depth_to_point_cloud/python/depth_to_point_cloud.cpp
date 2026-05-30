/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../../operator_util.hpp"

#include <depth_to_point_cloud.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class providing a Pythonic kwarg-based constructor matching the C++ defaults. */
class PyDepthToPointCloudOp : public DepthToPointCloudOp {
 public:
  using DepthToPointCloudOp::DepthToPointCloudOp;

  PyDepthToPointCloudOp(Fragment* fragment, const py::args& args,
                          std::shared_ptr<Allocator> allocator, float fx = 0.0f, float fy = 0.0f,
                          float cx = 0.0f, float cy = 0.0f, float depth_scale = 0.001f,
                          float depth_min = 0.0f, float depth_max = 100.0f,
                          float invalid_value = std::numeric_limits<float>::quiet_NaN(),
                          const std::string& depth_tensor_name = "",
                          const std::string& color_tensor_name = "",
                          const std::string& output_tensor_name = "point_cloud",
                          const std::string& output_color_tensor_name = "colors",
                          std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
                          const std::string& name = "depth_to_point_cloud")
      : DepthToPointCloudOp(ArgList{Arg{"allocator", allocator},
                                      Arg{"fx", fx},
                                      Arg{"fy", fy},
                                      Arg{"cx", cx},
                                      Arg{"cy", cy},
                                      Arg{"depth_scale", depth_scale},
                                      Arg{"depth_min", depth_min},
                                      Arg{"depth_max", depth_max},
                                      Arg{"invalid_value", invalid_value},
                                      Arg{"depth_tensor_name", depth_tensor_name},
                                      Arg{"color_tensor_name", color_tensor_name},
                                      Arg{"output_tensor_name", output_tensor_name},
                                      Arg{"output_color_tensor_name", output_color_tensor_name}}) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_depth_to_point_cloud, m) {
  m.doc() = R"pbdoc(
        DepthToPointCloudOp Python Bindings
        -------------------------------------
        .. currentmodule:: _depth_to_point_cloud
    )pbdoc";

  py::class_<DepthToPointCloudOp,
             PyDepthToPointCloudOp,
             Operator,
             std::shared_ptr<DepthToPointCloudOp>>(
      m, "DepthToPointCloudOp",
      "Deproject an organized depth image into an organized point cloud on the GPU.")
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<Allocator>,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "fx"_a = 0.0f,
           "fy"_a = 0.0f,
           "cx"_a = 0.0f,
           "cy"_a = 0.0f,
           "depth_scale"_a = 0.001f,
           "depth_min"_a = 0.0f,
           "depth_max"_a = 100.0f,
           "invalid_value"_a = std::numeric_limits<float>::quiet_NaN(),
           "depth_tensor_name"_a = ""s,
           "color_tensor_name"_a = ""s,
           "output_tensor_name"_a = "point_cloud"s,
           "output_color_tensor_name"_a = "colors"s,
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "depth_to_point_cloud"s)
      .def("setup", &DepthToPointCloudOp::setup, "spec"_a);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
