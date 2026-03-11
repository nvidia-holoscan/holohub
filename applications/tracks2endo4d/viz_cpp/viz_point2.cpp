/*
* SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "viz.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>
#include <holoviz/imgui/imgui.h>

#include <gxf/std/tensor.hpp>

#include <cuda_runtime.h>

#define CUDA_TRY(stmt)                                                                            \
  {                                                                                               \
    cudaError_t cuda_error = stmt;                                                                \
    if (cuda_error != cudaSuccess) {                                                              \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_error));     \
    }                                                                                             \
  }

namespace viz = holoscan::viz;



namespace holoscan::ops {

void VizOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("receivers");
  spec.param(in_, "receivers", "Input", "Input port.", &in_tensor);

  spec.param(device_allocator_, "device_allocator", "Allocator", "Output Allocator");

  spec.param(width_, "width", "Width", "Window width", 1280u);
  spec.param(height_, "height", "Height", "Window height", 720u);
  spec.param(window_title_, "window_title", "Window Title", "Window title", std::string("Holoviz"));
  spec.param(headless_, "headless", "Headless", "Headless mode", false);

  cuda_stream_handler_.define_params(spec);
}

void VizOp::start() {
  viz::Init(width_.get(), height_.get(), window_title_.get().c_str(),
            headless_.get() ? viz::InitFlags::HEADLESS : viz::InitFlags::NONE);
}

void VizOp::stop() {
  viz::Shutdown();
}

void VizOp::compute(InputContext& op_input, OutputContext& op_output,
                                          ExecutionContext& context) {
  auto maybe_tensormap = op_input.receive<TensorMap>("receivers");
  if (!maybe_tensormap) {
    std::string err_msg =
        fmt::format("Operator '{}' failed to receive input message on port 'receivers': {}",
                    name_,
                    maybe_tensormap.error().what());
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  auto& tensormap = maybe_tensormap.value();
  auto maybe_input_tensor = tensormap.find("point_coords");
  if (maybe_input_tensor == tensormap.end()) {
    std::string err_msg =
        fmt::format("Operator '{}' failed to find input tensor on port 'point_coords'", name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  auto input_tensor = maybe_input_tensor->second;
  const float* in_tensor_data = static_cast<float*>(input_tensor->data());
  auto num_points = input_tensor->shape()[0];

  if (!input_tensor->is_contiguous()) {
    throw std::runtime_error("Input tensor must have row-major memory layout.");
  }

  // float first_element = 0.0f;
  // cudaError_t err = cudaMemcpy(&first_element, input_tensor->data(), sizeof(float), cudaMemcpyDeviceToHost);
  // if (err != cudaSuccess) {
  //   HOLOSCAN_LOG_ERROR("Failed to copy tensor data from device to host: {}", cudaGetErrorString(err));
  // } else {
  //   std::cout << "In tensor data: " << first_element << std::endl;
  // }

  viz::Begin();
  
  viz::BeginGeometryLayer();

  const size_t stride_bytes = 3 * sizeof(float);
  CUdeviceptr dev_ptr_base = reinterpret_cast<CUdeviceptr>(in_tensor_data);

  for (size_t i = 0; i < num_points; ++i) {
      // Having alpha depend on the z-value involves moving the points to the host. Avoiding this for now.
      viz::Color(1.0f, 0.0f, 0.0f, 1.0f);
      viz::PointSize(5.F);
      viz::PrimitiveCudaDevice(viz::PrimitiveTopology::POINT_LIST, 1, 2, dev_ptr_base + i * stride_bytes);
  }
  viz::EndLayer();

  viz::End();

}
} // namespace holoscan::ops
