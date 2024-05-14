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

#include "tool_tracking_postprocessor.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include "gxf/std/tensor.hpp"

using holoscan::ops::tool_tracking_postprocessor::cuda_postprocess;

#define CUDA_TRY(stmt)                                                                   \
  ({                                                                                     \
    cudaError_t _holoscan_cuda_err = stmt;                                               \
    if (cudaSuccess != _holoscan_cuda_err) {                                             \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).", \
                    #stmt,                                                               \
                    __LINE__,                                                            \
                    __FILE__,                                                            \
                    cudaGetErrorString(_holoscan_cuda_err),                              \
                    _holoscan_cuda_err);                                                 \
    }                                                                                    \
    _holoscan_cuda_err;                                                                  \
  })

namespace holoscan::ops {

void ToolTrackingPostprocessorOp::setup(OperatorSpec& spec) {
  constexpr float DEFAULT_MIN_PROB = 0.5f;
  // 12 qualitative classes color scheme from colorbrewer2
  static const std::vector<std::vector<float>> DEFAULT_COLORS = {{0.12f, 0.47f, 0.71f},
                                                                 {0.20f, 0.63f, 0.17f},
                                                                 {0.89f, 0.10f, 0.11f},
                                                                 {1.00f, 0.50f, 0.00f},
                                                                 {0.42f, 0.24f, 0.60f},
                                                                 {0.69f, 0.35f, 0.16f},
                                                                 {0.65f, 0.81f, 0.89f},
                                                                 {0.70f, 0.87f, 0.54f},
                                                                 {0.98f, 0.60f, 0.60f},
                                                                 {0.99f, 0.75f, 0.44f},
                                                                 {0.79f, 0.70f, 0.84f},
                                                                 {1.00f, 1.00f, 0.60f}};

  auto& in_tensor = spec.input<gxf::Entity>("in");
  // Because coords is on host and mask is on device, emit them on separate ports for
  // compatibility with use of this operator in distributed applications.
  auto& out_coords = spec.output<gxf::Entity>("out_coords");
  auto& out_mask = spec.output<gxf::Entity>("out_mask");

  spec.param(in_, "in", "Input", "Input port.", &in_tensor);
  spec.param(
      out_coords_, "out_coords", "Output", "Output port for coordinates (on host).", &out_coords);
  spec.param(out_mask_, "out_mask", "Output", "Output port for mask (on device).", &out_mask);

  spec.param(
      min_prob_, "min_prob", "Minimum probability", "Minimum probability.", DEFAULT_MIN_PROB);

  spec.param(overlay_img_colors_,
             "overlay_img_colors",
             "Overlay Image Layer Colors",
             "Color of the image overlays, a list of RGB values with components between 0 and 1",
             DEFAULT_COLORS);

  spec.param(host_allocator_, "host_allocator", "Allocator", "Output Allocator");
  spec.param(device_allocator_, "device_allocator", "Allocator", "Output Allocator");

  cuda_stream_handler_.define_params(spec);
}

void ToolTrackingPostprocessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                          ExecutionContext& context) {
  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto in_message = op_input.receive<gxf::Entity>("in").value();
  auto maybe_tensor = in_message.get<Tensor>("probs");
  if (!maybe_tensor) { throw std::runtime_error("Tensor 'probs' not found in message."); }
  auto probs_tensor = maybe_tensor;

  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result =
      cuda_stream_handler_.fromMessage(context.context(), in_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  std::vector<float> probs(probs_tensor->size());
  CUDA_TRY(cudaMemcpyAsync(probs.data(),
                           probs_tensor->data(),
                           probs_tensor->nbytes(),
                           cudaMemcpyDeviceToHost,
                           cuda_stream_handler_.getCudaStream(context.context())));

  maybe_tensor = in_message.get<Tensor>("scaled_coords");
  if (!maybe_tensor) { throw std::runtime_error("Tensor 'scaled_coords' not found in message."); }
  auto scaled_coords_tensor = maybe_tensor;

  std::vector<float> scaled_coords(scaled_coords_tensor->size());
  CUDA_TRY(cudaMemcpyAsync(scaled_coords.data(),
                           scaled_coords_tensor->data(),
                           scaled_coords_tensor->nbytes(),
                           cudaMemcpyDeviceToHost,
                           cuda_stream_handler_.getCudaStream(context.context())));

  maybe_tensor = in_message.get<Tensor>("binary_masks");
  if (!maybe_tensor) { throw std::runtime_error("Tensor 'binary_masks' not found in message."); }
  auto binary_masks_tensor = maybe_tensor;

  // Create a new message (nvidia::nvidia::gxf::Entity) for host tensor(s)
  auto out_message_host = nvidia::gxf::Entity::New(context.context());

  // filter coordinates based on probability
  std::vector<uint32_t> visible_classes;
  {
    // wait for the CUDA memory copy to finish
    CUDA_TRY(cudaStreamSynchronize(cuda_stream_handler_.getCudaStream(context.context())));

    std::vector<float> filtered_scaled_coords;
    for (size_t index = 0; index < probs.size(); ++index) {
      if (probs[index] > min_prob_) {
        filtered_scaled_coords.push_back(scaled_coords[index * 2]);
        filtered_scaled_coords.push_back(scaled_coords[index * 2 + 1]);
        visible_classes.push_back(index);
      } else {
        filtered_scaled_coords.push_back(-1.f);
        filtered_scaled_coords.push_back(-1.f);
      }
    }

    auto out_coords_tensor = out_message_host.value().add<nvidia::gxf::Tensor>("scaled_coords");
    if (!out_coords_tensor) {
      throw std::runtime_error("Failed to allocate output tensor 'scaled_coords'");
    }

    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto host_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), host_allocator_.get()->gxf_cid());

    const nvidia::gxf::Shape output_shape{1, int32_t(filtered_scaled_coords.size() / 2), 2};
    out_coords_tensor.value()->reshape<float>(
        output_shape, nvidia::gxf::MemoryStorageType::kHost, host_allocator.value());
    if (!out_coords_tensor.value()->pointer()) {
      throw std::runtime_error(
          "Failed to allocate output tensor buffer for tensor 'scaled_coords'.");
    }
    memcpy(out_coords_tensor.value()->data<float>().value(),
           filtered_scaled_coords.data(),
           filtered_scaled_coords.size() * sizeof(float));
  }

  // Create a new message (nvidia::nvidia::gxf::Entity) for device tensor(s)
  auto out_message_device = nvidia::gxf::Entity::New(context.context());

  // filter binary mask
  {
    auto out_mask_tensor = out_message_device.value().add<nvidia::gxf::Tensor>("mask");
    if (!out_mask_tensor) { throw std::runtime_error("Failed to allocate output tensor 'mask'"); }

    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto device_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), device_allocator_.get()->gxf_cid());

    const nvidia::gxf::Shape output_shape{static_cast<int>(binary_masks_tensor->shape()[2]),
                                          static_cast<int>(binary_masks_tensor->shape()[3]),
                                          4};
    out_mask_tensor.value()->reshape<float>(
        output_shape, nvidia::gxf::MemoryStorageType::kDevice, device_allocator.value());
    if (!out_mask_tensor.value()->pointer()) {
      throw std::runtime_error("Failed to allocate output tensor buffer for tensor 'mask'.");
    }

    float* const out_data = out_mask_tensor.value()->data<float>().value();
    const size_t layer_size = output_shape.dimension(0) * output_shape.dimension(1);
    bool first = true;
    for (auto& index : visible_classes) {
      const auto& img_color =
          overlay_img_colors_.get()[std::min(index, uint32_t(overlay_img_colors_.get().size()))];
      const std::array<float, 3> color{{img_color[0], img_color[1], img_color[2]}};
      cuda_postprocess(output_shape.dimension(0),
                       output_shape.dimension(1),
                       color,
                       first,
                       static_cast<float*>(binary_masks_tensor->data()) + index * layer_size,
                       reinterpret_cast<float4*>(out_data),
                       cuda_stream_handler_.getCudaStream(context.context()));
      first = false;
    }
  }

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.toMessage(out_message_device);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
  }

  auto result_host = gxf::Entity(std::move(out_message_host.value()));
  auto result_device = gxf::Entity(std::move(out_message_device.value()));
  op_output.emit(result_host, "out_coords");
  op_output.emit(result_device, "out_mask");
}

}  // namespace holoscan::ops
