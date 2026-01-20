/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>

#include <gxf/std/tensor.hpp>

#include "tool_tracking_postprocessor.cuh"

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
  auto& out_tensor = spec.output<gxf::Entity>("out");

  spec.param(in_, "in", "Input", "Input port.", &in_tensor);
  spec.param(out_, "out", "Output", "Output port.", &out_tensor);

  spec.param(
      min_prob_, "min_prob", "Minimum probability", "Minimum probability.", DEFAULT_MIN_PROB);

  spec.param(overlay_img_colors_,
             "overlay_img_colors",
             "Overlay Image Layer Colors",
             "Color of the image overlays, a list of RGB values with components between 0 and 1",
             DEFAULT_COLORS);

  spec.param(device_allocator_, "device_allocator", "Allocator", "Output Allocator");

  cuda_stream_handler_.define_params(spec);
}

void ToolTrackingPostprocessorOp::stop() {
  if (dev_colors_) {
    CUDA_TRY(cudaFree(dev_colors_));
    dev_colors_ = nullptr;
  }
}

void ToolTrackingPostprocessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                          ExecutionContext& context) {
  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto in_message = op_input.receive<gxf::Entity>("in").value();

  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result =
      cuda_stream_handler_.from_message(context.context(), in_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  auto maybe_tensor = in_message.get<Tensor>("probs");
  if (!maybe_tensor) { throw std::runtime_error("Tensor 'probs' not found in message."); }
  auto probs_tensor = maybe_tensor;

  maybe_tensor = in_message.get<Tensor>("scaled_coords");
  if (!maybe_tensor) { throw std::runtime_error("Tensor 'scaled_coords' not found in message."); }
  auto scaled_coords_tensor = maybe_tensor;

  maybe_tensor = in_message.get<Tensor>("binary_masks");
  if (!maybe_tensor) { throw std::runtime_error("Tensor 'binary_masks' not found in message."); }
  auto binary_masks_tensor = maybe_tensor;

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto device_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), device_allocator_.get()->gxf_cid());

  // Create a new message (nvidia::nvidia::gxf::Entity) for the output
  auto out_message = nvidia::gxf::Entity::New(context.context());

  // Create a new tensor for the scaled coords
  auto out_coords_tensor = out_message.value().add<nvidia::gxf::Tensor>("scaled_coords");
  if (!out_coords_tensor) {
    throw std::runtime_error("Failed to allocate output tensor 'scaled_coords'");
  }

  const nvidia::gxf::Shape coords_shape{int32_t(probs_tensor->size()), 3};
  out_coords_tensor.value()->reshape<float>(
      coords_shape, nvidia::gxf::MemoryStorageType::kDevice, device_allocator.value());
  if (!out_coords_tensor.value()->pointer()) {
    throw std::runtime_error("Failed to allocate output tensor buffer for tensor 'scaled_coords'.");
  }

  // Create a new tensor for the mask
  auto out_mask_tensor = out_message.value().add<nvidia::gxf::Tensor>("mask");
  if (!out_mask_tensor) { throw std::runtime_error("Failed to allocate output tensor 'mask'"); }

  const nvidia::gxf::Shape mask_shape{static_cast<int>(binary_masks_tensor->shape()[2]),
                                      static_cast<int>(binary_masks_tensor->shape()[3]),
                                      4};
  out_mask_tensor.value()->reshape<float>(
      mask_shape, nvidia::gxf::MemoryStorageType::kDevice, device_allocator.value());
  if (!out_mask_tensor.value()->pointer()) {
    throw std::runtime_error("Failed to allocate output tensor buffer for tensor 'mask'.");
  }

  const cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

  if (num_colors_ != probs_tensor->size()) {
    num_colors_ = probs_tensor->size();
    if (dev_colors_) {
      CUDA_TRY(cudaFree(dev_colors_));
      dev_colors_ = nullptr;
    }
  }

  if (!dev_colors_) {
    // copy colors to CUDA device memory, this is needed by the postprocessing kernel
    CUDA_TRY(cudaMalloc(&dev_colors_, num_colors_ * sizeof(float3)));

    // build a vector with the colors, if more colors are required than specified, repeat the
    // last color
    std::vector<float3> colors;
    for (auto index = 0; index < num_colors_; ++index) {
      const auto& img_color =
          overlay_img_colors_.get()[std::min(index, int(overlay_img_colors_.get().size()))];
      colors.push_back(make_float3(img_color[0], img_color[1], img_color[2]));
    }

    CUDA_TRY(cudaMemcpyAsync(dev_colors_,
                             colors.data(),
                             num_colors_ * sizeof(float3),
                             cudaMemcpyHostToDevice,
                             cuda_stream));
  }

  // filter coordinates based on probability and create a colored mask from the binary mask
  cuda_postprocess(probs_tensor->size(),
                   min_prob_,
                   reinterpret_cast<const float*>(probs_tensor->data()),
                   reinterpret_cast<const float2*>(scaled_coords_tensor->data()),
                   reinterpret_cast<float3*>(out_coords_tensor.value()->pointer()),
                   mask_shape.dimension(0),
                   mask_shape.dimension(1),
                   reinterpret_cast<const float3*>(dev_colors_),
                   reinterpret_cast<const float*>(binary_masks_tensor->data()),
                   reinterpret_cast<float4*>(out_mask_tensor.value()->pointer()),
                   cuda_stream);

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.to_message(out_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
  }

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result, "out");
}

}  // namespace holoscan::ops
