/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Wayland Technologies. All rights reserved.
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

#include "point_cloud_filter_op.hpp"

#include <stdexcept>
#include <string>

#include "gxf/std/allocator.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/logger/logger.hpp"
#include "point_cloud_filter.cuh"

namespace holoscan::ops {

namespace {

constexpr const char* kStructuredTensorName = "structured_points";

inline void check_cuda(cudaError_t code, const char* message) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(code));
  }
}

}  // namespace

void PointCloudFilterOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("in_disparity");
  spec.input<std::shared_ptr<std::vector<float>>>("in_q_matrix");
  spec.output<holoscan::gxf::Entity>("out_structured_points");

  spec.param(structured_allocator_,
             "structured_allocator",
             "StructuredAllocator",
             "Structured light output allocator");
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CudaStreamPool",
             "CUDA stream pool used for async GPU operations");
}

void PointCloudFilterOp::start() {
  for (auto& entity : structured_output_entities_) { entity.reset(); }
  structured_output_entity_index_ = 0;
  structured_output_point_count_ = 0;
  first_cloud_logged_ = false;
}

void PointCloudFilterOp::stop() {
  holoscan::Operator::stop();
}

void PointCloudFilterOp::ensure_structured_output_entities(
    const holoscan::ExecutionContext& context, size_t point_count) {
  if (structured_output_point_count_ != point_count) {
    for (auto& entity : structured_output_entities_) { entity.reset(); }
    structured_output_point_count_ = point_count;
    structured_output_entity_index_ = 0;
  }

  if (structured_output_entities_[structured_output_entity_index_]) {
    return;
  }

  const nvidia::gxf::Shape shape{static_cast<int32_t>(point_count), 3};
  auto alloc = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), structured_allocator_.get()->gxf_cid());
  auto msg = nvidia::gxf::Entity::New(context.context());
  auto tensor = msg.value().add<nvidia::gxf::Tensor>(kStructuredTensorName);
  tensor.value()->reshape<float>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc.value());
  structured_output_entities_[structured_output_entity_index_].emplace(std::move(msg.value()));
}

void PointCloudFilterOp::compute(holoscan::InputContext& op_input,
                                 holoscan::OutputContext& op_output,
                                 holoscan::ExecutionContext& context) {
  auto q_mat_msg = op_input.receive<std::shared_ptr<std::vector<float>>>("in_q_matrix");
  auto q_mat = q_mat_msg.value();
  if (!q_mat || q_mat->size() != 16) {
    throw std::runtime_error("PointCloudFilterOp: invalid Q matrix input");
  }
  const float* h_Q = q_mat->data();

  auto in_entity_value = op_input.receive<holoscan::gxf::Entity>("in_disparity");
  auto in_entity = nvidia::gxf::Entity(in_entity_value.value());
  auto disp_tensor = in_entity.get<nvidia::gxf::Tensor>("disparity_map");
  if (!disp_tensor) {
    throw std::runtime_error("PointCloudFilterOp: failed to fetch disparity_map tensor");
  }

  const int32_t height = disp_tensor.value()->shape().dimension(0);
  const int32_t width = disp_tensor.value()->shape().dimension(1);
  const int16_t* d_disp_map = reinterpret_cast<const int16_t*>(disp_tensor.value()->pointer());
  const size_t step = width * sizeof(int16_t);

  cudaStream_t cuda_stream = cudaStreamDefault;
  op_input.receive_cuda_stream("in_disparity", true, cuda_stream);
  if (cuda_stream == cudaStreamDefault) {
    auto maybe_stream = context.allocate_cuda_stream("point_cloud_filter_stream");
    if (maybe_stream) {
      cuda_stream = maybe_stream.value();
    }
  }

  int total_points = width * height;

  ensure_structured_output_entities(context, total_points);
  auto& out_entity_opt = structured_output_entities_[structured_output_entity_index_].value();
  auto out_gxf_entity = nvidia::gxf::Entity(out_entity_opt);
  auto tensor = out_gxf_entity.get<nvidia::gxf::Tensor>(kStructuredTensorName);
  float* d_out_points = reinterpret_cast<float*>(tensor.value()->pointer());

  launch_point_cloud_filter(d_disp_map,
                            width,
                            height,
                            step,
                            h_Q,
                            0.0f,
                            5000.0f,
                            5000.0f,
                            5000.0f,
                            d_out_points,
                            cuda_stream);
  check_cuda(cudaGetLastError(), "Failed to launch point cloud filter kernel");

  if (!first_cloud_logged_) {
    first_cloud_logged_ = true;
    HOLOSCAN_LOG_INFO(
        "PointCloudFilterOp: first structured cloud dispatched asynchronously with {} max points",
        total_points);
  }

  op_output.set_cuda_stream(cuda_stream, "out_structured_points");
  holoscan::gxf::Entity out(out_entity_opt);
  op_output.emit(out, "out_structured_points");
  structured_output_entities_[structured_output_entity_index_].reset();
  structured_output_entity_index_ =
      (structured_output_entity_index_ + 1) % structured_output_entities_.size();
}

}  // namespace holoscan::ops
