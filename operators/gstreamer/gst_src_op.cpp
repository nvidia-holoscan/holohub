/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gst_src_op.hpp"
#include <holoscan/core/domain/tensor_map.hpp>

namespace holoscan {

void GstSrcOp::setup(OperatorSpec& spec) {
  spec.input<TensorMap>("input");

  spec.param(
      gst_src_resource_, "gst_src_resource", "GStreamerSource", "GStreamer source resource object");
}

void GstSrcOp::compute(InputContext& input, OutputContext& output,
                       ExecutionContext& context) {
  static int frame_count = 0;
  frame_count++;

  HOLOSCAN_LOG_INFO("GstSrcOp::compute() - Frame #{} - Receiving tensor map", frame_count);

  // Receive the tensor map from the input port
  auto tensor_map = input.receive<TensorMap>("input").value();
  HOLOSCAN_LOG_INFO("Frame #{} - TensorMap received, converting to GStreamer buffer", frame_count);

  // Convert tensor map to GStreamer buffer
  auto buffer = gst_src_resource_.get()->create_buffer_from_tensor_map(tensor_map);
  if (buffer.get_size() == 0) {
    HOLOSCAN_LOG_ERROR("Frame #{} - Failed to convert entity to buffer", frame_count);
    return;
  }

  HOLOSCAN_LOG_INFO("Frame #{} - Buffer created, size: {} bytes", frame_count, buffer.get_size());

  // Push buffer into the GStreamer pipeline
  HOLOSCAN_LOG_INFO("Frame #{} - Pushing buffer to GstSrcResource", frame_count);

  if (!gst_src_resource_.get()->push_buffer(std::move(buffer))) {
    HOLOSCAN_LOG_ERROR("Frame #{} - Failed to push buffer to GstSrcResource", frame_count);
    return;
  }

  HOLOSCAN_LOG_INFO("Frame #{} - Buffer successfully pushed to GstSrcResource", frame_count);
}

void GstSrcOp::stop() {
  HOLOSCAN_LOG_INFO(
      "GstSrcOp::stop() - Operator stopping, sending EOS to GStreamer pipeline");

  // Send EOS to signal end of stream
  // The application should wait for the EOS message on the pipeline bus
  gst_src_resource_.get()->send_eos();

  HOLOSCAN_LOG_INFO("GstSrcOp::stop() - EOS sent");
}

}  // namespace holoscan
