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

#include "gst_src_operator.hpp"

namespace holoscan {

void GstSrcOperator::setup(OperatorSpec& spec) {
  spec.input<gxf::Entity>("input");
  
  spec.param(gst_src_resource_, "gst_src_resource", "GStreamerSource", 
             "GStreamer source resource object");
  spec.param(timeout_ms_, "timeout_ms", "Timeout (ms)", 
             "Timeout in milliseconds for buffer push (0 = try immediately, no waiting)", 
             0UL);
}

void GstSrcOperator::compute(InputContext& input, OutputContext& output, 
                              ExecutionContext& context) {
  static int frame_count = 0;
  frame_count++;
  
  HOLOSCAN_LOG_INFO("GstSrcOperator::compute() - Frame #{} - Receiving entity", frame_count);
  
  // Receive the entity from the input port
  auto entity = input.receive<gxf::Entity>("input").value();
  HOLOSCAN_LOG_INFO("Frame #{} - Entity received, converting to GStreamer buffer", frame_count);

  // Convert entity to GStreamer buffer
  auto buffer = gst_src_resource_.get()->create_buffer_from_entity(entity);
  if (buffer.size() == 0) {
    HOLOSCAN_LOG_ERROR("Frame #{} - Failed to convert entity to buffer", frame_count);
    return;
  }

  HOLOSCAN_LOG_INFO("Frame #{} - Buffer created, size: {} bytes", frame_count, buffer.size());

  // Push buffer into the GStreamer pipeline
  // Convert uint64_t milliseconds to std::chrono::milliseconds
  auto timeout = std::chrono::milliseconds(timeout_ms_.get());
  HOLOSCAN_LOG_INFO("Frame #{} - Pushing buffer to GstSrcResource (timeout: {}ms)", 
                    frame_count, timeout_ms_.get());
  
  if (!gst_src_resource_.get()->push_buffer(std::move(buffer), timeout)) {
    HOLOSCAN_LOG_ERROR("Frame #{} - Failed to push buffer to GstSrcResource (timeout or error)", 
                       frame_count);
    return;
  }
  
  HOLOSCAN_LOG_INFO("Frame #{} - Buffer successfully pushed to GstSrcResource", frame_count);
}

void GstSrcOperator::stop() {
  HOLOSCAN_LOG_INFO("GstSrcOperator::stop() - Operator stopping, sending EOS to GStreamer pipeline");
  
  // Send EOS to signal end of stream
  // The application should wait for the EOS message on the pipeline bus
  gst_src_resource_.get()->send_eos();
  
  HOLOSCAN_LOG_INFO("GstSrcOperator::stop() - EOS sent");
}

}  // namespace holoscan

