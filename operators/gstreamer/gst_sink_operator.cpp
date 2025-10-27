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

#include "gst_sink_operator.hpp"

#include <chrono>
#include <future>

#include "gst/buffer.hpp"

namespace holoscan {

void GstSinkOperator::setup(OperatorSpec& spec) {
  spec.output<TensorMap>("output");

  // Add parameters to the operator spec
  spec.param(gst_sink_resource_, "gst_sink_resource", "GStreamerSink", 
             "GStreamer sink resource object");
  spec.param(timeout_ms_, "timeout_ms", "Timeout (ms)", 
             "Timeout in milliseconds for waiting for buffer from GStreamer pipeline",
             1000UL);
}

void GstSinkOperator::compute(InputContext& input, OutputContext& output, 
                               ExecutionContext& context) {
  // Pop a buffer asynchronously from the GStreamer pipeline (blocks until available)
  auto buffer_future = gst_sink_resource_.get()->pop_buffer();
  
  // Wait for buffer with timeout to avoid hanging
  if (buffer_future.wait_for(std::chrono::milliseconds(timeout_ms_.get())) == 
      std::future_status::timeout) {
    HOLOSCAN_LOG_ERROR("Timeout waiting for buffer - no data received in {} ms", timeout_ms_.get());
    return;
  }

  // Get the buffer
  gst::Buffer buffer = buffer_future.get();

  // Create TensorMap - supports both packed (RGBA) and planar (I420, NV12) formats
  auto tensor_map = gst_sink_resource_.get()->create_tensor_map_from_buffer(buffer);
  if (tensor_map.empty()) {
    HOLOSCAN_LOG_ERROR("Failed to create tensor map from buffer data");
    return;
  }
  
  output.emit(tensor_map, "output");
}

}  // namespace holoscan

