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

#include "gst_video_recorder_operator.hpp"

namespace holoscan {

void GstVideoRecorderOperator::setup(OperatorSpec& spec) {
  // Initialize the future from the promise on first setup
  if (!element_future_.valid()) {
    element_future_ = element_promise_.get_future();
  }
  
  spec.input<gxf::Entity>("input");
  
  spec.param(width_, "width", "Width",
             "Video frame width in pixels",
             640);
  spec.param(height_, "height", "Height",
             "Video frame height in pixels",
             480);
  spec.param(framerate_, "framerate", "Framerate",
             "Video framerate (fps)",
             30);
  spec.param(format_, "format", "Format",
             "Video format (e.g., RGBA, RGB, I420, NV12)",
             std::string("RGBA"));
  spec.param(storage_type_, "storage_type", "Storage Type",
             "Memory storage type (0=host, 1=device/CUDA)",
             0);
  spec.param(queue_limit_, "queue_limit", "Queue Limit",
             "Maximum number of buffers to queue (0 = unlimited)",
             size_t(10));
  spec.param(timeout_ms_, "timeout_ms", "Timeout (ms)", 
             "Timeout in milliseconds for buffer push",
             1000UL);
}

void GstVideoRecorderOperator::initialize() {
  Operator::initialize();
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::initialize() - Initializing GStreamer bridge");
  
  // Build caps string with actual width, height, framerate, and memory type
  // Memory feature must come right after media type: video/x-raw(memory:CUDAMemory)
  std::string capabilities;
  
  if (storage_type_.get() == 1) {
    // CUDA memory: insert memory feature after media type
    capabilities = "video/x-raw(memory:CUDAMemory),format=" + format_.get();
  } else {
    // Host memory: use default caps
    capabilities = "video/x-raw,format=" + format_.get();
  }
  
  capabilities += ",width=" + std::to_string(width_.get()) + 
                  ",height=" + std::to_string(height_.get()) + 
                  ",framerate=" + std::to_string(framerate_.get()) + "/1";
  
  HOLOSCAN_LOG_INFO("Video parameters: {}x{}@{}fps, format={}, storage={}",
                    width_.get(), height_.get(), framerate_.get(), 
                    format_.get(), storage_type_.get() == 1 ? "device" : "host");
  HOLOSCAN_LOG_INFO("Capabilities: '{}'", capabilities);
  HOLOSCAN_LOG_INFO("Queue limit: {}", queue_limit_.get());
  HOLOSCAN_LOG_INFO("Timeout: {}ms", timeout_ms_.get());
  
  // Create the GstSrcBridge (initialization happens in constructor)
  bridge_ = std::make_shared<holoscan::gst::GstSrcBridge>(
    name(),           // Use operator name as bridge name
    capabilities,
    queue_limit_.get()
  );
  
  // Set the promise with the GStreamer element so callers can wait for it
  element_promise_.set_value(bridge_->get_gst_element());
  
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::initialize() - Bridge initialized successfully");
}

void GstVideoRecorderOperator::compute(InputContext& input, OutputContext& output, 
                              ExecutionContext& context) {
  static int frame_count = 0;
  frame_count++;
  
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::compute() - Frame #{} - Receiving entity", frame_count);
  
  // Receive the video frame entity from the input port
  auto entity = input.receive<gxf::Entity>("input").value();
  HOLOSCAN_LOG_INFO("Frame #{} - Entity received, converting to GStreamer buffer", frame_count);

  // Convert entity to GStreamer buffer using the bridge
  auto buffer = bridge_->create_buffer_from_entity(entity);
  if (buffer.size() == 0) {
    HOLOSCAN_LOG_ERROR("Frame #{} - Failed to convert entity to buffer", frame_count);
    return;
  }

  HOLOSCAN_LOG_INFO("Frame #{} - Buffer created, size: {} bytes", frame_count, buffer.size());

  // Push buffer into the GStreamer encoding pipeline
  auto timeout = std::chrono::milliseconds(timeout_ms_.get());
  HOLOSCAN_LOG_INFO("Frame #{} - Pushing buffer to encoding pipeline (timeout: {}ms)", 
                    frame_count, timeout_ms_.get());
  
  if (!bridge_->push_buffer(std::move(buffer), timeout)) {
    HOLOSCAN_LOG_ERROR("Frame #{} - Failed to push buffer to encoding pipeline (timeout or error)", 
                       frame_count);
    return;
  }
  
  HOLOSCAN_LOG_INFO("Frame #{} - Buffer successfully pushed to encoding pipeline", frame_count);
}

void GstVideoRecorderOperator::stop() {
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - Recording stopping, finalizing output file");
  
  // Send EOS with 2 second wait time to allow encoding and muxing to complete
  // This is critical for proper file finalization (writing headers/trailers)
  bridge_->send_eos(1000);
  
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - Recording complete, file finalized");
}

}  // namespace holoscan
