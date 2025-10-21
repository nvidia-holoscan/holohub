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

namespace {

/**
 * @brief Monitor the GStreamer pipeline bus for errors, EOS, and state changes
 * 
 * This function runs in a separate thread and processes bus messages until
 * an EOS or error is received.
 * 
 * @param pipeline The GStreamer pipeline to monitor
 */
void monitor_pipeline_bus(GstElement* pipeline) {
  auto bus = holoscan::gst::make_gst_object_guard(gst_element_get_bus(pipeline));
  
  while (true) {
    auto msg = holoscan::gst::make_gst_message_guard(
        gst_bus_timed_pop_filtered(bus.get(), 100 * GST_MSECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_STATE_CHANGED)));
    
    if (msg) {
      switch (GST_MESSAGE_TYPE(msg.get())) {
        case GST_MESSAGE_ERROR: {
          GError* error;
          gchar* debug_info;
          gst_message_parse_error(msg.get(), &error, &debug_info);
          auto error_guard = holoscan::gst::make_gst_error_guard(error);
          HOLOSCAN_LOG_ERROR("GStreamer error: {}", error_guard->message);
          if (debug_info) {
            HOLOSCAN_LOG_DEBUG("Debug info: {}", debug_info);
            g_free(debug_info);
          }
          return;
        }
        case GST_MESSAGE_EOS:
          HOLOSCAN_LOG_INFO("End of stream reached");
          return;
        case GST_MESSAGE_STATE_CHANGED: {
          // Only check state changes from the pipeline (not individual elements)
          if (GST_MESSAGE_SRC(msg.get()) == GST_OBJECT(pipeline)) {
            GstState old_state, new_state, pending_state;
            gst_message_parse_state_changed(msg.get(), &old_state, &new_state, &pending_state);
            
            // If pipeline transitions to NULL unexpectedly, stop monitoring
            if (new_state == GST_STATE_NULL && old_state != GST_STATE_NULL) {
              HOLOSCAN_LOG_INFO("GStreamer window closed");
              return;
            }
          }
          break;
        }
        default:
          break;
      }
    }
  }
}

}  // namespace

namespace holoscan {

void GstVideoRecorderOperator::setup(OperatorSpec& spec) {
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
  spec.param(pipeline_desc_, "pipeline_desc", "Pipeline Description",
             "GStreamer pipeline description (first element must be named 'first')",
             std::string("videoconvert name=first ! autovideosink"));
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
  HOLOSCAN_LOG_INFO("Pipeline: '{}'", pipeline_desc_.get());
  
  // Create the GstSrcBridge (initialization happens in constructor)
  bridge_ = std::make_shared<holoscan::gst::GstSrcBridge>(
    name(),           // Use operator name as bridge name
    capabilities,
    queue_limit_.get()
  );
  
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::initialize() - Bridge initialized successfully");
}

void GstVideoRecorderOperator::start() {
  Operator::start();
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::start() - Setting up GStreamer pipeline");
  
  HOLOSCAN_LOG_INFO("Setting up GStreamer pipeline");

  // Get the source element from the bridge
  GstElement* src_element_ptr = bridge_->get_gst_element();
  if (!src_element_ptr) {
    throw std::runtime_error("Failed to get source element from bridge");
  }
  
  // Parse the sink pipeline
  GError* error = nullptr;
  pipeline_ = holoscan::gst::make_gst_object_guard(
      gst_parse_launch(pipeline_desc_.get().c_str(), &error));
  if (error) {
    auto error_guard = holoscan::gst::make_gst_error_guard(error);
    HOLOSCAN_LOG_ERROR("Failed to parse pipeline: {}", error_guard->message);
    throw std::runtime_error("Failed to parse GStreamer pipeline description");
  }
  
  // Add source element to pipeline
  // Note: gst_bin_add() takes ownership by sinking the floating reference (doesn't add a new ref).
  // Since our bridge will call gst_object_unref() when destroyed,
  // we need to manually add a ref here so both the bin and the bridge have their own references.
  gst_object_ref(src_element_ptr);
  gst_bin_add(GST_BIN(pipeline_.get()), src_element_ptr);
  
  // Find and link the "first" element
  // Note: gst_bin_get_by_name returns a new reference, so wrap it in a guard
  auto first_element = holoscan::gst::make_gst_object_guard(
      gst_bin_get_by_name(GST_BIN(pipeline_.get()), "first"));
  if (!first_element) {
    HOLOSCAN_LOG_ERROR("Could not find element named 'first' in pipeline");
    HOLOSCAN_LOG_ERROR("Please name your first pipeline element as 'first', e.g.: 'videoconvert name=first'");
    throw std::runtime_error("Could not find element named 'first' to connect from source");
  }
  
  HOLOSCAN_LOG_INFO("Linking source to {}", gst_element_get_name(first_element.get()));
  
  if (!gst_element_link(src_element_ptr, first_element.get())) {
    HOLOSCAN_LOG_ERROR("Failed to link source to {}", gst_element_get_name(first_element.get()));
    throw std::runtime_error("Failed to link source to pipeline");
  }
  
  HOLOSCAN_LOG_INFO("Pipeline setup complete");

  // Start the GStreamer pipeline
  GstStateChangeReturn ret = gst_element_set_state(pipeline_.get(), GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    HOLOSCAN_LOG_ERROR("Failed to start GStreamer pipeline");
    throw std::runtime_error("Failed to start GStreamer pipeline");
  }

  HOLOSCAN_LOG_INFO("GStreamer pipeline started (will transition to PLAYING asynchronously as data flows)");

  // Start bus monitoring in a background thread
  bus_monitor_future_ = std::async(std::launch::async, 
                                    monitor_pipeline_bus, pipeline_.get());
  
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::start() - Pipeline setup complete");
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
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - Recording stopping, sending EOS");
  
  // Send EOS to signal end of stream
  bridge_->send_eos();
  
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - EOS sent, waiting for pipeline to finish");
  
  // Wait for pipeline to finish processing (EOS message on bus)
  if (bus_monitor_future_.valid()) {
    bus_monitor_future_.wait();
    HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - Pipeline finished processing");
  }

  // Stop and cleanup pipeline
  if (pipeline_ && pipeline_.get() && GST_IS_ELEMENT(pipeline_.get())) {
    gst_element_set_state(pipeline_.get(), GST_STATE_NULL);
    pipeline_.reset();
  }
  
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - Stop complete");
}

}  // namespace holoscan
