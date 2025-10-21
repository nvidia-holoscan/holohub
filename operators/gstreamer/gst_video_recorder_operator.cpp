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

#include <algorithm>
#include <cstring>
#include <gxf/core/gxf.h>

namespace {

/**
 * @brief Determine the parser element name from the encoder element using GStreamer introspection
 * 
 * Queries the encoder's output capabilities to determine the appropriate parser.
 * This approach is future-proof and works with any codec GStreamer supports.
 * 
 * @param encoder The encoder element (e.g., nvh264enc, x265enc)
 * @return The corresponding parser element name (e.g., "h264parse", "h265parse")
 */
std::string get_parser_from_encoder(GstElement* encoder) {
  if (!encoder) {
    throw std::runtime_error("Encoder element is null");
  }
  
  // Get the encoder factory from the element
  GstElementFactory* factory = gst_element_get_factory(encoder);
  if (!factory) {
    throw std::runtime_error("Could not get factory from encoder element");
  }
  
  std::string encoder_name = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
  
  // Get the src pad template (encoder output)
  const GList* pad_templates = gst_element_factory_get_static_pad_templates(factory);
  for (const GList* item = pad_templates; item != nullptr; item = item->next) {
    GstStaticPadTemplate* templ = static_cast<GstStaticPadTemplate*>(item->data);
    
    // Look for src pad (output)
    if (templ->direction != GST_PAD_SRC) {
      continue;
    }
    
    // Get caps from template
    GstCaps* caps = gst_static_caps_get(&templ->static_caps);
    if (!caps) {
      continue;
    }
    
    // Get the first structure (media type)
    std::string parser_name;
    if (gst_caps_get_size(caps) > 0) {
      GstStructure* structure = gst_caps_get_structure(caps, 0);
      const char* media_type = gst_structure_get_name(structure);
      
      if (media_type) {
        std::string media_type_str(media_type);
        HOLOSCAN_LOG_DEBUG("Encoder '{}' outputs media type: {}", encoder_name, media_type_str);
        
        // Extract codec name from media type: "video/x-{codec}" -> "{codec}parse"
        // This works for all codecs following GStreamer's naming convention:
        // x-h264 -> h264parse, x-h265 -> h265parse, x-vp8 -> vp8parse, etc.
        size_t pos = media_type_str.find("x-");
        if (pos != std::string::npos && pos + 2 < media_type_str.length()) {
          std::string codec = media_type_str.substr(pos + 2);
          parser_name = codec + "parse";
          HOLOSCAN_LOG_DEBUG("Derived parser '{}' from media type '{}'", parser_name, media_type_str);
        }
      }
    }
    
    // Cleanup caps
    gst_caps_unref(caps);
    
    // Return parser name if we found one
    if (!parser_name.empty()) {
      return parser_name;
    }
  }
  
  // If we couldn't determine the parser, fail explicitly
  throw std::runtime_error("Could not determine parser for encoder '" + encoder_name + 
                           "'. The encoder may not output a video codec with a standard parser.");
}

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
  
  spec.param(encoder_name_, "encoder", "Encoder",
             "Encoder base name (e.g., nvh264, nvh265, x264, x265). 'enc' suffix is appended automatically.",
             std::string("nvh264"));
  spec.param(framerate_, "framerate", "Framerate",
             "Video framerate (fps)",
             30);
  spec.param(queue_limit_, "queue_limit", "Queue Limit",
             "Maximum number of buffers to queue (0 = unlimited)",
             size_t(10));
  spec.param(timeout_ms_, "timeout_ms", "Timeout (ms)", 
             "Timeout in milliseconds for buffer push",
             1000UL);
  spec.param(filename_, "filename", "Output Filename",
             "Output video filename",
             std::string("output.mp4"));
}

void GstVideoRecorderOperator::start() {
  Operator::start();
  
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator - Starting");
  HOLOSCAN_LOG_INFO("Output filename: '{}'", filename_.get());
  HOLOSCAN_LOG_INFO("Encoder: {}enc", encoder_name_.get());
  HOLOSCAN_LOG_INFO("Framerate: {}fps", framerate_.get());
  HOLOSCAN_LOG_INFO("Queue limit: {}", queue_limit_.get());
  HOLOSCAN_LOG_INFO("Timeout: {}ms", timeout_ms_.get());
  HOLOSCAN_LOG_INFO("Video parameters (width, height, format, storage) will be detected from first frame");
  HOLOSCAN_LOG_INFO("Setting up GStreamer pipeline (without source)");
  
  // Create pipeline
  pipeline_ = holoscan::gst::make_gst_object_guard(gst_pipeline_new("video-recorder-pipeline"));
  if (!pipeline_) {
    throw std::runtime_error("Failed to create GStreamer pipeline");
  }
  
  // Create encoder element first (append "enc" suffix to encoder base name)
  std::string encoder_element = encoder_name_.get() + "enc";
  encoder_ = holoscan::gst::make_gst_object_guard(
      gst_element_factory_make(encoder_element.c_str(), "encoder"));
  if (!encoder_) {
    HOLOSCAN_LOG_ERROR("Failed to create encoder element '{}'", encoder_element);
    throw std::runtime_error("Failed to create encoder element: " + encoder_element);
  }
  
  // Determine parser from encoder element
  std::string parser_name = get_parser_from_encoder(encoder_.get());
  HOLOSCAN_LOG_INFO("Auto-detected parser: {}", parser_name);
  
  // Create remaining pipeline elements (without source and converter - those will be added on first frame)
  auto parser = holoscan::gst::make_gst_object_guard(
      gst_element_factory_make(parser_name.c_str(), "parser"));
  auto muxer = holoscan::gst::make_gst_object_guard(
      gst_element_factory_make("mp4mux", "muxer"));
  auto filesink = holoscan::gst::make_gst_object_guard(
      gst_element_factory_make("filesink", "filesink"));
  
  if (!parser || !muxer || !filesink) {
    HOLOSCAN_LOG_ERROR("Failed to create one or more GStreamer elements");
    throw std::runtime_error("Failed to create GStreamer pipeline elements");
  }
  
  // Configure filesink with output filename
  g_object_set(filesink.get(), "location", filename_.get().c_str(), nullptr);
  HOLOSCAN_LOG_INFO("Output file: {}", filename_.get());
  
  // Add all elements to pipeline (this sinks their floating references)
  // We need to add refs since our guards will unref them
  gst_object_ref(encoder_.get());
  gst_object_ref(parser.get());
  gst_object_ref(muxer.get());
  gst_object_ref(filesink.get());
  
  gst_bin_add_many(GST_BIN(pipeline_.get()), 
                   encoder_.get(), parser.get(), 
                   muxer.get(), filesink.get(), nullptr);
  
  // Link elements: encoder -> parser -> muxer -> filesink
  // Source and converter will be added and linked on first frame
  if (!gst_element_link_many(encoder_.get(), parser.get(), 
                             muxer.get(), filesink.get(), nullptr)) {
    HOLOSCAN_LOG_ERROR("Failed to link pipeline elements");
    throw std::runtime_error("Failed to link pipeline elements");
  }
  
  HOLOSCAN_LOG_INFO("Pipeline created: {}enc -> {} -> mp4mux -> filesink", 
                    encoder_name_.get(), parser_name);
  HOLOSCAN_LOG_INFO("Source and converter will be added on first frame based on detected storage type");
  
  // Start the GStreamer pipeline
  GstStateChangeReturn ret = gst_element_set_state(pipeline_.get(), GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    HOLOSCAN_LOG_ERROR("Failed to start GStreamer pipeline");
    throw std::runtime_error("Failed to start GStreamer pipeline");
  }
 
  HOLOSCAN_LOG_INFO("GStreamer pipeline started (waiting for source to be added)");
 
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
  HOLOSCAN_LOG_INFO("Frame #{} - Entity received", frame_count);

  // Initialize bridge on first frame
  if (!bridge_initialized_) {
    HOLOSCAN_LOG_INFO("Frame #{} - First frame, detecting video parameters from tensor", frame_count);
    
    // Get the first tensor from the entity using GXF C API
    gxf_uid_t component_ids[64];
    uint64_t num_components = 64;
    gxf_result_t result = GxfComponentFindAll(entity.context(), entity.eid(), 
                                              &num_components, component_ids);
    if (result != GXF_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to find components in entity");
      throw std::runtime_error("Failed to find components in entity");
    }
    
    nvidia::gxf::Tensor* tensor = nullptr;
    
    // Find the first tensor component
    for (uint64_t i = 0; i < num_components; i++) {
      gxf_tid_t tid;
      result = GxfComponentType(entity.context(), component_ids[i], &tid);
      if (result != GXF_SUCCESS) continue;
      
      const char* type_name = nullptr;
      result = GxfComponentTypeName(entity.context(), tid, &type_name);
      if (result != GXF_SUCCESS || !type_name) continue;
      
      if (std::strcmp(type_name, "nvidia::gxf::Tensor") == 0) {
        void* tensor_ptr = nullptr;
        result = GxfComponentPointer(entity.context(), component_ids[i], 
                                      GxfTidNull(), &tensor_ptr);
        if (result == GXF_SUCCESS) {
          tensor = static_cast<nvidia::gxf::Tensor*>(tensor_ptr);
          break;
        }
      }
    }
    
    if (!tensor) {
      HOLOSCAN_LOG_ERROR("No tensor found in entity");
      throw std::runtime_error("No tensor found in entity");
    }
    
    // Extract video parameters from tensor
    auto shape = tensor->shape();
    if (shape.rank() < 3) {
      HOLOSCAN_LOG_ERROR("Tensor rank is {}, expected at least 3 (height, width, channels)", shape.rank());
      throw std::runtime_error("Invalid tensor shape for video data");
    }
    
    int height = shape.dimension(0);
    int width = shape.dimension(1);
    int channels = shape.dimension(2);
    
    // Determine format based on number of channels
    std::string format;
    if (channels == 4) {
      format = "RGBA";
    } else if (channels == 3) {
      format = "RGB";
    } else if (channels == 1) {
      format = "GRAY8";
    } else {
      HOLOSCAN_LOG_ERROR("Unsupported number of channels: {}", channels);
      throw std::runtime_error("Unsupported number of channels");
    }
    
    // Determine storage type from tensor memory location
    bool is_device_memory = (tensor->storage_type() == nvidia::gxf::MemoryStorageType::kDevice);
    std::string storage_str = is_device_memory ? "device" : "host";
    
    HOLOSCAN_LOG_INFO("Detected video parameters: {}x{}@{}fps, format={}, channels={}, storage={}",
                      width, height, framerate_.get(), format, channels, storage_str);
    
    // Build caps string with detected parameters
    std::string capabilities;
    if (is_device_memory) {
      // CUDA memory: insert memory feature after media type
      capabilities = "video/x-raw(memory:CUDAMemory),format=" + format;
    } else {
      // Host memory: use default caps
      capabilities = "video/x-raw,format=" + format;
    }
    
    capabilities += ",width=" + std::to_string(width) + 
                    ",height=" + std::to_string(height) + 
                    ",framerate=" + std::to_string(framerate_.get()) + "/1";
    
    HOLOSCAN_LOG_INFO("Capabilities: '{}'", capabilities);
    
    // Create the GstSrcBridge
    bridge_ = std::make_shared<holoscan::gst::GstSrcBridge>(
      name(),           // Use operator name as bridge name
      capabilities,
      queue_limit_.get()
    );
    
    HOLOSCAN_LOG_INFO("Bridge created");
    
    // Get the source element from the bridge
    GstElement* src_element_ptr = bridge_->get_gst_element();
    if (!src_element_ptr) {
      throw std::runtime_error("Failed to get source element from bridge");
    }
    
    // Create appropriate converter based on storage type
    const char* converter_name = is_device_memory ? "cudaconvert" : "videoconvert";
    HOLOSCAN_LOG_INFO("Creating {} for {} memory", converter_name, storage_str);
    
    auto converter = holoscan::gst::make_gst_object_guard(
        gst_element_factory_make(converter_name, "converter"));
    if (!converter) {
      HOLOSCAN_LOG_ERROR("Failed to create {} element", converter_name);
      throw std::runtime_error(std::string("Failed to create ") + converter_name + " element");
    }
    
    // Add source and converter elements to pipeline
    // Note: gst_bin_add() takes ownership by sinking the floating reference (doesn't add a new ref).
    // Since our bridge will call gst_object_unref() when destroyed,
    // we need to manually add a ref here so both the bin and the bridge have their own references.
    gst_object_ref(src_element_ptr);
    gst_object_ref(converter.get());
    
    gst_bin_add_many(GST_BIN(pipeline_.get()), src_element_ptr, converter.get(), nullptr);
    
    // Set elements to PLAYING state to match the pipeline
    GstStateChangeReturn ret = gst_element_set_state(src_element_ptr, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      HOLOSCAN_LOG_ERROR("Failed to set source element to PLAYING state");
      throw std::runtime_error("Failed to set source element to PLAYING state");
    }
    
    ret = gst_element_set_state(converter.get(), GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      HOLOSCAN_LOG_ERROR("Failed to set converter element to PLAYING state");
      throw std::runtime_error("Failed to set converter element to PLAYING state");
    }
    
    // Link elements: source -> converter -> encoder
    if (!gst_element_link(src_element_ptr, converter.get())) {
      HOLOSCAN_LOG_ERROR("Failed to link source to converter");
      throw std::runtime_error("Failed to link source to converter");
    }
    
    if (!gst_element_link(converter.get(), encoder_.get())) {
      HOLOSCAN_LOG_ERROR("Failed to link converter to encoder");
      throw std::runtime_error("Failed to link converter to encoder");
    }
    
    HOLOSCAN_LOG_INFO("Pipeline complete: source -> {} -> {}enc -> parser -> mp4mux -> filesink",
                      converter_name, encoder_name_.get());
    bridge_initialized_ = true;
  }

  HOLOSCAN_LOG_INFO("Frame #{} - Converting entity to GStreamer buffer", frame_count);

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
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - Recording stopping");
  
  // Send EOS to signal end of stream (only if bridge was initialized)
  if (bridge_initialized_ && bridge_) {
    HOLOSCAN_LOG_INFO("Sending EOS to bridge");
    bridge_->send_eos();
    
    HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - EOS sent, waiting for pipeline to finish");
    
    // Wait for pipeline to finish processing (EOS message on bus)
    if (bus_monitor_future_.valid()) {
      bus_monitor_future_.wait();
      HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - Pipeline finished processing");
    }
  } else {
    HOLOSCAN_LOG_INFO("Bridge was never initialized, no frames were processed");
  }
  
  // Stop and cleanup pipeline
  if (pipeline_ && pipeline_.get() && GST_IS_ELEMENT(pipeline_.get())) {
    gst_element_set_state(pipeline_.get(), GST_STATE_NULL);
    pipeline_.reset();
  }
  
  HOLOSCAN_LOG_INFO("GstVideoRecorderOperator::stop() - Stop complete");
}

}  // namespace holoscan
