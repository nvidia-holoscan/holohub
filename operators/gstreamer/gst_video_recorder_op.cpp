/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gst_video_recorder_op.hpp"

#include <gxf/core/gxf.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <holoscan/core/domain/tensor_map.hpp>
#include <regex>

#include "gst/config.hpp"
#include "gst/error.hpp"
#include "gst/message.hpp"

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
#include <gst/cuda/gstcudamemory.h>
#endif

// Timeout for waiting for EOS to be processed during pipeline shutdown
constexpr std::chrono::seconds kEosTimeoutSeconds{30};

// Define CUDA memory feature name if not already defined by GStreamer (available since 1.22).
#ifndef GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY
#define GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY "memory:CUDAMemory"
#endif

namespace holoscan {
namespace {

/**
 * @brief Normalize framerate string to GStreamer fraction format
 *
 * Accepts and normalizes:
 * - Integer: "30" -> "30/1", "0" -> "0/1" (live mode)
 * - Fraction: "30000/1001" -> "30000/1001" (no change)
 * - Decimal (converted to fraction): "29.97" -> "2997/100"
 *
 * @param framerate Framerate string to normalize (frame per second)
 * @return Normalized framerate as fraction string
 * @throws std::runtime_error if framerate format is invalid
 */
std::string normalize_framerate(const std::string& framerate) {
  // Regex for integer (e.g., "30") or fraction (e.g., "30000/1001")
  std::regex rational_regex(R"(^\s*(\d+)\s*(/\s*(\d+)\s*)?$)");
  std::smatch match;

  if (std::regex_match(framerate, match, rational_regex)) {
    // If no denominator, add "/1"
    if (match[2].str().empty()) {
      return framerate + "/1";
    }
    return framerate;
  }

  // Also accept decimal format (e.g., "29.97") and convert to fraction
  std::regex decimal_regex(R"(^\s*(\d+)\.(\d+)\s*$)");
  if (std::regex_match(framerate, match, decimal_regex)) {
    int whole = std::stoi(match[1].str());
    std::string decimal_part = match[2].str();
    int decimal_places = static_cast<int>(decimal_part.length());

    // Limit decimal places to prevent overflow (9 digits max for int32)
    if (decimal_places > 9) {
      throw std::runtime_error("Framerate decimal precision too high: '" + framerate +
                               "'. Maximum 9 decimal places supported.");
    }

    int fractional = std::stoi(decimal_part);

    // Convert to rational: 29.97 = 2997/100
    int denominator = 1;
    for (int i = 0; i < decimal_places; ++i) { denominator *= 10; }
    int numerator = whole * denominator + fractional;

    std::string result = std::to_string(numerator) + "/" + std::to_string(denominator);
    HOLOSCAN_LOG_DEBUG("Converted decimal framerate to fraction: {}", result);
    return result;
  }

  throw std::runtime_error("Invalid framerate format: '" + framerate +
                           "'. Expected formats: '30', '30/1', '30000/1001', or '29.97'");
}

/**
 * @brief Get the muxer element name and potentially update filename based on file extension
 *
 * Maps file extensions to appropriate GStreamer muxer elements:
 * - .mp4 → mp4mux
 * - .mkv → matroskamux
 *
 * If no extension is provided, defaults to mp4mux and appends .mp4 to filename.
 *
 * @param filename Reference to filename (may be modified to add .mp4 extension)
 * @return The muxer element name
 * @note Never throws - defaults to mp4mux for unsupported extensions
 */
std::string get_muxer_from_extension(std::string& filename) {
  std::filesystem::path filepath(filename);
  std::string extension = filepath.extension().string();

  // Convert to lowercase for case-insensitive comparison
  std::transform(extension.begin(),
                 extension.end(),
                 extension.begin(),
                 static_cast<int (*)(int)>(std::tolower));

  // If no extension, append .mp4 and set extension
  if (extension.empty()) {
    extension = ".mp4";
    filename += extension;
    HOLOSCAN_LOG_INFO("No file extension provided, defaulting to mp4: {}", filename);
  }

  // Map extension to muxer (extension includes the dot)
  if (extension == ".mp4") {
    return "mp4mux";
  } else if (extension == ".mkv") {
    return "matroskamux";
  }

  // Unsupported extension - default to mp4mux
  HOLOSCAN_LOG_WARN("Unsupported file extension '{}', defaulting to mp4mux", extension);
  return "mp4mux";
}

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

    // Get caps from template (RAII wrapper handles cleanup automatically)
    holoscan::gst::Caps caps(gst_static_caps_get(&templ->static_caps));

    // Get the first structure (media type)
    if (caps.get_size() > 0) {
      GstStructure* structure = caps.get_structure(0);
      const char* media_type = gst_structure_get_name(structure);

      if (media_type) {
        std::string_view media_type_str(media_type);
        HOLOSCAN_LOG_DEBUG("Encoder '{}' outputs media type: {}", encoder_name, media_type_str);

        // Extract codec name from media type: "video/x-{codec}" -> "{codec}parse"
        // This works for all codecs following GStreamer's naming convention:
        // video/x-h264 -> h264parse, video/x-h265 -> h265parse, video/x-vp8 -> vp8parse, etc.
        size_t pos = media_type_str.find("video/x-");
        if (pos != std::string_view::npos && pos + 8 < media_type_str.length()) {
          auto codec = media_type_str.substr(pos + 8);
          auto parser_name = std::string(codec) + "parse";
          HOLOSCAN_LOG_DEBUG(
              "Derived parser '{}' from media type '{}'", parser_name, media_type_str);
          return parser_name;
        }
      }
    }
  }

  // If we couldn't determine the parser, fail explicitly
  throw std::runtime_error("Could not determine parser for encoder '" + encoder_name +
                           "'. The encoder may not output a video codec with a standard parser.");
}

/**
 * @brief Create GstSrcBridge from tensor map by detecting video parameters
 *
 * Inspects the tensor map to extract video parameters from the first tensor
 * and creates a GstSrcBridge with appropriate capabilities.
 *
 * @param tensor_map The TensorMap containing one or more tensors
 * @param operator_name Name for the bridge
 * @param format Pixel format (e.g., "RGBA", "RGB", "BGRA", "GRAY8")
 * @param framerate Framerate string (e.g., "30/1")
 * @param max_buffers Maximum queue size
 * @param block Whether push_buffer() should block when queue is full (default: true)
 * @return Shared pointer to the created GstSrcBridge
 * @throws std::runtime_error if tensor map is empty, tensor is null, or tensor shape is invalid
 */
std::shared_ptr<holoscan::GstSrcBridge> create_src_bridge_from_tensor_map(
    const holoscan::TensorMap& tensor_map, const std::string& operator_name,
    const std::string& format, const std::string& framerate, size_t max_buffers,
    bool block = true) {
  // Get the first tensor from the map
  if (tensor_map.empty()) {
    HOLOSCAN_LOG_ERROR("TensorMap is empty");
    throw std::runtime_error("TensorMap is empty");
  }

  const auto& first_tensor_ptr = tensor_map.begin()->second;
  if (!first_tensor_ptr) {
    HOLOSCAN_LOG_ERROR("First tensor in TensorMap is null");
    throw std::runtime_error("First tensor in TensorMap is null");
  }

  // Extract video parameters from tensor
  auto shape = first_tensor_ptr->shape();
  if (shape.size() < 2 || shape.size() > 3) {
    HOLOSCAN_LOG_ERROR(
        "Tensor rank is {}, expected 2 (height, width) "
        "or 3 (height, width, channels) dimensions",
        shape.size());
    throw std::runtime_error("Invalid tensor shape for video data");
  }

  int height = shape[0];
  int width = shape[1];
  // Validate channels if present (rank 3)
  if (shape.size() == 3) {
    int channels = shape[2];
    // Common video formats: GRAY8(1), RGB(3), RGBA/BGRA(4)
    if (channels != 1 && channels != 3 && channels != 4) {
      HOLOSCAN_LOG_ERROR("Tensor has invalid channel count {} (expected 1, 3, or 4)", channels);
      throw std::runtime_error("Invalid channel count for video data");
    }
    HOLOSCAN_LOG_DEBUG("Detected {} channels from tensor shape", channels);
  }

  // Determine storage type from tensor memory location
  auto device = first_tensor_ptr->device();
  // Only kDLCUDA and kDLCUDAManaged map to nvidia::gxf::MemoryStorageType::kDevice
  // kDLCUDAHost is pinned host memory (kHost), not device memory
  bool is_device_memory = (device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged);
  std::string storage_str = is_device_memory ? "device" : "host";

  HOLOSCAN_LOG_INFO("Detected video parameters: {}x{}@{}fps, format={}, storage={}",
                    width,
                    height,
                    framerate,
                    format,
                    storage_str);

  // Build caps string with detected parameters
  std::string capabilities = "video/x-raw";
  if (is_device_memory) {
    capabilities += "(" GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY ")";
  }
  capabilities += ",format=" + format + ",width=" + std::to_string(width) +
                  ",height=" + std::to_string(height) + ",framerate=" + framerate;

  HOLOSCAN_LOG_INFO("Capabilities: '{}'", capabilities);

  // Create and return the GstSrcBridge
  return std::make_shared<holoscan::GstSrcBridge>(operator_name, capabilities, max_buffers, block);
}

/**
 * @brief Create appropriate converter element based on memory type
 *
 * Creates cudaconvert for device (CUDA) memory or videoconvert for host memory.
 *
 * @param is_device_memory True if using CUDA memory, false for host memory
 * @return GstElement wrapping the created converter element
 * @throws std::runtime_error if element creation fails
 */
holoscan::gst::Element create_converter_element(bool is_device_memory) {
  const char* converter_name = is_device_memory ? "cudaconvert" : "videoconvert";
  std::string storage_str = is_device_memory ? "device" : "host";
  HOLOSCAN_LOG_INFO("Creating {} for {} memory", converter_name, storage_str);

  auto converter =
      holoscan::gst::Element(gst_element_factory_make(converter_name, "converter")).ref_sink();
  if (!converter) {
    HOLOSCAN_LOG_ERROR("Failed to create {} element", converter_name);
    throw std::runtime_error(std::string("Failed to create ") + converter_name + " element");
  }

  return converter;
}

/**
 * @brief Add source and converter elements to pipeline and link them to encoder
 *
 * @param pipeline The pipeline to add elements to
 * @param src_element Source element (appsrc from bridge) - modified to set state
 * @param converter Converter element (cudaconvert or videoconvert) - modified to set state
 * @param encoder Encoder element to link to
 * @throws std::runtime_error if state change or linking fails
 */
void add_and_link_source_converter(holoscan::gst::Pipeline& pipeline,
                                   holoscan::gst::Element& src_element,
                                   holoscan::gst::Element& converter,
                                   const holoscan::gst::Element& encoder) {
  // Add source and converter elements to pipeline using type-safe add_many
  pipeline.add_many(src_element, converter);

  // Set elements to PLAYING state to match the pipeline
  GstStateChangeReturn ret = src_element.set_state(GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    throw std::runtime_error("Failed to set source element to PLAYING state");
  }

  ret = converter.set_state(GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    throw std::runtime_error("Failed to set converter element to PLAYING state");
  }

  // Link elements: source -> converter -> encoder
  if (!src_element.link(converter)) {
    throw std::runtime_error("Failed to link source to converter");
  }

  if (!converter.link(encoder)) {
    throw std::runtime_error("Failed to link converter to encoder");
  }

  HOLOSCAN_LOG_INFO(
      "Pipeline complete: source -> converter -> encoder -> "
      "parser -> muxer -> filesink");
}

}  // unnamed namespace

void GstVideoRecorderOp::setup(OperatorSpec& spec) {
  spec.input<TensorMap>("input");

  // Register converter for std::map<std::string, std::string> to enable Arg() support
  register_converter<std::map<std::string, std::string>>();

  spec.param(encoder_name_,
             "encoder",
             "Encoder",
             "Encoder base name (e.g., nvh264, nvh265, x264, x265). "
             "'enc' suffix is appended automatically.",
             std::string("nvh264"));
  spec.param(format_,
             "format",
             "Pixel Format",
             "Video pixel format (e.g., RGBA, RGB, BGRA, BGR, GRAY8)",
             std::string("RGBA"));
  spec.param(framerate_,
             "framerate",
             "Framerate",
             "Video framerate as fraction (e.g., '30/1', '30000/1001', '29.97')",
             std::string("30/1"));
  spec.param(max_buffers_,
             "max-buffers",
             "Max Buffers",
             "Maximum number of buffers to queue (0 = unlimited)",
             size_t(10));
  spec.param(block_,
             "block",
             "Block When Full",
             "Whether push_buffer() should block when the internal queue is full "
             "(true = block, false = non-blocking, may drop/timeout)",
             true);
  spec.param(
      filename_, "filename", "Output Filename", "Output video filename", std::string("output.mp4"));
  spec.param(properties_,
             "properties",
             "Encoder Properties",
             "Map of encoder-specific properties (e.g., bitrate, preset, gop-size)",
             std::map<std::string, std::string>());
}

void GstVideoRecorderOp::start() {
  Operator::start();

  // Initialize frame counter
  frame_count_ = 0;

  HOLOSCAN_LOG_INFO("GstVideoRecorderOp - Starting");
  HOLOSCAN_LOG_INFO("Output filename: '{}'", filename_.get());
  HOLOSCAN_LOG_INFO("Encoder: {}enc", encoder_name_.get());

  // Normalize framerate format and update the parameter
  framerate_ = normalize_framerate(framerate_.get());
  HOLOSCAN_LOG_INFO("Framerate: {} fps", framerate_.get());

  HOLOSCAN_LOG_INFO("Max buffers: {}", max_buffers_.get());
  HOLOSCAN_LOG_INFO("Block when queue is full: {}", block_.get());
  HOLOSCAN_LOG_INFO(
      "Video parameters (width, height, format, storage) will be "
      "detected from first frame");
  HOLOSCAN_LOG_INFO("Setting up GStreamer pipeline (without source)");

  // Create pipeline
  pipeline_ = gst::Pipeline::create("video-recorder-pipeline");

  // Create encoder element first (append "enc" suffix to encoder base name)
  std::string encoder_element = encoder_name_.get() + "enc";
  encoder_ = gst::Element(gst_element_factory_make(encoder_element.c_str(), "encoder")).ref_sink();
  if (!encoder_) {
    HOLOSCAN_LOG_ERROR("Failed to create encoder element '{}'", encoder_element);
    throw std::runtime_error("Failed to create encoder element: " + encoder_element);
  }

  // Apply encoder properties from the properties map
  const auto& props = properties_.get();
  if (!props.empty()) {
    HOLOSCAN_LOG_INFO("Applying {} encoder properties:", props.size());
    for (const auto& [key, value] : props) {
      if (encoder_.find_and_set_property(key, value)) {
        HOLOSCAN_LOG_INFO("  {} = {}", key, value);
      } else {
        HOLOSCAN_LOG_WARN(
            "  {} = {} (failed to set property on encoder '{}')", key, value, encoder_element);
      }
    }
  }

  // Determine parser from encoder element
  std::string parser_name = get_parser_from_encoder(encoder_.get());
  HOLOSCAN_LOG_INFO("Auto-detected parser: {}", parser_name);

  // Determine muxer from file extension (may modify filename to add .mp4 if no extension)
  std::string output_filename = filename_.get();
  std::string muxer_name = get_muxer_from_extension(output_filename);
  HOLOSCAN_LOG_INFO("Auto-detected muxer: {} for extension in '{}'", muxer_name, output_filename);

  // Create remaining pipeline elements (without source and converter -
  // those will be added on first frame)
  auto parser = gst::Element(gst_element_factory_make(parser_name.c_str(), "parser")).ref_sink();
  auto muxer = gst::Element(gst_element_factory_make(muxer_name.c_str(), "muxer")).ref_sink();
  auto filesink = gst::Element(gst_element_factory_make("filesink", "filesink")).ref_sink();

  if (!parser) {
    HOLOSCAN_LOG_ERROR("Failed to create parser element '{}'", parser_name);
    throw std::runtime_error("Failed to create parser element: " + parser_name);
  }
  if (!muxer) {
    HOLOSCAN_LOG_ERROR("Failed to create muxer element '{}'", muxer_name);
    throw std::runtime_error("Failed to create muxer element: " + muxer_name);
  }
  if (!filesink) {
    HOLOSCAN_LOG_ERROR("Failed to create filesink element");
    throw std::runtime_error("Failed to create filesink element");
  }

  // Configure filesink with output filename (potentially modified with .mp4 extension)
  filesink.set_properties("location", output_filename);
  HOLOSCAN_LOG_INFO("Output file: {}", output_filename);

  // Add all elements to pipeline
  pipeline_.add_many(encoder_, parser, muxer, filesink);

  // Link elements: encoder -> parser -> muxer -> filesink
  // Source and converter will be added and linked on first frame
  if (!encoder_.link_many(parser, muxer, filesink)) {
    HOLOSCAN_LOG_ERROR("Failed to link pipeline elements");
    throw std::runtime_error("Failed to link pipeline elements");
  }

  HOLOSCAN_LOG_INFO("Pipeline created: {}enc -> {} -> {} -> filesink",
                    encoder_name_.get(),
                    parser_name,
                    muxer_name);
  HOLOSCAN_LOG_INFO(
      "Source and converter will be added on first frame based on "
      "detected storage type");

  // Start the GStreamer pipeline
  GstStateChangeReturn ret = pipeline_.set_state(GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    HOLOSCAN_LOG_ERROR("Failed to start GStreamer pipeline");
    throw std::runtime_error("Failed to start GStreamer pipeline");
  }

  HOLOSCAN_LOG_INFO("GStreamer pipeline started (waiting for source to be added)");

  // Start bus monitoring in a background thread
  bus_monitor_ = std::make_unique<gst::PipelineBusMonitor>(pipeline_);
  bus_monitor_->start();

  HOLOSCAN_LOG_INFO("GstVideoRecorderOp::start() - Pipeline setup complete");
}

void GstVideoRecorderOp::compute(InputContext& input, OutputContext& output,
                                       ExecutionContext& context) {
  frame_count_++;

  HOLOSCAN_LOG_DEBUG(
      "GstVideoRecorderOp::compute() - Frame #{} - "
      "Receiving tensor map",
      frame_count_);

  // Receive the video frame tensor map from the input port
  auto tensor_map = input.receive<TensorMap>("input").value();
  HOLOSCAN_LOG_DEBUG("Frame #{} - TensorMap received", frame_count_);

  // Initialize bridge on first frame
  // Only upon receiving the first frame, we know the frame parameters
  if (!src_bridge_) {
    HOLOSCAN_LOG_INFO("Frame #{} - First frame, detecting video parameters from tensor",
                      frame_count_);
    // Create bridge from tensor map (detects video parameters automatically)
    src_bridge_ = create_src_bridge_from_tensor_map(
        tensor_map, name(), format_.get(), framerate_.get(), max_buffers_.get(), block_.get());
    HOLOSCAN_LOG_INFO("Bridge created");

    // Create appropriate converter based on storage type and add to pipeline
    auto converter = create_converter_element(
        src_bridge_->get_caps().has_feature(GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY));
    // Add source and converter to pipeline and link them to encoder
    gst::Element src_element = src_bridge_->get_gst_element();
    add_and_link_source_converter(pipeline_, src_element, converter, encoder_);
  }

  HOLOSCAN_LOG_DEBUG("Frame #{} - Converting tensor map to GStreamer buffer", frame_count_);

  // Convert tensor map to GStreamer buffer using the bridge
  auto buffer = src_bridge_->create_buffer_from_tensor_map(tensor_map);
  if (buffer.get_size() == 0) {
    HOLOSCAN_LOG_ERROR("Frame #{} - Failed to convert tensor map to buffer", frame_count_);
    return;
  }

  HOLOSCAN_LOG_DEBUG("Frame #{} - Buffer created, size: {} bytes", frame_count_, buffer.get_size());

  // Push buffer into the GStreamer encoding pipeline
  if (!src_bridge_->push_buffer(std::move(buffer))) {
    HOLOSCAN_LOG_ERROR("Frame #{} - Failed to push buffer to encoding pipeline (timeout or error)",
                       frame_count_);
    return;
  }

  HOLOSCAN_LOG_DEBUG("Frame #{} - Buffer successfully pushed to encoding pipeline", frame_count_);
}

void GstVideoRecorderOp::stop() {
  HOLOSCAN_LOG_INFO("GstVideoRecorderOp::stop() - Recording stopping");

  // Send EOS to signal end of stream (only if bridge was initialized)
  if (src_bridge_) {
    HOLOSCAN_LOG_INFO("Sending EOS to bridge");
    src_bridge_->send_eos();
  } else {
    HOLOSCAN_LOG_INFO("Bridge was never initialized, no frames were processed");
  }

  // Wait for the bus monitor thread to complete (it will exit when EOS is received)
  if (bus_monitor_) {
    HOLOSCAN_LOG_INFO("GstVideoRecorderOp::stop() - Waiting for EOS to be processed");
    auto status = bus_monitor_->get_completion_future().wait_for(kEosTimeoutSeconds);
    if (status == std::future_status::ready) {
      HOLOSCAN_LOG_INFO("EOS processed, pipeline finished cleanly");
    } else {
      HOLOSCAN_LOG_WARN("EOS not processed within {} seconds, forcing shutdown",
                        kEosTimeoutSeconds.count());
      bus_monitor_->stop();  // Force stop as fallback only
    }
  }

  // Now it's safe to set pipeline to NULL
  if (pipeline_) {
    HOLOSCAN_LOG_INFO("Setting pipeline to NULL state");
    GstStateChangeReturn ret = pipeline_.set_state(GST_STATE_NULL);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      HOLOSCAN_LOG_WARN("Failed to set pipeline to NULL state");
    }
  }

  // Cleanup pipeline.
  if (pipeline_) {
    pipeline_.reset();
  }

  HOLOSCAN_LOG_INFO("GstVideoRecorderOp::stop() - Stop complete");
}

}  // namespace holoscan
