/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdint>

#include <gst/gst.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <gst_src_resource.hpp>
#include <gst_src_op.hpp>
#include <gst/pipeline.hpp>
#include <gst_pipeline_bus_monitor.hpp>
#include "pattern_generator.hpp"

namespace {

/**
 * @brief Convert string to numeric type (template specializations)
 *
 * @tparam T Numeric type to convert to
 * @param str String to convert
 * @param pos Pointer to size_t to store position after conversion
 * @return Converted value
 * @throws std::invalid_argument or std::out_of_range on conversion failure
 */
template<typename T>
T string_to(const std::string& str, size_t* pos);

// Specialization for int
template<>
int string_to<int>(const std::string& str, size_t* pos) {
  return std::stoi(str, pos);
}

// Specialization for int64_t
template<>
int64_t string_to<int64_t>(const std::string& str, size_t* pos) {
  return std::stoll(str, pos);
}

/**
 * @brief Safely parse a numeric value with validation
 *
 * @tparam T Numeric type to parse (int, int64_t, etc.)
 * @param value_str String to parse
 * @param param_name Parameter name for error messages
 * @param min_value Minimum allowed value (inclusive)
 * @param max_value Maximum allowed value (inclusive)
 * @return Parsed and validated value
 * @throws std::invalid_argument if parsing fails or value is out of range
 */
template<typename T>
T parse_validated(const std::string& value_str, const std::string& param_name,
                  T min_value, T max_value) {
  try {
    size_t pos = 0;
    T value = string_to<T>(value_str, &pos);

    // Check if entire string was consumed
    if (pos != value_str.length()) {
      throw std::invalid_argument("Invalid characters in value");
    }

    // Validate range
    if (value < min_value || value > max_value) {
      throw std::invalid_argument(
          "Value " + std::to_string(value) + " is out of range [" +
          std::to_string(min_value) + ", " + std::to_string(max_value) + "]");
    }

    return value;
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(
        "Invalid " + param_name + ": " + value_str + " (" + e.what() + ")");
  } catch (const std::out_of_range& e) {
    throw std::invalid_argument(
        param_name + " value is too large: " + value_str);
  }
}

/**
 * @brief Configuration parameters for the Holoscan to GStreamer bridge
 */
struct AppConfig {
  int64_t iteration_count = INT64_MAX;  // Default: run forever
  std::string pipeline_desc = "cudadownload name=first ! videoconvert ! autovideosink sync=false";
  std::string caps = "video/x-raw,format=RGBA";
  int width = 1920;
  int height = 1080;
  int framerate = 30;
  int pattern = 0;  // 0=gradient, 1=checkerboard, 2=color bars
  int storage_type = 1;  // 0=host, 1=device (default to device for CUDA pipeline)

  // Source selection
  std::string source = "pattern";  // "pattern" or "v4l2"

  // V4L2 camera configuration
  std::string v4l2_device = "/dev/video0";
  std::string v4l2_pixel_format = "auto";  // "YUYV", "MJPEG", "auto", etc.
};

/**
 * @brief Print usage information for the application
 *
 * @param program_name Name of the program (argv[0])
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Note: When running through holohub, use modes instead:\n";
    std::cout << "  ./holohub run holo_to_gst v4l2      # For V4L2 camera\n";
    std::cout << "  ./holohub run holo_to_gst pattern   # For test patterns\n";
    std::cout << "The --source option below is for direct binary execution.\n\n";
    std::cout << "Options:\n";
    std::cout << "  --source <type>          Video source: pattern or v4l2 "
                 "(default: pattern)\n";
    std::cout << "                            Note: Use holohub modes "
                 "(v4l2/pattern) instead when possible\n";
    std::cout << "  -c, --count <number>     Number of frames to "
                 "capture/generate (default: unlimited)\n";
    std::cout << "  -w, --width <pixels>     Frame width (default: 1920)\n";
    std::cout << "  -h, --height <pixels>    Frame height (default: 1080)\n";
    std::cout << "  -f, --framerate <fps>    Frame rate in frames per second (default: 30)\n";
    std::cout << "  --pattern <type>         Pattern type: 0=gradient, 1=checkerboard, "
                 "2=color bars (default: 0)\n";
    std::cout << "                            Only used when source=pattern\n";
    std::cout << "  --storage <type>         Memory storage type: 0=host, 1=device/CUDA "
                 "(default: 1)\n";
    std::cout << "                            Only used when source=pattern\n";
    std::cout << "  -p, --pipeline <desc>    GStreamer pipeline description\n";
    std::cout << "                            (default: cudadownload name=first ! "
                 "videoconvert ! autovideosink sync=false)\n";
    std::cout << "                            IMPORTANT: Your pipeline MUST name the "
                 "first element as 'first'\n";
    std::cout << "  --caps <caps_string>     GStreamer capabilities string for the source\n";
    std::cout << "                            (default: video/x-raw,format=RGBA)\n";
    std::cout << "  --help                   Show this help message\n\n";
    std::cout << "V4L2 Camera Options:\n";
    std::cout << "  --device <path>          V4L2 device path "
                 "(default: /dev/video0)\n";
    std::cout << "                            Use 'v4l2-ctl --list-devices' to "
                 "find your camera\n";
    std::cout << "  --pixel-format <format>  V4L2 pixel format as 4-character "
                 "FourCC code (default: auto)\n";
    std::cout << "                            Examples: YUYV, MJPG, NV12, auto\n";
    std::cout << "                            Note: MJPG supports higher resolutions "
                 "than uncompressed formats\n\n";
    std::cout << "Pipeline Requirements:\n";
    std::cout << "  - The first element in your pipeline MUST be "
                 "named 'first'\n";
    std::cout << "  - You can construct ANY GStreamer pipeline using "
                 "1000+ available plugins\n";
    std::cout << "  - The examples shown are just starting points for "
                 "customization\n\n";
    std::cout << "File Output:\n";
    std::cout << "  - When running in container (via holohub), use absolute paths:\n";
    std::cout << "    location=/workspace/holohub/output.mp4\n";
    std::cout << "  - Files will be accessible on host at your workspace root\n\n";
    std::cout << "Examples:\n";
    std::cout << "  Display animated gradient (default):\n";
    std::cout << "    " << program_name << "\n\n";
    std::cout << "  Display from V4L2 camera:\n";
    std::cout << "    " << program_name << " --source v4l2 --count 300\n\n";
    std::cout << "  Display from V4L2 camera with 720p resolution:\n";
    std::cout << "    " << program_name
                 << " --source v4l2 --width 1280 --height 720\n\n";
    std::cout << "  Record 4K video from V4L2 camera (requires MJPG format):\n";
    std::cout << "    " << program_name
                 << " --source v4l2 --width 3840 --height 2160 "
                 "--pixel-format MJPG --count 300 --pipeline "
                 "\"cudaconvert name=first ! nvh264enc ! h264parse ! mp4mux ! "
                 "filesink location=output_4k.mp4\"\n\n";
    std::cout << "  Display checkerboard pattern:\n";
    std::cout << "    " << program_name << " --pattern 1\n\n";
    std::cout << "  Display color bars:\n";
    std::cout << "    " << program_name << " --pattern 2\n\n";
    std::cout << "  Custom resolution and framerate:\n";
    std::cout << "    " << program_name
                 << " --width 1280 --height 720 --framerate 60\n\n";
    std::cout << "  Save pattern to file (CPU-based encoding):\n";
    std::cout << "    " << program_name << " --count 300 --pipeline "
                 "\"cudadownload name=first ! videoconvert ! x264enc ! mp4mux ! "
                 "filesink location=/workspace/holohub/output.mp4\"\n\n";
    std::cout << "  Save pattern to file (GPU-based encoding):\n";
    std::cout << "    " << program_name << " --count 300 --pipeline "
                 "\"cudaconvert name=first ! nvh264enc ! h264parse ! mp4mux ! "
                 "filesink location=/workspace/holohub/output.mp4\"\n\n";
    std::cout << "  Record from V4L2 camera to file (1080p):\n";
    std::cout << "    " << program_name << " --source v4l2 --count 300 "
                 "--pipeline \"cudadownload name=first ! videoconvert ! "
                 "x264enc ! mp4mux ! filesink "
                 "location=/workspace/holohub/camera.mp4\"\n\n";
    std::cout << "  Stream over network (RTP):\n";
    std::cout << "    " << program_name << " --pipeline "
                 "\"cudaconvert name=first ! nvh264enc ! h264parse ! rtph264pay ! "
                 "udpsink host=127.0.0.1 port=5000\"\n";
    std::cout << "  Receive stream: gst-launch-1.0 udpsrc port=5000 "
                 "caps=\\\"application/x-rtp,encoding-name=H264\\\" ! "
                 "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
                 "autovideosink\n";
}

/**
 * @brief Parse command-line arguments into application configuration
 *
 * @param argc Argument count
 * @param argv Argument values
 * @param config Output parameter for parsed configuration
 * @return true if parsing succeeded, false on error (prints error message internally)
 */
bool parse_arguments(int argc, char** argv, AppConfig& config) {
  try {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "--help") {
        print_usage(argv[0]);
        std::exit(0);  // Exit successfully after showing help
      } else if (arg == "--source" && i + 1 < argc) {
        config.source = argv[++i];
        if (config.source != "pattern" && config.source != "v4l2") {
          throw std::invalid_argument(
              "Invalid source '" + config.source + "' (must be 'pattern' or 'v4l2')");
        }
      } else if (arg == "--device" && i + 1 < argc) {
        config.v4l2_device = argv[++i];
        if (config.v4l2_device.empty()) {
          throw std::invalid_argument("V4L2 device path cannot be empty");
        }
      } else if (arg == "--pixel-format" && i + 1 < argc) {
        config.v4l2_pixel_format = argv[++i];
        if (config.v4l2_pixel_format.empty()) {
          throw std::invalid_argument("Pixel format cannot be empty");
        }
      } else if ((arg == "-c" || arg == "--count") && i + 1 < argc) {
        // Validate: 1 to 1 billion frames
        config.iteration_count = parse_validated<int64_t>(argv[++i], "frame count", 1, 1000000000);
      } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
        // Validate: 64 to 8192 pixels (reasonable video width range)
        config.width = parse_validated<int>(argv[++i], "width", 64, 8192);
      } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
        // Validate: 64 to 8192 pixels (reasonable video height range)
        config.height = parse_validated<int>(argv[++i], "height", 64, 8192);
      } else if ((arg == "-f" || arg == "--framerate") && i + 1 < argc) {
        // Validate: 1 to 240 fps
        config.framerate = parse_validated<int>(argv[++i], "framerate", 1, 240);
      } else if (arg == "--pattern" && i + 1 < argc) {
        // Validate: 0=gradient, 1=checkerboard, 2=color bars
        config.pattern = parse_validated<int>(argv[++i], "pattern", 0, 2);
      } else if (arg == "--storage" && i + 1 < argc) {
        // Validate: 0=host, 1=device
        config.storage_type = parse_validated<int>(argv[++i], "storage type", 0, 1);
      } else if ((arg == "-p" || arg == "--pipeline") && i + 1 < argc) {
        config.pipeline_desc = argv[++i];
        if (config.pipeline_desc.empty()) {
          throw std::invalid_argument("Pipeline description cannot be empty");
        }
      } else if (arg == "--caps" && i + 1 < argc) {
        config.caps = argv[++i];
        if (config.caps.empty()) {
          throw std::invalid_argument("Caps string cannot be empty");
        }
      } else if ((arg == "--source" || arg == "-c" || arg == "--count" ||
                arg == "-w" || arg == "--width" || arg == "-h" || arg == "--height" ||
                arg == "-f" || arg == "--framerate" || arg == "--pattern" ||
                arg == "--storage" || arg == "-p" || arg == "--pipeline" ||
                arg == "--caps" || arg == "--device" || arg == "--pixel-format") && i + 1 >= argc) {
        throw std::invalid_argument("Missing value for argument: " + arg);
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    return true;  // Success
  } catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl;
    print_usage(argv[0]);
    return false;  // Failure
  }
}

}  // namespace

namespace holoscan {

// Constants
constexpr size_t DEFAULT_MAX_BUFFERS = 10;  // Default max buffers for GStreamer source

/**
 * @brief Custom pipeline bus monitor that sends SIGINT on error or window close
 */
class AppPipelineBusMonitor : public holoscan::gst::PipelineBusMonitor {
 public:
  using holoscan::gst::PipelineBusMonitor::PipelineBusMonitor;

 protected:
  void on_error(const holoscan::gst::Error& error, const std::string& debug_info) override {
    // Call base implementation for logging
    holoscan::gst::PipelineBusMonitor::on_error(error, debug_info);

    // Send SIGINT to interrupt the application
    HOLOSCAN_LOG_INFO("Sending interrupt signal due to GStreamer error");
    std::raise(SIGINT);
  }

  void on_state_changed(GstState old_state, GstState new_state, GstState pending_state) override {
    // Call base implementation
    holoscan::gst::PipelineBusMonitor::on_state_changed(old_state, new_state, pending_state);

    // If pipeline transitions to NULL unexpectedly, send SIGINT
    if (new_state == GST_STATE_NULL && old_state != GST_STATE_NULL) {
      HOLOSCAN_LOG_INFO("GStreamer window closed");
      HOLOSCAN_LOG_INFO("Sending interrupt signal due to window closure");
      std::raise(SIGINT);
    }
  }
};

/**
 * @brief GStreamerApp - Manages GStreamer pipeline lifecycle
 *
 * Encapsulates all GStreamer pipeline operations for consuming from holoscansrc
 */
class GStreamerApp {
 public:
  GStreamerApp(const std::string& pipeline_desc,
               holoscan::gst::Element src_element)
    : pipeline_desc_(pipeline_desc),
      src_element_(src_element) {
    HOLOSCAN_LOG_INFO("Setting up GStreamer pipeline");

    // Validate source element
    if (!src_element_ || !src_element_.get()) {
      throw std::runtime_error("Invalid source element");
    }

    // Parse the sink pipeline
    GError* error = nullptr;
    pipeline_ = holoscan::gst::Pipeline(GST_PIPELINE(
        gst_parse_launch(pipeline_desc_.c_str(), &error)));
    if (error) {
      auto error_object = holoscan::gst::Error(error);
      HOLOSCAN_LOG_ERROR("Failed to parse pipeline: {}", error_object->message);
      throw std::runtime_error("Failed to parse GStreamer pipeline description");
    }
    pipeline_.ref_sink();

    // Add src element to pipeline
    // Note: gst_bin_add() properly handles reference counting - it adds a new reference
    // when the element doesn't have a floating reference (which was already sunk in GstSrcBridge).
    pipeline_.add(src_element_);

    // Find and link the "first" element
    // Note: gst_bin_get_by_name returns a new reference, so wrap it in a guard
    auto first_element = holoscan::gst::Element(
        gst_bin_get_by_name(GST_BIN(pipeline_.get()), "first"));
    if (!first_element) {
      HOLOSCAN_LOG_ERROR("Could not find element named 'first' in pipeline");
      HOLOSCAN_LOG_ERROR("Please name your first pipeline element as 'first', "
                         "e.g.: 'videoconvert name=first'");
      throw std::runtime_error("Could not find element named 'first' to "
                               "connect from source");
    }

    HOLOSCAN_LOG_INFO("Linking source to {}",
                      gst_element_get_name(first_element.get()));

    if (!gst_element_link(src_element_.get(), first_element.get())) {
      HOLOSCAN_LOG_ERROR("Failed to link source to {}",
                         gst_element_get_name(first_element.get()));
      throw std::runtime_error("Failed to link source to pipeline");
    }

    HOLOSCAN_LOG_INFO("Pipeline setup complete");

    // Start the GStreamer pipeline
    if (pipeline_.set_state(GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
      HOLOSCAN_LOG_ERROR("Failed to start GStreamer pipeline");
      throw std::runtime_error("Failed to start GStreamer pipeline");
    }

    HOLOSCAN_LOG_INFO("GStreamer pipeline started");

    // Check pipeline state (non-fatal - pipeline may not reach PLAYING until data flows)
    // This is normal for appsrc-based pipelines, especially with file sinks
    GstState state;
    auto state_result = pipeline_.get_state(&state, nullptr, 2 * GST_SECOND);
    if (state_result == GST_STATE_CHANGE_ASYNC || state == GST_STATE_PLAYING) {
      HOLOSCAN_LOG_INFO("GStreamer pipeline is PLAYING or transitioning to "
                        "PLAYING");
    } else {
      HOLOSCAN_LOG_WARN("Pipeline not yet in PLAYING state (current: {}), "
                        "will transition as data flows",
                        gst_element_state_get_name(state));
    }

    // Start bus monitoring in a background thread
    bus_monitor_ = std::make_unique<AppPipelineBusMonitor>(pipeline_);
    bus_monitor_->start();
  }

  ~GStreamerApp() {
    // Stop bus monitoring
    if (bus_monitor_) {
      bus_monitor_->stop();
    }

    // Stop and cleanup pipeline
    if (pipeline_) {
      pipeline_.set_state(GST_STATE_NULL);
      pipeline_.reset();
    }
  }

  // Delete copy constructor and assignment
  GStreamerApp(const GStreamerApp&) = delete;
  GStreamerApp& operator=(const GStreamerApp&) = delete;

  std::shared_future<void> get_bus_monitor_future() {
    return bus_monitor_->get_completion_future();
  }

 private:
  std::string pipeline_desc_;
  holoscan::gst::Element src_element_;
  holoscan::gst::Pipeline pipeline_;
  std::unique_ptr<AppPipelineBusMonitor> bus_monitor_;
};

/**
 * @brief Holoscan application that pushes data from pattern generator or V4L2 camera into GStreamer
 */
class GstSrcApp : public Application {
 public:
  explicit GstSrcApp(const AppConfig& config)
    : iteration_count_(config.iteration_count),
      caps_(config.caps),
      width_(config.width),
      height_(config.height),
      framerate_(config.framerate),
      pattern_(config.pattern),
      storage_type_(config.storage_type),
      source_(config.source),
      v4l2_device_(config.v4l2_device),
      v4l2_pixel_format_(config.v4l2_pixel_format) {}

  void compose() override {
    // Build caps string with actual width, height, framerate, and memory type
    // Memory feature must come right after media type: video/x-raw(memory:CUDAMemory)
    std::string full_caps;

    if (storage_type_ == 1) {
      // CUDA memory: insert memory feature after media type
      full_caps = "video/x-raw(memory:CUDAMemory),format=RGBA";
    } else {
      // Host memory: use default caps
      full_caps = caps_;
    }

    full_caps += ",width=" + std::to_string(width_) +
                 ",height=" + std::to_string(height_) +
                 ",framerate=" + std::to_string(framerate_) + "/1";

    HOLOSCAN_LOG_INFO("Source caps: {}", full_caps);

    // Create the GStreamer source resource for data bridging
    holoscan_gst_src_ = make_resource<GstSrcResource>("holoscan_src",
        Arg("caps", full_caps),
        Arg("max_buffers", DEFAULT_MAX_BUFFERS));

    // Create an allocator for tensor memory
    auto allocator = make_resource<UnboundedAllocator>("allocator");

    // Create the video source operator based on source selection
    std::shared_ptr<Operator> source_op;
    std::string source_output_port;

    // Create the GStreamer source operator (common to both sources)
    auto gst_src_op = make_operator<GstSrcOp>(
        "gst_src_op",
        Arg("gst_src_resource", holoscan_gst_src_));

    if (source_ == "v4l2") {
      // Create V4L2 camera capture operator
      // Note: V4L2VideoCaptureOp expects width/height as uint32_t
      source_op = make_operator<ops::V4L2VideoCaptureOp>(
          "v4l2_source",
          make_condition<CountCondition>(iteration_count_),
          Arg("allocator", allocator),
          Arg("device", v4l2_device_),
          Arg("width", static_cast<uint32_t>(width_)),
          Arg("height", static_cast<uint32_t>(height_)),
          Arg("pixel_format", v4l2_pixel_format_));
      source_output_port = "signal";

      HOLOSCAN_LOG_INFO("Using V4L2 camera source: device={}, {}x{}, pixel_format={}",
                        v4l2_device_, width_, height_, v4l2_pixel_format_);

      // V4L2 requires a FormatConverterOp to properly convert the camera output
      // V4L2 outputs RGBA8888, we keep it as-is for GStreamer
      auto format_converter = make_operator<ops::FormatConverterOp>(
          "format_converter",
          Arg("in_dtype", std::string("rgba8888")),
          Arg("out_dtype", std::string("rgba8888")),
          Arg("pool", allocator));

      // Connect: V4L2 -> FormatConverter -> GstSrc
      add_flow(source_op, format_converter, {{source_output_port, "source_video"}});
      add_flow(format_converter, gst_src_op, {{"tensor", "input"}});
    } else {
      // Create pattern generator operator
      source_op = make_operator<PatternGenOperator>(
          "pattern_gen_op",
          make_condition<CountCondition>(iteration_count_),
          Arg("allocator", allocator),
          Arg("width", width_),
          Arg("height", height_),
          Arg("pattern", pattern_),
          Arg("storage_type", storage_type_));
      source_output_port = "output";

      HOLOSCAN_LOG_INFO("Using pattern generator source: {}x{}, pattern={}",
                        width_, height_, get_pattern_name(pattern_));

      // Connect: Pattern -> GstSrc (direct)
      add_flow(source_op, gst_src_op, {{source_output_port, "input"}});
    }
  }

  std::shared_ptr<GstSrcResource> get_src_resource() const {
    return holoscan_gst_src_;
  }

 private:
  int64_t iteration_count_;
  std::string caps_;
  int width_;
  int height_;
  int framerate_;
  int pattern_;
  int storage_type_;
  std::string source_;
  std::string v4l2_device_;
  std::string v4l2_pixel_format_;
  std::shared_ptr<GstSrcResource> holoscan_gst_src_;
};

}  // namespace holoscan

int main(int argc, char** argv) {
  // Parse command-line arguments
  AppConfig config;
  if (!parse_arguments(argc, argv, config)) {
    return 1;  // Error already printed by parse_arguments
  }

  try {
    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Log configuration before creating app
    HOLOSCAN_LOG_INFO("Starting Holoscan to GStreamer Bridge");
    HOLOSCAN_LOG_INFO("Source: {}", config.source);

    if (config.source == "v4l2") {
      HOLOSCAN_LOG_INFO("V4L2 camera: device={}, pixel_format={}",
                        config.v4l2_device, config.v4l2_pixel_format);
    } else {
      HOLOSCAN_LOG_INFO("Pattern: {}x{}, type: {}, storage: {}",
                        config.width, config.height,
                        holoscan::get_pattern_name(config.pattern),
                        config.storage_type == 1 ? "device" : "host");
    }

    HOLOSCAN_LOG_INFO("Configuration: {} iterations, {}x{}@{}fps, "
                      "pipeline: '{}', caps: '{}'",
                      config.iteration_count, config.width, config.height,
                      config.framerate, config.pipeline_desc, config.caps);
    HOLOSCAN_LOG_INFO("This will push data from Holoscan into "
                      "GStreamer");

    // Create the Holoscan application with parsed configuration
    auto holoscan_app = std::make_shared<holoscan::GstSrcApp>(config);

    // Run the Holoscan application asynchronously
    auto app_future = holoscan_app->run_async();

    // Wait for the source element to be initialized
    auto src_element_future =
        holoscan_app->get_src_resource()->get_gst_element();
    if (!src_element_future.valid()) {
      throw std::runtime_error("Source element future is invalid");
    }

    // Wait for the element to be ready with timeout
    if (src_element_future.wait_for(std::chrono::seconds(1)) !=
        std::future_status::ready) {
      throw std::runtime_error("Timeout waiting for source element "
                               "initialization");
    }

    holoscan::gst::Element src_element = src_element_future.get();
    if (!src_element) {
      throw std::runtime_error("Failed to get initialized source element");
    }

    // Create the GStreamer application with the source element and start it
    auto gstreamer_app = std::make_shared<holoscan::GStreamerApp>(
        config.pipeline_desc, src_element);

    // Wait for Holoscan to finish generating frames
    app_future.wait();
    HOLOSCAN_LOG_INFO("Holoscan frame generation complete");

    // Wait for GStreamer to finish processing (EOS message on bus)
    HOLOSCAN_LOG_INFO("Waiting for GStreamer pipeline to finish");
    gstreamer_app->get_bus_monitor_future().wait();

    // Clean up GStreamer first (this stops the bus monitor and pipeline)
    // This must be done BEFORE Holoscan app goes out of scope to avoid
    // threading issues with the bus monitor accessing deactivated entities
    HOLOSCAN_LOG_INFO("Stopping GStreamer pipeline...");
    gstreamer_app.reset();

    // Give additional time for complete cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    HOLOSCAN_LOG_INFO("Application finished");
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }

  return 0;
}
