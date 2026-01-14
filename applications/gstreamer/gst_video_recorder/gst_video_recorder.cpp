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

#include <iostream>
#include <map>
#include <memory>
#include <cstdint>

#include <gst/gst.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <gst_video_recorder_op.hpp>
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
 * @brief Configuration parameters for the video recorder application
 */
struct AppConfig {
  int64_t iteration_count = INT64_MAX;  // Default: run forever
  std::string filename = "output.mp4";
  std::string encoder = "nvh264";
  int width = 1920;
  int height = 1080;
  std::string framerate = "30/1";
  int pattern = 0;  // 0=gradient, 1=checkerboard, 2=color bars
  int storage_type = 1;  // 0=host, 1=device (default to device for CUDA pipeline)
  std::map<std::string, std::string> properties;  // Encoder properties

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
    std::cout << "Options:\n";
    std::cout << "  --source <type>          Video source: pattern or v4l2 (default: pattern)\n";
    std::cout << "  -c, --count <number>     Number of frames to capture (default: unlimited)\n";
    std::cout << "  -w, --width <pixels>     Frame width (default: 1920)\n";
    std::cout << "  -h, --height <pixels>    Frame height (default: 1080)\n";
    std::cout << "  -f, --framerate <rate>   Frame rate as fraction or decimal (default: 30/1)\n";
    std::cout << "                            Examples: '30/1', '30000/1001', '29.97', '60'\n";
    std::cout << "                            Use '0/1' for live mode (no throttling, "
                 "real-time timestamps)\n";
    std::cout << "  --pattern <type>         Pattern type: 0=gradient, 1=checkerboard, "
                 "2=color bars (default: 0)\n";
    std::cout << "                            Only used when source=pattern\n";
    std::cout << "  --storage <type>         Memory storage type: 0=host, 1=device/CUDA "
                 "(default: 1)\n";
    std::cout << "                            Only used when source=pattern\n";
    std::cout << "  -o, --output <filename>  Output video filename (default: output.mp4)\n";
    std::cout << "                            Supported formats: .mp4, .mkv\n";
    std::cout << "                            If no extension, defaults to .mp4\n";
    std::cout << "  -e, --encoder <name>     Encoder base name (default: nvh264)\n";
    std::cout << "                            Examples: nvh264, nvh265, x264, x265\n";
    std::cout << "                            Note: 'enc' suffix is automatically appended\n";
    std::cout << "  --property <key=value>   Set encoder property (can be used multiple times)\n";
    std::cout << "                            Examples: --property bitrate=8000 "
                 "--property preset=1\n";
    std::cout << "                            Property types are automatically detected "
                 "and converted\n";
    std::cout << "\n";
    std::cout << "V4L2 Camera Options:\n";
    std::cout << "  --device <path>          V4L2 device path (default: /dev/video0)\n";
    std::cout << "  --pixel-format <format>  V4L2 pixel format (default: auto)\n";
    std::cout << "                            Examples: YUYV, MJPEG, auto\n";
    std::cout << "  --help                   Show this help message\n\n";
    std::cout << "Pipeline:\n";
    std::cout << "  The application automatically detects video parameters and selects "
                 "the appropriate converter.\n";
    std::cout << "  Parser element is automatically determined from the encoder.\n";
    std::cout << "  Muxer element is automatically determined from the file extension.\n\n";
    std::cout << "Examples:\n";
    std::cout << "  Record from V4L2 camera:\n";
    std::cout << "    " << program_name << " --source v4l2 --count 300 --output camera.mp4\n\n";
    std::cout << "  Record from V4L2 camera with specific device and format:\n";
    std::cout << "    " << program_name << " --source v4l2 --device /dev/video1 "
                 "--pixel-format YUYV --output camera.mp4\n\n";
    std::cout << "  Record animated gradient with H.264:\n";
    std::cout << "    " << program_name << " --source pattern --count 300 "
                 "--output gradient.mp4\n\n";
    std::cout << "  Record with H.265/HEVC:\n";
    std::cout << "    " << program_name << " --count 300 --encoder nvh265 --output video.mp4\n\n";
    std::cout << "  Record to MKV container:\n";
    std::cout << "    " << program_name << " --count 300 --encoder nvh265 --output video.mkv\n\n";
    std::cout << "  Record checkerboard pattern:\n";
    std::cout << "    " << program_name << " --count 300 --pattern 1 --output checkerboard.mp4\n\n";
    std::cout << "  Custom resolution and framerate (pattern only):\n";
    std::cout << "    " << program_name << " --count 300 --width 1280 --height 720 "
                 "--framerate 60 --output hd.mp4\n\n";
    std::cout << "  Record with NTSC framerate (exact 30000/1001):\n";
    std::cout << "    " << program_name << " --count 300 --framerate 30000/1001 "
                 "--output ntsc.mp4\n\n";
    std::cout << "  Live mode (no throttling, real-time timestamps):\n";
    std::cout << "    " << program_name << " --count 300 --framerate 0/1 --output live.mp4\n\n";
    std::cout << "  Use CPU encoder (x264) with host memory:\n";
    std::cout << "    " << program_name << " --count 300 --storage 0 --encoder x264 "
                 "--output cpu.mp4\n\n";
    std::cout << "  Custom encoder properties (bitrate, preset, GOP size):\n";
    std::cout << "    " << program_name << " --count 300 --property bitrate=8000 "
                 "--property preset=1 --property gop-size=30 --output custom.mp4\n";
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
        config.framerate = argv[++i];
        // Basic validation: must not be empty
        if (config.framerate.empty()) {
          throw std::invalid_argument("Framerate cannot be empty");
        }
      } else if (arg == "--pattern" && i + 1 < argc) {
        // Validate: 0=gradient, 1=checkerboard, 2=color bars
        config.pattern = parse_validated<int>(argv[++i], "pattern", 0, 2);
      } else if (arg == "--storage" && i + 1 < argc) {
        // Validate: 0=host, 1=device
        config.storage_type = parse_validated<int>(argv[++i], "storage type", 0, 1);
      } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
        config.filename = argv[++i];
        if (config.filename.empty()) {
          throw std::invalid_argument("Output filename cannot be empty");
        }
      } else if ((arg == "-e" || arg == "--encoder") && i + 1 < argc) {
        config.encoder = argv[++i];
        if (config.encoder.empty()) {
          throw std::invalid_argument("Encoder name cannot be empty");
        }
      } else if (arg == "--property" && i + 1 < argc) {
        std::string prop = argv[++i];
        size_t eq_pos = prop.find('=');
        if (eq_pos == std::string::npos) {
          throw std::invalid_argument(
              "Invalid property format '" + prop + "' (expected key=value)");
        }
        if (eq_pos == 0) {
          throw std::invalid_argument("Property key cannot be empty");
        }
        if (eq_pos == prop.length() - 1) {
          throw std::invalid_argument("Property value cannot be empty");
        }

        std::string key = prop.substr(0, eq_pos);
        std::string value = prop.substr(eq_pos + 1);
        config.properties[key] = value;
      } else if ((arg == "--source" || arg == "-c" || arg == "--count" ||
                arg == "-w" || arg == "--width" || arg == "-h" || arg == "--height" ||
                arg == "-f" || arg == "--framerate" || arg == "--pattern" ||
                arg == "--storage" || arg == "-o" || arg == "--output" ||
                arg == "-e" || arg == "--encoder" || arg == "--property" ||
                arg == "--device" || arg == "--pixel-format") && i + 1 >= argc) {
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
constexpr size_t DEFAULT_MAX_BUFFERS = 10;  // Default max buffers for GStreamer recorder

/**
 * @brief Holoscan application that records video from pattern generator or V4L2 camera
 */
class GstVideoRecorderApp : public Application {
 public:
  explicit GstVideoRecorderApp(const AppConfig& config)
    : iteration_count_(config.iteration_count),
      source_(config.source),
      width_(config.width),
      height_(config.height),
      framerate_(config.framerate),
      pattern_(config.pattern),
      storage_type_(config.storage_type),
      v4l2_device_(config.v4l2_device),
      v4l2_pixel_format_(config.v4l2_pixel_format),
      filename_(config.filename),
      encoder_(config.encoder),
      properties_(config.properties) {}

  void compose() override {
    using namespace holoscan;

    // Create an allocator for tensor memory
    auto allocator = make_resource<UnboundedAllocator>("allocator");

    // Create the video source operator based on source selection
    std::shared_ptr<Operator> source_op;
    std::string source_output_port;

    // Create the GStreamer video recorder operator (common to both sources)
    auto recorder_op = make_operator<GstVideoRecorderOp>(
        "gst_recorder_op",
        Arg("encoder", encoder_),
        Arg("framerate", framerate_),
        Arg("properties", properties_),
        Arg("max-buffers", DEFAULT_MAX_BUFFERS),
        Arg("filename", filename_));

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
      // V4L2 outputs RGBA8888, we keep it as-is for the recorder
      auto format_converter = make_operator<ops::FormatConverterOp>(
          "format_converter",
          Arg("in_dtype", std::string("rgba8888")),
          Arg("out_dtype", std::string("rgba8888")),
          Arg("pool", allocator));

      // Connect: V4L2 -> FormatConverter -> Recorder
      add_flow(source_op, format_converter, {{source_output_port, "source_video"}});
      add_flow(format_converter, recorder_op, {{"tensor", "input"}});
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

      // Connect: Pattern -> Recorder (direct)
      add_flow(source_op, recorder_op, {{source_output_port, "input"}});
    }
  }

 private:
  int64_t iteration_count_;
  std::string source_;
  int width_;
  int height_;
  std::string framerate_;
  int pattern_;
  int storage_type_;
  std::string v4l2_device_;
  std::string v4l2_pixel_format_;
  std::string filename_;
  std::string encoder_;
  std::map<std::string, std::string> properties_;
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
    HOLOSCAN_LOG_INFO("Starting Holoscan Video Recorder with GStreamer");
    HOLOSCAN_LOG_INFO("Source: {}", config.source);
    HOLOSCAN_LOG_INFO("Configuration: {} iterations, {}fps, encoder: {}, output: '{}'",
                      config.iteration_count, config.framerate, config.encoder, config.filename);

    if (config.source == "v4l2") {
      HOLOSCAN_LOG_INFO("V4L2 camera: device={}, pixel_format={}",
                        config.v4l2_device, config.v4l2_pixel_format);
    } else {
      HOLOSCAN_LOG_INFO("Pattern: {}x{}, type: {}, storage: {}",
                        config.width, config.height,
                        holoscan::get_pattern_name(config.pattern),
                        config.storage_type == 1 ? "device" : "host");
    }

    if (!config.properties.empty()) {
      HOLOSCAN_LOG_INFO("Encoder properties: {} properties configured", config.properties.size());
      for (const auto& [key, value] : config.properties) {
        HOLOSCAN_LOG_INFO("  {} = {}", key, value);
      }
    }
    HOLOSCAN_LOG_INFO("Video parameters (width, height, format, storage) will be "
                      "auto-detected from frames");

    // Create the Holoscan application with parsed configuration
    auto holoscan_app = std::make_shared<holoscan::GstVideoRecorderApp>(config);

    // Run the Holoscan application - the operator manages the GStreamer pipeline internally
    holoscan_app->run();

    HOLOSCAN_LOG_INFO("Application finished");
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }
  return 0;
}
