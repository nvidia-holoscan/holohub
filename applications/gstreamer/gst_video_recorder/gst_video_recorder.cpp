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

#include <iostream>
#include <map>
#include <memory>
#include <cstdint>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <gst/gst.h>
#include <holoscan/holoscan.hpp>
#include <gst_video_recorder_operator.hpp>

namespace {

// Pattern types
enum class PatternType {
  Gradient = 0,
  Checkerboard = 1,
  ColorBars = 2
};

// Video format constants
constexpr int RGBA_CHANNELS = 4;
constexpr int RGB_CHANNELS = 3;
constexpr uint8_t ALPHA_OPAQUE = 255;

// Pattern-specific constants
constexpr int CHECKERBOARD_BASE_SIZE = 64;
constexpr int CHECKERBOARD_VARIATION = 32;
constexpr int SMPTE_COLOR_BARS = 7;

// Animation constants
constexpr float GRADIENT_TIME_STEP = 0.02f;
constexpr float CHECKERBOARD_TIME_STEP = 0.05f;

const char* get_pattern_name(int pattern) {
  const char* pattern_names[] = {"animated gradient", "animated checkerboard", "color bars"};
  return (pattern >= 0 && pattern <= 2) ? pattern_names[pattern] : "unknown";
}

// Overload for type-safe enum usage
const char* get_pattern_name(PatternType pattern) {
  return get_pattern_name(static_cast<int>(pattern));
}

/**
 * @brief Helper function to set RGBA pixel values
 *
 * @param data Pointer to pixel data buffer
 * @param idx Index of the pixel (calculated as (y * width + x) * RGBA_CHANNELS)
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 * @param a Alpha component (0-255, default is opaque)
 */
inline void set_rgba_pixel(uint8_t* data, int idx, uint8_t r, uint8_t g, uint8_t b,
                           uint8_t a = ALPHA_OPAQUE) {
  data[idx + 0] = r;
  data[idx + 1] = g;
  data[idx + 2] = b;
  data[idx + 3] = a;
}

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
};

/**
 * @brief Print usage information for the application
 *
 * @param program_name Name of the program (argv[0])
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --count <number>     Number of frames to generate (default: unlimited)\n";
    std::cout << "  -w, --width <pixels>     Frame width (default: 1920)\n";
    std::cout << "  -h, --height <pixels>    Frame height (default: 1080)\n";
    std::cout << "  -f, --framerate <rate>   Frame rate as fraction or decimal (default: 30/1)\n";
    std::cout << "                            Examples: '30/1', '30000/1001', '29.97', '60'\n";
    std::cout << "                            Use '0/1' for live mode (no throttling, "
                 "real-time timestamps)\n";
    std::cout << "  --pattern <type>         Pattern type: 0=gradient, 1=checkerboard, "
                 "2=color bars (default: 0)\n";
    std::cout << "  --storage <type>         Memory storage type: 0=host, 1=device/CUDA "
                 "(default: 1)\n";
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
    std::cout << "  --help                   Show this help message\n\n";
    std::cout << "Pipeline:\n";
    std::cout << "  The application automatically detects video parameters and selects "
                 "the appropriate converter.\n";
    std::cout << "  Parser element is automatically determined from the encoder.\n";
    std::cout << "  Muxer element is automatically determined from the file extension.\n\n";
    std::cout << "Examples:\n";
    std::cout << "  Record animated gradient with H.264:\n";
    std::cout << "    " << program_name << " --count 300 --output gradient.mp4\n\n";
    std::cout << "  Record with H.265/HEVC:\n";
    std::cout << "    " << program_name << " --count 300 --encoder nvh265 --output video.mp4\n\n";
    std::cout << "  Record to MKV container:\n";
    std::cout << "    " << program_name << " --count 300 --encoder nvh265 --output video.mkv\n\n";
    std::cout << "  Record checkerboard pattern:\n";
    std::cout << "    " << program_name << " --count 300 --pattern 1 --output checkerboard.mp4\n\n";
    std::cout << "  Custom resolution and framerate:\n";
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
      } else if ((arg == "-c" || arg == "--count" || arg == "-w" || arg == "--width" ||
                arg == "-h" || arg == "--height" || arg == "-f" || arg == "--framerate" ||
                arg == "--pattern" || arg == "--storage" || arg == "-o" || arg == "--output" ||
                arg == "-e" || arg == "--encoder" || arg == "--property") && i + 1 >= argc) {
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
 * @brief Abstract base class for pattern entity generation
 *
 * This class provides a template method pattern for generating video frame patterns.
 * Derived classes implement specific pattern generation logic.
 */
class PatternEntityGenerator {
 public:
  virtual ~PatternEntityGenerator() = default;

  /**
   * @brief Generate a pattern entity with animated content
   *
   * @param width Frame width in pixels
   * @param height Frame height in pixels
   * @param storage_type Memory storage type (host or device)
   * @param allocator Holoscan allocator for tensor memory
   * @return GXF entity containing the pattern tensor
   */
  holoscan::gxf::Entity generate(int width, int height,
                                               nvidia::gxf::MemoryStorageType storage_type,
                                               holoscan::Allocator* allocator) {
    HOLOSCAN_LOG_DEBUG("Generating {}x{} pattern entity (storage: {})",
                       width, height,
                     storage_type == nvidia::gxf::MemoryStorageType::kDevice ? "device" : "host");

  gxf_context_t context = allocator->gxf_context();

  // Create allocator handle
  auto allocator_handle =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context, allocator->gxf_cid());
  if (!allocator_handle) {
    HOLOSCAN_LOG_ERROR("Failed to create allocator handle");
    return holoscan::gxf::Entity();
  }

  // Create GXF entity with tensor using CreateTensorMap
  auto gxf_entity = nvidia::gxf::CreateTensorMap(
      context,
      allocator_handle.value(),
      {{"video_frame",
        storage_type,
        nvidia::gxf::Shape{static_cast<int32_t>(height),
                          static_cast<int32_t>(width),
                            static_cast<int32_t>(RGBA_CHANNELS)},
        nvidia::gxf::PrimitiveType::kUnsigned8,
        0,
        nvidia::gxf::ComputeTrivialStrides(
            nvidia::gxf::Shape{static_cast<int32_t>(height),
                              static_cast<int32_t>(width),
                                static_cast<int32_t>(RGBA_CHANNELS)},
            sizeof(uint8_t))}},
      false);

  if (!gxf_entity) {
    HOLOSCAN_LOG_ERROR("Failed to create GXF entity. Error code: {}",
                       static_cast<int>(gxf_entity.error()));
    return holoscan::gxf::Entity();
  }

  // Get the tensor from the entity
  auto maybe_tensor = gxf_entity.value().get<nvidia::gxf::Tensor>("video_frame");
  if (!maybe_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to get tensor from entity");
    return holoscan::gxf::Entity();
  }

    size_t buffer_size = width * height * RGBA_CHANNELS;

  // For device memory, generate pattern in host buffer first, then copy to device
  if (storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
    // Allocate temporary host buffer
    std::vector<uint8_t> host_buffer(buffer_size);

      // Generate pattern in host buffer (call derived class implementation)
      generate_pattern_data(host_buffer.data(), width, height);

    // Copy from host to device
    cudaError_t cuda_result = cudaMemcpy(
        maybe_tensor.value()->pointer(),
        host_buffer.data(),
        buffer_size,
        cudaMemcpyHostToDevice);

    if (cuda_result != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Failed to copy pattern to device memory: {}",
                        cudaGetErrorString(cuda_result));
      return holoscan::gxf::Entity();
    }
  } else {
    // Host memory - generate pattern directly
    uint8_t* data = static_cast<uint8_t*>(maybe_tensor.value()->pointer());
    if (!data) {
      HOLOSCAN_LOG_ERROR("Failed to get tensor data pointer");
      return holoscan::gxf::Entity();
    }

      // Generate pattern (call derived class implementation)
      generate_pattern_data(data, width, height);
  }

  // Wrap the GXF entity in a Holoscan entity and return
  return holoscan::gxf::Entity(std::move(gxf_entity.value()));
}

 protected:
  /**
   * @brief Pure virtual function to generate pattern data
   *
   * Derived classes must implement this to provide specific pattern generation logic.
   *
   * @param data Pointer to the buffer where pattern should be written (RGBA format)
   * @param width Frame width in pixels
   * @param height Frame height in pixels
   */
  virtual void generate_pattern_data(uint8_t* data, int width, int height) = 0;
};

/**
 * @brief Gradient pattern generator with animated sine wave colors
 */
class GradientPatternGenerator : public PatternEntityGenerator {
 public:
  GradientPatternGenerator() : time_offset_(0.0f) {}

 protected:
  void generate_pattern_data(uint8_t* data, int width, int height) override {
    time_offset_ += GRADIENT_TIME_STEP;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * RGBA_CHANNELS;
        // Animated color gradient - std::sin/cos are overloaded for float in C++
        uint8_t r = static_cast<uint8_t>(128 + 127 * std::sin(x * 0.01f + time_offset_));
        uint8_t g = static_cast<uint8_t>(128 + 127 * std::sin(y * 0.01f + time_offset_));
        uint8_t b = static_cast<uint8_t>(128 + 127 * std::cos((x + y) * 0.005f + time_offset_));
        set_rgba_pixel(data, idx, r, g, b);
      }
    }
  }

 private:
  float time_offset_;  // Animation state for gradient
};

/**
 * @brief Checkerboard pattern generator with animated square size
 */
class CheckerboardPatternGenerator : public PatternEntityGenerator {
 public:
  CheckerboardPatternGenerator() : animation_time_(0.0f) {}

 protected:
  void generate_pattern_data(uint8_t* data, int width, int height) override {
    animation_time_ += CHECKERBOARD_TIME_STEP;

    // Calculate square size with safety against division by zero
    int square_size = CHECKERBOARD_BASE_SIZE +
                      static_cast<int>(CHECKERBOARD_VARIATION * std::sin(animation_time_));
    square_size = std::max(1, square_size);  // Ensure at least 1 to avoid division by zero

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * RGBA_CHANNELS;
        bool is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
        uint8_t color = is_white ? ALPHA_OPAQUE : 0;
        set_rgba_pixel(data, idx, color, color, color);
      }
    }
  }

 private:
  float animation_time_;  // Animation state for checkerboard
};

/**
 * @brief SMPTE color bars pattern generator (static pattern)
 */
class ColorBarsPatternGenerator : public PatternEntityGenerator {
 protected:
  void generate_pattern_data(uint8_t* data, int width, int height) override {
    // SMPTE color bars (7 bars) - static to avoid recreating on each call
    static constexpr uint8_t colors[SMPTE_COLOR_BARS][RGB_CHANNELS] = {
      {255, 255, 255},  // White
      {255, 255, 0},    // Yellow
      {0, 255, 255},    // Cyan
      {0, 255, 0},      // Green
      {255, 0, 255},    // Magenta
      {255, 0, 0},      // Red
      {0, 0, 255}       // Blue
    };

    int bar_width = width / SMPTE_COLOR_BARS;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * RGBA_CHANNELS;
        int bar_idx = x / bar_width;
        if (bar_idx >= SMPTE_COLOR_BARS) bar_idx = SMPTE_COLOR_BARS - 1;

        set_rgba_pixel(data, idx,
                      colors[bar_idx][0],   // R
                      colors[bar_idx][1],   // G
                      colors[bar_idx][2]);  // B
      }
    }
  }
};

/**
 * PatternGenOperator - Generates pattern data as GXF entities with tensors
 */
class PatternGenOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PatternGenOperator)

  void setup(OperatorSpec& spec) override {
    spec.output<gxf::Entity>("output");

    spec.param(allocator_, "allocator", "Allocator", "Memory allocator for tensor allocation");
    spec.param(width_, "width", "Width", "Frame width in pixels", 1920);
    spec.param(height_, "height", "Height", "Frame height in pixels", 1080);
    spec.param(pattern_, "pattern", "Pattern",
               "Pattern type: 0=gradient, 1=checkerboard, 2=color bars", 0);
    spec.param(storage_type_, "storage_type", "StorageType",
               "Memory storage type: 0=host, 1=device", 0);
  }

  void start() override {
    // Create the appropriate pattern generator based on the pattern type
    int pattern = pattern_.get();
    switch (static_cast<PatternType>(pattern)) {
      case PatternType::Gradient:
        generator_ = std::make_unique<GradientPatternGenerator>();
        HOLOSCAN_LOG_INFO("Using animated gradient pattern generator");
        break;
      case PatternType::Checkerboard:
        generator_ = std::make_unique<CheckerboardPatternGenerator>();
        HOLOSCAN_LOG_INFO("Using animated checkerboard pattern generator");
        break;
      case PatternType::ColorBars:
        generator_ = std::make_unique<ColorBarsPatternGenerator>();
        HOLOSCAN_LOG_INFO("Using SMPTE color bars pattern generator");
        break;
      default:
        generator_ = std::make_unique<GradientPatternGenerator>();
        HOLOSCAN_LOG_WARN("Invalid pattern type {}, defaulting to gradient", pattern);
        break;
    }
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    HOLOSCAN_LOG_DEBUG("Generating pattern");

    // Convert storage_type parameter (0=host, 1=device) to MemoryStorageType enum
    nvidia::gxf::MemoryStorageType storage = (storage_type_.get() == 1)
        ? nvidia::gxf::MemoryStorageType::kDevice
        : nvidia::gxf::MemoryStorageType::kHost;

    // Generate a pattern entity with tensors using the polymorphic generator
    auto entity = generator_->generate(width_.get(), height_.get(),
                                          storage, allocator_.get().get());
    if (!entity) {
      HOLOSCAN_LOG_ERROR("Failed to generate pattern entity");
      return;
    }

    HOLOSCAN_LOG_DEBUG("Pattern entity generated, emitting to output");

    // Emit the entity to the output port
    output.emit(entity, "output");
    HOLOSCAN_LOG_DEBUG("Pattern entity emitted");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<int> width_;
  Parameter<int> height_;
  Parameter<int> pattern_;
  Parameter<int> storage_type_;
  std::unique_ptr<PatternEntityGenerator> generator_;  // Polymorphic pattern generator
};

/**
 * @brief Holoscan application that pushes generated pattern data into GStreamer
 */
class GstVideoRecorderApp : public Application {
 public:
  GstVideoRecorderApp(int64_t iteration_count, int width, int height,
            std::string framerate, int pattern, int storage_type,
            std::string filename, std::string encoder,
            std::map<std::string, std::string> properties)
    : iteration_count_(iteration_count), width_(width), height_(height),
      framerate_(std::move(framerate)), pattern_(pattern), storage_type_(storage_type),
      filename_(std::move(filename)), encoder_(std::move(encoder)),
      properties_(std::move(properties)) {}

  void compose() override {
    // Create an allocator for tensor memory
    auto allocator = make_resource<UnboundedAllocator>("allocator");

    // Create the pattern generator operator
    auto pattern_gen_op = make_operator<PatternGenOperator>(
        "pattern_gen_op",
        make_condition<CountCondition>(iteration_count_),
        Arg("allocator", allocator),
        Arg("width", width_),
        Arg("height", height_),
        Arg("pattern", pattern_),
        Arg("storage_type", storage_type_));

    // Create the GStreamer video recorder operator - it manages the pipeline internally
    // Note: width, height, format, and storage type are automatically detected from incoming frames
    auto recorder_op = make_operator<GstVideoRecorderOperator>(
        "gst_recorder_op",
        Arg("encoder", encoder_),
        Arg("framerate", framerate_),
        Arg("properties", properties_),
        Arg("max-buffers", DEFAULT_MAX_BUFFERS),
        Arg("filename", filename_));

    // Connect the operators: pattern generator -> GStreamer recorder
    add_flow(pattern_gen_op, recorder_op);
  }

 private:
  int64_t iteration_count_;
  int width_;
  int height_;
  std::string framerate_;
  int pattern_;
  int storage_type_;
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
    HOLOSCAN_LOG_INFO("Starting Holoscan Pattern to GStreamer Video Recorder");
    HOLOSCAN_LOG_INFO("Configuration: {} iterations, {}x{}@{}fps, pattern: {}, "
                      "storage: {}, encoder: {}, output: '{}'",
                      config.iteration_count, config.width, config.height,
                      config.framerate, get_pattern_name(config.pattern),
                      config.storage_type == 1 ? "device" : "host",
                      config.encoder, config.filename);
    if (!config.properties.empty()) {
      HOLOSCAN_LOG_INFO("Encoder properties: {} properties configured", config.properties.size());
      for (const auto& [key, value] : config.properties) {
        HOLOSCAN_LOG_INFO("  {} = {}", key, value);
      }
    }
    HOLOSCAN_LOG_INFO("Video parameters (width, height, format, storage) will be "
                      "auto-detected from frames");

    // Create the Holoscan application with parsed configuration
    auto holoscan_app = std::make_shared<holoscan::GstVideoRecorderApp>(
        config.iteration_count, config.width, config.height,
        std::move(config.framerate), config.pattern, config.storage_type,
        std::move(config.filename), std::move(config.encoder),
        std::move(config.properties));

    // Run the Holoscan application - the operator manages the GStreamer pipeline internally
    holoscan_app->run();

    HOLOSCAN_LOG_INFO("Application finished");
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }
  return 0;
}
