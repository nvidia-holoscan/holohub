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

const char* get_pattern_name(int pattern) {
  const char* pattern_names[] = {"animated gradient", "animated checkerboard", "color bars"};
  return (pattern >= 0 && pattern <= 2) ? pattern_names[pattern] : "unknown";
}

void generate_gradient_pattern(uint8_t* data, int width, int height) {
  static float time_offset = 0.0f;
  time_offset += 0.02f;
  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 4;
      // Animated color gradient
      data[idx + 0] = static_cast<uint8_t>(128 + 127 * std::sin(x * 0.01f + time_offset));  // R
      data[idx + 1] = static_cast<uint8_t>(128 + 127 * std::sin(y * 0.01f + time_offset));  // G
      data[idx + 2] = static_cast<uint8_t>(128 + 127 * std::cos((x + y) * 0.005f + time_offset));  // B
      data[idx + 3] = 255;  // A (fully opaque)
    }
  }
}

void generate_checkerboard_pattern(uint8_t* data, int width, int height) {
  static float animation_time = 0.0f;
  animation_time += 0.05f;
  
  int square_size = 64 + static_cast<int>(32 * std::sin(animation_time));
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 4;
      bool is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
      uint8_t color = is_white ? 255 : 0;
      data[idx + 0] = color;  // R
      data[idx + 1] = color;  // G
      data[idx + 2] = color;  // B
      data[idx + 3] = 255;    // A
    }
  }
}

void generate_color_bars_pattern(uint8_t* data, int width, int height) {
  // SMPTE color bars (7 bars)
  const uint8_t colors[7][3] = {
    {255, 255, 255},  // White
    {255, 255, 0},    // Yellow
    {0, 255, 255},    // Cyan
    {0, 255, 0},      // Green
    {255, 0, 255},    // Magenta
    {255, 0, 0},      // Red
    {0, 0, 255}       // Blue
  };
  
  int bar_width = width / 7;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 4;
      int bar_idx = x / bar_width;
      if (bar_idx >= 7) bar_idx = 6;
      
      data[idx + 0] = colors[bar_idx][0];  // R
      data[idx + 1] = colors[bar_idx][1];  // G
      data[idx + 2] = colors[bar_idx][2];  // B
      data[idx + 3] = 255;                 // A
    }
  }
}

holoscan::gxf::Entity generate_pattern_entity(int width, int height, int pattern, 
                                               nvidia::gxf::MemoryStorageType storage_type,
                                               holoscan::Allocator* allocator) {
  HOLOSCAN_LOG_DEBUG("Generating {}x{} pattern entity (type {}, storage: {})", 
                     width, height, pattern, 
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
                          static_cast<int32_t>(4)},  // RGBA channels
        nvidia::gxf::PrimitiveType::kUnsigned8,
        0,
        nvidia::gxf::ComputeTrivialStrides(
            nvidia::gxf::Shape{static_cast<int32_t>(height),
                              static_cast<int32_t>(width),
                              static_cast<int32_t>(4)},
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
  
  size_t buffer_size = width * height * 4;  // RGBA
  
  // For device memory, generate pattern in host buffer first, then copy to device
  if (storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
    // Allocate temporary host buffer
    std::vector<uint8_t> host_buffer(buffer_size);
    
    // Generate pattern in host buffer
    switch (pattern) {
      case 0:  // Animated gradient
        generate_gradient_pattern(host_buffer.data(), width, height);
        break;
      case 1:  // Animated checkerboard
        generate_checkerboard_pattern(host_buffer.data(), width, height);
        break;
      case 2:  // Color bars (SMPTE style)
        generate_color_bars_pattern(host_buffer.data(), width, height);
        break;
      default:
        generate_gradient_pattern(host_buffer.data(), width, height);
    }
    
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
    
    // Generate pattern based on type
    switch (pattern) {
      case 0:  // Animated gradient
        generate_gradient_pattern(data, width, height);
        break;
      case 1:  // Animated checkerboard
        generate_checkerboard_pattern(data, width, height);
        break;
      case 2:  // Color bars (SMPTE style)
        generate_color_bars_pattern(data, width, height);
        break;
      default:
        generate_gradient_pattern(data, width, height);
    }
  }
  
  // Wrap the GXF entity in a Holoscan entity and return
  return holoscan::gxf::Entity(std::move(gxf_entity.value()));
}

}

namespace holoscan {

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
    spec.param(pattern_, "pattern", "Pattern", "Pattern type: 0=gradient, 1=checkerboard, 2=color bars", 0);
    spec.param(storage_type_, "storage_type", "StorageType", 
               "Memory storage type: 0=host, 1=device", 0);
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    HOLOSCAN_LOG_DEBUG("Generating pattern");
    
    // Convert storage_type parameter (0=host, 1=device) to MemoryStorageType enum
    nvidia::gxf::MemoryStorageType storage = (storage_type_.get() == 1) 
        ? nvidia::gxf::MemoryStorageType::kDevice 
        : nvidia::gxf::MemoryStorageType::kHost;
    
    // Generate a pattern entity with tensors
    auto entity = generate_pattern_entity(width_.get(), height_.get(), pattern_.get(), 
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
};

/**
 * @brief Holoscan application that pushes generated pattern data into GStreamer
 */
class GstVideoRecorderApp : public Application {
 public:
  GstVideoRecorderApp(int64_t iteration_count, int width, int height, 
            const std::string& framerate, int pattern, int storage_type, 
            const std::string& filename, const std::string& encoder,
            const std::map<std::string, std::string>& properties)
    : iteration_count_(iteration_count), width_(width), height_(height), 
      framerate_(framerate), pattern_(pattern), storage_type_(storage_type),
      filename_(filename), encoder_(encoder), properties_(properties) {}

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
        Arg("storage_type", storage_type_)
    );

    // Create the GStreamer video recorder operator - it manages the pipeline internally
    // Note: width, height, format, and storage type are automatically detected from incoming frames
    auto recorder_op = make_operator<GstVideoRecorderOperator>(
        "gst_recorder_op",
        Arg("encoder", encoder_),
        Arg("framerate", framerate_),
        Arg("properties", properties_),
        Arg("max-buffers", size_t(10)),
        Arg("timeout_ms", static_cast<uint64_t>(1000)),
        Arg("filename", filename_)
    );

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

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --count <number>     Number of frames to generate (default: unlimited)\n";
    std::cout << "  -w, --width <pixels>     Frame width (default: 1920)\n";
    std::cout << "  -h, --height <pixels>    Frame height (default: 1080)\n";
    std::cout << "  -f, --framerate <rate>   Frame rate as fraction or decimal (default: 30/1)\n";
    std::cout << "                            Examples: '30/1', '30000/1001', '29.97', '60'\n";
    std::cout << "                            Use '0/1' for live mode (no throttling, real-time timestamps)\n";
    std::cout << "  --pattern <type>         Pattern type: 0=gradient, 1=checkerboard, 2=color bars (default: 0)\n";
    std::cout << "  --storage <type>         Memory storage type: 0=host, 1=device/CUDA (default: 1)\n";
    std::cout << "  -o, --output <filename>  Output video filename (default: output.mp4)\n";
    std::cout << "                            Supported formats: .mp4, .mkv\n";
    std::cout << "                            If no extension, defaults to .mp4\n";
    std::cout << "  -e, --encoder <name>     Encoder base name (default: nvh264)\n";
    std::cout << "                            Examples: nvh264, nvh265, x264, x265\n";
    std::cout << "                            Note: 'enc' suffix is automatically appended\n";
    std::cout << "  --property <key=value>   Set encoder property (can be used multiple times)\n";
    std::cout << "                            Examples: --property bitrate=8000 --property preset=1\n";
    std::cout << "                            Property types are automatically detected and converted\n";
    std::cout << "  --help                   Show this help message\n\n";
    std::cout << "Pipeline:\n";
    std::cout << "  The application automatically detects video parameters and selects the appropriate converter.\n";
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
    std::cout << "    " << program_name << " --count 300 --width 1280 --height 720 --framerate 60 --output hd.mp4\n\n";
    std::cout << "  Record with NTSC framerate (exact 30000/1001):\n";
    std::cout << "    " << program_name << " --count 300 --framerate 30000/1001 --output ntsc.mp4\n\n";
    std::cout << "  Live mode (no throttling, real-time timestamps):\n";
    std::cout << "    " << program_name << " --count 300 --framerate 0/1 --output live.mp4\n\n";
    std::cout << "  Use CPU encoder (x264) with host memory:\n";
    std::cout << "    " << program_name << " --count 300 --storage 0 --encoder x264 --output cpu.mp4\n\n";
    std::cout << "  Custom encoder properties (bitrate, preset, GOP size):\n";
    std::cout << "    " << program_name << " --count 300 --property bitrate=8000 --property preset=1 --property gop-size=30 --output custom.mp4\n";
}

int main(int argc, char** argv) {
  int64_t iteration_count = INT64_MAX;  // Default: run forever
  std::string filename = "output.mp4";  // Default output file
  std::string encoder = "nvh264";        // Default encoder base name (enc suffix added by operator)
  int width = 1920;
  int height = 1080;
  std::string framerate = "30/1";
  int pattern = 0;  // 0=gradient, 1=checkerboard, 2=color bars
  int storage_type = 1;  // 0=host, 1=device (default to device for CUDA pipeline)
  std::map<std::string, std::string> properties;  // Encoder properties

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help") {
      print_usage(argv[0]);
      return 0;
    }
    else if ((arg == "-c" || arg == "--count") && i + 1 < argc) {
      iteration_count = std::stoll(argv[++i]);
    }
    else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
      width = std::stoi(argv[++i]);
    }
    else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
      height = std::stoi(argv[++i]);
    }
    else if ((arg == "-f" || arg == "--framerate") && i + 1 < argc) {
      framerate = argv[++i];
    }
    else if (arg == "--pattern" && i + 1 < argc) {
      pattern = std::stoi(argv[++i]);
    }
    else if (arg == "--storage" && i + 1 < argc) {
      storage_type = std::stoi(argv[++i]);
    }
    else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
      filename = argv[++i];
    }
    else if ((arg == "-e" || arg == "--encoder") && i + 1 < argc) {
      encoder = argv[++i];
    }
    else if (arg == "--property" && i + 1 < argc) {
      std::string prop = argv[++i];
      size_t eq_pos = prop.find('=');
      if (eq_pos != std::string::npos) {
        std::string key = prop.substr(0, eq_pos);
        std::string value = prop.substr(eq_pos + 1);
        properties[key] = value;
      } else {
        std::cerr << "Invalid property format (expected key=value): " << prop << std::endl;
        print_usage(argv[0]);
        return 1;
      }
    }
    else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  try {
    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Create the Holoscan application with parsed parameters
    auto holoscan_app = std::make_shared<holoscan::GstVideoRecorderApp>(
        iteration_count, width, height, framerate, pattern, storage_type, filename, encoder, properties);

    HOLOSCAN_LOG_INFO("Starting Holoscan Pattern to GStreamer Video Recorder");
    HOLOSCAN_LOG_INFO("Configuration: {} iterations, {}x{}@{}fps, pattern: {}, storage: {}, encoder: {}, output: '{}'", 
                      iteration_count, width, height, framerate, get_pattern_name(pattern), 
                      storage_type == 1 ? "device" : "host", encoder, filename);
    if (!properties.empty()) {
      HOLOSCAN_LOG_INFO("Encoder properties: {} properties configured", properties.size());
      for (const auto& [key, value] : properties) {
        HOLOSCAN_LOG_INFO("  {} = {}", key, value);
      }
    }
    HOLOSCAN_LOG_INFO("Video parameters (width, height, format, storage) will be auto-detected from frames");

    // Run the Holoscan application - the operator manages the GStreamer pipeline internally
    holoscan_app->run();
    
    HOLOSCAN_LOG_INFO("Application finished");

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }

  return 0;
}
