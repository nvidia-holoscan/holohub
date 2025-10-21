#include <iostream>
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
            int framerate, int pattern, int storage_type, 
            const std::string& pipeline_desc)
    : iteration_count_(iteration_count), width_(width), height_(height), 
      framerate_(framerate), pattern_(pattern), storage_type_(storage_type),
      pipeline_desc_(pipeline_desc) {}

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
    auto recorder_op = make_operator<GstVideoRecorderOperator>(
        "gst_recorder_op",
        Arg("width", width_),
        Arg("height", height_),
        Arg("framerate", framerate_),
        Arg("format", std::string("RGBA")),
        Arg("storage_type", storage_type_),
        Arg("queue_limit", size_t(10)),
        Arg("timeout_ms", static_cast<uint64_t>(1000)),
        Arg("pipeline_desc", pipeline_desc_)
    );

    // Connect the operators: pattern generator -> GStreamer recorder
    add_flow(pattern_gen_op, recorder_op);
  }

 private:
  int64_t iteration_count_;
  int width_;
  int height_;
  int framerate_;
  int pattern_;
  int storage_type_;
  std::string pipeline_desc_;
};

}  // namespace holoscan

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --count <number>     Number of frames to generate (default: unlimited)\n";
    std::cout << "  -w, --width <pixels>     Frame width (default: 1920)\n";
    std::cout << "  -h, --height <pixels>    Frame height (default: 1080)\n";
    std::cout << "  -f, --framerate <fps>    Frame rate in frames per second (default: 30)\n";
    std::cout << "  --pattern <type>         Pattern type: 0=gradient, 1=checkerboard, 2=color bars (default: 0)\n";
    std::cout << "  --storage <type>         Memory storage type: 0=host, 1=device/CUDA (default: 0)\n";
    std::cout << "  -p, --pipeline <desc>    GStreamer pipeline description (default: videoconvert ! autovideosink)\n";
    std::cout << "                            IMPORTANT: Your pipeline MUST name the first element as 'first'\n";
    std::cout << "  --help                   Show this help message\n\n";
    std::cout << "Pipeline Requirements:\n";
    std::cout << "  - The first element in your pipeline MUST be named 'first'\n";
    std::cout << "  - GStreamer automatically handles dynamic linking within the pipeline\n\n";
    std::cout << "Examples:\n";
    std::cout << "  Display animated gradient (default):\n";
    std::cout << "    " << program_name << "\n\n";
    std::cout << "  Display checkerboard pattern:\n";
    std::cout << "    " << program_name << " --pattern 1\n\n";
    std::cout << "  Display color bars:\n";
    std::cout << "    " << program_name << " --pattern 2\n\n";
    std::cout << "  Generate pattern in CUDA device memory:\n";
    std::cout << "    " << program_name << " --storage 1\n\n";
    std::cout << "  Custom resolution and framerate:\n";
    std::cout << "    " << program_name << " --width 1280 --height 720 --framerate 60\n\n";
    std::cout << "  Save to file:\n";
    std::cout << "    " << program_name << " --count 300 --pipeline \"videoconvert name=first ! x264enc ! mp4mux ! filesink location=output.mp4\"\n\n";
    std::cout << "  Stream over network:\n";
    std::cout << "    " << program_name << " --pipeline \"videoconvert name=first ! x264enc ! rtph264pay ! udpsink host=127.0.0.1 port=5000\"\n";
}

int main(int argc, char** argv) {
  int64_t iteration_count = INT64_MAX;  // Default: run forever
  std::string pipeline_desc = "videoconvert name=first ! autovideosink";  // Default value
  int width = 1920;
  int height = 1080;
  int framerate = 30;
  int pattern = 0;  // 0=gradient, 1=checkerboard, 2=color bars
  int storage_type = 0;  // 0=host, 1=device

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
      framerate = std::stoi(argv[++i]);
    }
    else if (arg == "--pattern" && i + 1 < argc) {
      pattern = std::stoi(argv[++i]);
    }
    else if (arg == "--storage" && i + 1 < argc) {
      storage_type = std::stoi(argv[++i]);
    }
    else if ((arg == "-p" || arg == "--pipeline") && i + 1 < argc) {
      pipeline_desc = argv[++i];
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
        iteration_count, width, height, framerate, pattern, storage_type, pipeline_desc);

    HOLOSCAN_LOG_INFO("Starting Holoscan Pattern to GStreamer Video Recorder");
    HOLOSCAN_LOG_INFO("Configuration: {} iterations, {}x{}@{}fps, pattern: {}, storage: {}, pipeline: '{}'", 
                      iteration_count, width, height, framerate, get_pattern_name(pattern), 
                      storage_type == 1 ? "device" : "host", pipeline_desc);
    HOLOSCAN_LOG_INFO("This will generate pattern data from Holoscan and push it into GStreamer");

    // Run the Holoscan application - the operator manages the GStreamer pipeline internally
    holoscan_app->run();
    
    HOLOSCAN_LOG_INFO("Application finished");

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }

  return 0;
}
