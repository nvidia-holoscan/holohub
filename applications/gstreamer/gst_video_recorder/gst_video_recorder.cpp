#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <cstdint>
#include <vector>
#include <cmath>
#include <csignal>

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
            int framerate, int pattern, int storage_type)
    : iteration_count_(iteration_count), width_(width), height_(height), 
      framerate_(framerate), pattern_(pattern), storage_type_(storage_type) {}

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

    // Create the GStreamer video recorder operator - it builds its own caps string
    recorder_op_ = make_operator<GstVideoRecorderOperator>(
        "gst_recorder_op",
        Arg("width", width_),
        Arg("height", height_),
        Arg("framerate", framerate_),
        Arg("format", std::string("RGBA")),
        Arg("storage_type", storage_type_),
        Arg("queue_limit", size_t(10)),
        Arg("timeout_ms", static_cast<uint64_t>(1000))
    );

    // Connect the operators: pattern generator -> GStreamer recorder
    add_flow(pattern_gen_op, recorder_op_);
  }

  std::shared_ptr<GstVideoRecorderOperator> get_recorder_operator() const {
    return recorder_op_;
  }

 private:
  int64_t iteration_count_;
  int width_;
  int height_;
  int framerate_;
  int pattern_;
  int storage_type_;
  std::shared_ptr<GstVideoRecorderOperator> recorder_op_;
};

}  // namespace holoscan

/**
 * @brief GStreamerApp - Manages GStreamer pipeline lifecycle
 * 
 * Encapsulates all GStreamer pipeline operations for consuming from holoscansrc
 */
class GStreamerApp {
public:
  GStreamerApp(const std::string& pipeline_desc, 
               holoscan::gst::GstElementGuard src_element)
    : pipeline_desc_(pipeline_desc), 
      src_element_(src_element),
      stop_bus_monitor_(false) {
    HOLOSCAN_LOG_INFO("Setting up GStreamer pipeline");

    // Validate source element
    if (!src_element_ || !src_element_.get()) {
      throw std::runtime_error("Invalid source element");
    }
    
    // Parse the sink pipeline
    GError* error = nullptr;
    pipeline_ = holoscan::gst::make_gst_object_guard(gst_parse_launch(pipeline_desc_.c_str(), &error));
    if (error) {
      auto error_guard = holoscan::gst::make_gst_error_guard(error);
      HOLOSCAN_LOG_ERROR("Failed to parse pipeline: {}", error_guard->message);
      throw std::runtime_error("Failed to parse GStreamer pipeline description");
    }
    
    // Add sink element to pipeline
    // Note: gst_bin_add() takes ownership by sinking the floating reference (doesn't add a new ref).
    // Since our shared_ptr in GstSinkResource will call gst_object_unref() when destroyed,
    // we need to manually add a ref here so both the bin and our shared_ptr have their own references.
    // Without this: bin sinks the only ref → bin destroyed unrefs to 0 → GstSinkResource tries to unref freed memory.
    gst_object_ref(src_element_.get());
    gst_bin_add(GST_BIN(pipeline_.get()), src_element_.get());
    
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
    
    if (!gst_element_link(src_element_.get(), first_element.get())) {
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

    HOLOSCAN_LOG_INFO("GStreamer pipeline started");
    
    // Wait for pipeline to reach PLAYING state and be ready to accept data
    GstState state;
    ret = gst_element_get_state(pipeline_.get(), &state, nullptr, 2 * GST_SECOND);
    if (ret == GST_STATE_CHANGE_FAILURE || state != GST_STATE_PLAYING) {
      HOLOSCAN_LOG_ERROR("Pipeline failed to reach PLAYING state");
      throw std::runtime_error("Pipeline failed to reach PLAYING state");
    }
    HOLOSCAN_LOG_INFO("GStreamer pipeline is PLAYING and ready");

    // Start bus monitoring in a background thread
    bus_monitor_future_ = std::async(std::launch::async, [this]() { monitor_pipeline_bus(); });
  }

  ~GStreamerApp() {
    // Stop bus monitoring
    stop_bus_monitor_ = true;
    bus_monitor_future_.wait();
    
    // Stop and cleanup pipeline
    if (pipeline_ && pipeline_.get() && GST_IS_ELEMENT(pipeline_.get())) {
      gst_element_set_state(pipeline_.get(), GST_STATE_NULL);
      pipeline_.reset();
    }
  }

  // Delete copy constructor and assignment
  GStreamerApp(const GStreamerApp&) = delete;
  GStreamerApp& operator=(const GStreamerApp&) = delete;

  std::shared_future<void> get_bus_monitor_future() {
    return bus_monitor_future_;
  }

private:
  void monitor_pipeline_bus() {
    auto bus = holoscan::gst::make_gst_object_guard(gst_element_get_bus(pipeline_.get()));
    
    while (!stop_bus_monitor_) {
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
            if (GST_MESSAGE_SRC(msg.get()) == GST_OBJECT(pipeline_.get())) {
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

  std::string pipeline_desc_;
  holoscan::gst::GstElementGuard src_element_;
  holoscan::gst::GstElementGuard pipeline_;
  std::atomic<bool> stop_bus_monitor_;
  std::shared_future<void> bus_monitor_future_;
};

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
        iteration_count, width, height, framerate, pattern, storage_type);

    HOLOSCAN_LOG_INFO("Starting Holoscan Pattern to GStreamer Video Recorder");
    HOLOSCAN_LOG_INFO("Configuration: {} iterations, {}x{}@{}fps, pattern: {}, storage: {}, pipeline: '{}'", 
                      iteration_count, width, height, framerate, get_pattern_name(pattern), 
                      storage_type == 1 ? "device" : "host", pipeline_desc);
    HOLOSCAN_LOG_INFO("This will generate pattern data from Holoscan and push it into GStreamer");

    // Run the Holoscan application asynchronously
    auto app_future = holoscan_app->run_async();

    // Wait for the source element to be initialized
    auto src_element_future = holoscan_app->get_recorder_operator()->get_gst_element();
    if (!src_element_future.valid()) {
      throw std::runtime_error("Source element future is invalid");
    }
    
    // Wait for the element to be ready with timeout
    if (src_element_future.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
      throw std::runtime_error("Timeout waiting for source element initialization");
    }
    
    holoscan::gst::GstElementGuard src_element = src_element_future.get();
    if (!src_element || !src_element.get()) {
      throw std::runtime_error("Failed to get initialized source element");
    }

    // Create the GStreamer application with the source element and start it
    auto gstreamer_app = std::make_shared<GStreamerApp>(pipeline_desc, src_element);

    // Wait for Holoscan to finish generating frames
    app_future.wait();
    HOLOSCAN_LOG_INFO("Holoscan frame generation complete");
    
    // Wait for GStreamer to finish processing (EOS message on bus)
    HOLOSCAN_LOG_INFO("Waiting for GStreamer pipeline to finish");
    gstreamer_app->get_bus_monitor_future().wait();
    
    // Give the pipeline additional time to fully process EOS through all elements
    // The bus monitor returns when EOS is seen on the bus, but the pipeline needs
    // time to flush through encoder -> muxer -> filesink to finalize the output file
    HOLOSCAN_LOG_INFO("Allowing pipeline to finalize output file...");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    HOLOSCAN_LOG_INFO("Application finished");

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }

  return 0;
}
