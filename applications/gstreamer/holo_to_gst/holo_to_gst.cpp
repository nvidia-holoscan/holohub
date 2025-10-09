#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <cstdint>
#include <vector>
#include <cmath>
#include <csignal>

#include <gst/gst.h>
#include <holoscan/holoscan.hpp>
#include "../../operators/gstreamer/gst_src_resource.hpp"

namespace {

const char* get_pattern_name(int pattern) {
  const char* pattern_names[] = {"animated gradient", "animated checkerboard", "color bars"};
  return (pattern >= 0 && pattern <= 2) ? pattern_names[pattern] : "unknown";
}

}

namespace holoscan {

/**
 * GstSrcOperator - Source operator that generates pattern data and pushes into GStreamer
 */
class GstSrcOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GstSrcOperator)

  void setup(OperatorSpec& spec) override {
    /// No input - this is a source operator that generates its own data
    
    /// Add parameters to the operator spec
    spec.param(gst_src_resource_, "gst_src_resource", "GStreamerSource", "GStreamer source resource object");
    spec.param(width_, "width", "Width", "Frame width in pixels", 1920);
    spec.param(height_, "height", "Height", "Frame height in pixels", 1080);
    spec.param(pattern_, "pattern", "Pattern", "Pattern type: 0=gradient, 1=checkerboard, 2=color bars", 0);
  }

  void start() override {
    frame_count_ = 0;
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("GstSrcOperator::compute() called - frame {}", frame_count_);
    
    // Generate a pattern buffer
    auto buffer = generate_pattern_buffer();
    if (buffer.size() == 0) {
      HOLOSCAN_LOG_ERROR("Failed to generate pattern buffer");
      return;
    }

    HOLOSCAN_LOG_DEBUG("Generated buffer of size {} bytes", buffer.size());

    // Push buffer into the GStreamer pipeline (non-blocking with future)
    auto push_future = gst_src_resource_.get()->push_buffer(std::move(buffer));
    
    HOLOSCAN_LOG_DEBUG("Buffer pushed to GstSrcResource");
    
    // Optionally wait for the buffer to be consumed
    // For now we just let it push asynchronously
    // push_future.wait();
    
    frame_count_++;
  }

 private:
  /**
   * Generate a GStreamer buffer with a test pattern (RGBA format)
   */
  gst::Buffer generate_pattern_buffer() {
    int width = width_.get();
    int height = height_.get();
    int pattern = pattern_.get();
    
    HOLOSCAN_LOG_DEBUG("Generating {}x{} pattern (type {})", width, height, pattern);
    
    // Allocate buffer for RGBA data (4 bytes per pixel)
    size_t buffer_size = width * height * 4;
    
    // Create GStreamer buffer
    ::GstBuffer* gst_buffer = gst_buffer_new_allocate(nullptr, buffer_size, nullptr);
    if (!gst_buffer) {
      HOLOSCAN_LOG_ERROR("Failed to allocate GStreamer buffer");
      return gst::Buffer();
    }
    
    // Map buffer for writing
    GstMapInfo map_info;
    if (!gst_buffer_map(gst_buffer, &map_info, GST_MAP_WRITE)) {
      HOLOSCAN_LOG_ERROR("Failed to map buffer for writing");
      gst_buffer_unref(gst_buffer);
      return gst::Buffer();
    }
    
    uint8_t* data = map_info.data;
    
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
    
    // Unmap buffer
    gst_buffer_unmap(gst_buffer, &map_info);
    
    // Set timestamps
    GstClockTime timestamp = frame_count_ * GST_SECOND / 30;  // 30 fps
    GST_BUFFER_PTS(gst_buffer) = timestamp;
    GST_BUFFER_DTS(gst_buffer) = timestamp;
    GST_BUFFER_DURATION(gst_buffer) = GST_SECOND / 30;
    
    return gst::Buffer(gst_buffer);
  }
  
  void generate_gradient_pattern(uint8_t* data, int width, int height) {
    float time_offset = frame_count_ * 0.02f;
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
    int square_size = 64 + static_cast<int>(32 * std::sin(frame_count_ * 0.05f));
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

  Parameter<GstSrcResourcePtr> gst_src_resource_;
  Parameter<int> width_;
  Parameter<int> height_;
  Parameter<int> pattern_;
  uint64_t frame_count_ = 0;
};

/**
 * @brief Holoscan application that pushes generated pattern data into GStreamer
 */
class GstSrcApp : public Application {
 public:
  GstSrcApp(int64_t iteration_count, const std::string& caps, int width, int height, int pattern)
    : iteration_count_(iteration_count), caps_(caps), width_(width), height_(height), pattern_(pattern) {}

  void compose() override {
    // Build caps string with actual width and height
    std::string full_caps = caps_ + ",width=" + std::to_string(width_) + 
                           ",height=" + std::to_string(height_) + 
                           ",framerate=30/1";
    
    // Create the GStreamer source resource for data bridging
    holoscan_gst_src_ = make_resource<GstSrcResource>("holoscan_src", 
        Arg("capabilities", full_caps));

    // Create the operator that generates pattern and pushes to GStreamer
    // Use PeriodicCondition to run at ~30 fps
    auto gst_op = make_operator<GstSrcOperator>(
        "gst_src_op",
        make_condition<CountCondition>(iteration_count_),
        make_condition<PeriodicCondition>("periodic", Arg("recess_period", std::string("33ms"))),  // ~30 fps
        Arg("gst_src_resource", holoscan_gst_src_),
        Arg("width", width_),
        Arg("height", height_),
        Arg("pattern", pattern_)
    );

    // Add operator to the application
    add_operator(gst_op);
  }

  std::shared_ptr<GstSrcResource> get_src_resource() const {
    return holoscan_gst_src_;
  }

 private:
  int64_t iteration_count_;
  std::string caps_;
  int width_;
  int height_;
  int pattern_;
  std::shared_ptr<GstSrcResource> holoscan_gst_src_;
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
    
    // Add source element to pipeline
    // Note: gst_bin_add() takes ownership by sinking the floating reference.
    // We need to add an extra ref so both the bin and our shared_ptr have their own references.
    gst_object_ref(src_element_.get());
    gst_bin_add(GST_BIN(pipeline_.get()), src_element_.get());
    
    // Find and link the "first" element
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
    
    // Start bus monitoring in a background thread
    stop_bus_monitor_ = false;
    bus_monitor_thread_ = std::thread([this]() {
      monitor_pipeline_bus();
    });
  }

  ~GStreamerApp() {
    // Stop bus monitoring
    stop_bus_monitor_ = true;
    if (bus_monitor_thread_.joinable()) {
      bus_monitor_thread_.join();
    }
    
    // Stop and cleanup pipeline
    if (pipeline_ && pipeline_.get() && GST_IS_ELEMENT(pipeline_.get())) {
      gst_element_set_state(pipeline_.get(), GST_STATE_NULL);
      pipeline_.reset();
    }
  }

  // Delete copy constructor and assignment
  GStreamerApp(const GStreamerApp&) = delete;
  GStreamerApp& operator=(const GStreamerApp&) = delete;

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
            stop_bus_monitor_ = true;
            // Signal the application to terminate gracefully
            std::raise(SIGINT);
            break;
          }
          case GST_MESSAGE_EOS:
            HOLOSCAN_LOG_INFO("End of stream reached");
            stop_bus_monitor_ = true;
            // Signal the application to terminate gracefully
            std::raise(SIGINT);
            break;
          case GST_MESSAGE_STATE_CHANGED: {
            // Only check state changes from the pipeline (not individual elements)
            if (GST_MESSAGE_SRC(msg.get()) == GST_OBJECT(pipeline_.get())) {
              GstState old_state, new_state, pending_state;
              gst_message_parse_state_changed(msg.get(), &old_state, &new_state, &pending_state);
              
              // If pipeline transitions to NULL unexpectedly, stop monitoring
              if (new_state == GST_STATE_NULL && old_state != GST_STATE_NULL) {
                HOLOSCAN_LOG_INFO("GStreamer window closed");
                stop_bus_monitor_ = true;
                // Signal the application to terminate gracefully
                std::raise(SIGINT);
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
  std::thread bus_monitor_thread_;
  std::atomic<bool> stop_bus_monitor_;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --count <number>     Number of frames to generate (default: unlimited)\n";
    std::cout << "  -w, --width <pixels>     Frame width (default: 1920)\n";
    std::cout << "  -h, --height <pixels>    Frame height (default: 1080)\n";
    std::cout << "  --pattern <type>         Pattern type: 0=gradient, 1=checkerboard, 2=color bars (default: 0)\n";
    std::cout << "  -p, --pipeline <desc>    GStreamer pipeline description (default: videoconvert ! autovideosink)\n";
    std::cout << "                            IMPORTANT: Your pipeline MUST name the first element as 'first'\n";
    std::cout << "  --caps <caps_string>     GStreamer capabilities string for the source (default: video/x-raw,format=RGBA)\n";
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
    std::cout << "  Custom resolution:\n";
    std::cout << "    " << program_name << " --width 1280 --height 720\n\n";
    std::cout << "  Save to file:\n";
    std::cout << "    " << program_name << " --count 300 --pipeline \"videoconvert name=first ! x264enc ! mp4mux ! filesink location=output.mp4\"\n\n";
    std::cout << "  Stream over network:\n";
    std::cout << "    " << program_name << " --pipeline \"videoconvert name=first ! x264enc ! rtph264pay ! udpsink host=127.0.0.1 port=5000\"\n";
}

int main(int argc, char** argv) {
  int64_t iteration_count = INT64_MAX;  // Default: run forever
  std::string pipeline_desc = "videoconvert name=first ! autovideosink";  // Default value
  std::string caps = "video/x-raw,format=RGBA";  // Default value
  int width = 1920;
  int height = 1080;
  int pattern = 0;  // 0=gradient, 1=checkerboard, 2=color bars

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
    else if (arg == "--pattern" && i + 1 < argc) {
      pattern = std::stoi(argv[++i]);
    }
    else if ((arg == "-p" || arg == "--pipeline") && i + 1 < argc) {
      pipeline_desc = argv[++i];
    }
    else if (arg == "--caps" && i + 1 < argc) {
      caps = argv[++i];
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
    auto holoscan_app = std::make_shared<holoscan::GstSrcApp>(iteration_count, caps, width, height, pattern);

    HOLOSCAN_LOG_INFO("Starting Holoscan GStreamer Source Example");
    HOLOSCAN_LOG_INFO("Configuration: {} iterations, {}x{}, pattern: {}, pipeline: '{}', caps: '{}'", 
                      iteration_count, width, height, get_pattern_name(pattern), pipeline_desc, caps);
    HOLOSCAN_LOG_INFO("This will generate pattern data and push it into GStreamer");

    // Run the Holoscan application asynchronously
    auto app_future = holoscan_app->run_async();

    // Wait for the source element to be initialized
    auto src_element_future = holoscan_app->get_src_resource()->get_gst_element();
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
    GStreamerApp gstreamer_app(pipeline_desc, src_element);
    
    // Wait for the application to complete (will be signaled if window closes)
    app_future.wait();
    holoscan_app.reset();

    HOLOSCAN_LOG_INFO("Application finished");

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }

  return 0;
}

