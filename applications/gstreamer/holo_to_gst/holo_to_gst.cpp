#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>

#include <holoscan/holoscan.hpp>
#include <gst_src_resource.hpp>
#include <gst_src_op.hpp>
#include <gst/pipeline.hpp>
#include <gst_pipeline_bus_monitor.hpp>
#include "pattern_generator.hpp"

namespace holoscan {

/**
 * @brief Holoscan application that pushes generated pattern data into GStreamer
 */
class GstSrcApp : public Application {
 public:
  GstSrcApp(int64_t iteration_count, const std::string& caps, int width, int height, 
            int framerate, int pattern, int storage_type)
    : iteration_count_(iteration_count), caps_(caps), width_(width), height_(height), 
      framerate_(framerate), pattern_(pattern), storage_type_(storage_type) {}

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
        Arg("caps", full_caps));

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

    // Create the GStreamer source operator
    auto gst_src_op = make_operator<GstSrcOp>(
        "gst_src_op",
        Arg("gst_src_resource", holoscan_gst_src_)
    );

    // Connect the operators: pattern generator -> GStreamer source
    add_flow(pattern_gen_op, gst_src_op);
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
  std::shared_ptr<GstSrcResource> holoscan_gst_src_;
};

}  // namespace holoscan

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
    pipeline_ = holoscan::gst::Pipeline(GST_PIPELINE(gst_parse_launch(pipeline_desc_.c_str(), &error)));
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
      HOLOSCAN_LOG_INFO("GStreamer pipeline is PLAYING or transitioning to PLAYING");
    } else {
      HOLOSCAN_LOG_WARN("Pipeline not yet in PLAYING state (current: {}), will transition as data flows", 
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
    return bus_monitor_->get_future();
  }

private:
  std::string pipeline_desc_;
  holoscan::gst::Element src_element_;
  holoscan::gst::Pipeline pipeline_;
  std::unique_ptr<AppPipelineBusMonitor> bus_monitor_;
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
  std::string caps = "video/x-raw,format=RGBA";  // Default value
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
    auto holoscan_app = std::make_shared<holoscan::GstSrcApp>(
        iteration_count, caps, width, height, framerate, pattern, storage_type);

    HOLOSCAN_LOG_INFO("Starting Holoscan Pattern to GStreamer Example");
    HOLOSCAN_LOG_INFO("Configuration: {} iterations, {}x{}@{}fps, pattern: {}, storage: {}, pipeline: '{}', caps: '{}'", 
                      iteration_count, width, height, framerate, holoscan::get_pattern_name(pattern), 
                      storage_type == 1 ? "device" : "host", pipeline_desc, caps);
    HOLOSCAN_LOG_INFO("This will generate pattern data from Holoscan and push it into GStreamer");

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
    
    holoscan::gst::Element src_element = src_element_future.get();
    if (!src_element) {
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

