#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <cstdint>

#include <gst/gst.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include "../../operators/gstreamer/gst_sink_resource.hpp"

using namespace holoscan;
using namespace holoscan::gst;

/**
 * GstSinkOperator - Operator for bridging GStreamer data into Holoscan
 */
class GstSinkOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GstSinkOperator)


  void setup(OperatorSpec& spec) override {
    /// Add output for video frames to Holoviz (Entity can contain multiple tensors/planes)
    spec.output<holoscan::gxf::Entity>("output");

    /// Add parameters to the operator spec
    spec.param(gst_sink_resource_, "gst_sink_resource", "GStreamerSink", "GStreamer sink resource object");
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    // Get a buffer asynchronously from the GStreamer pipeline (blocks until available)
    auto buffer_future = gst_sink_resource_.get()->get_buffer();
    // Wait for buffer with timeout to avoid hanging
    if (buffer_future.wait_for(std::chrono::seconds(1)) == std::future_status::timeout) {
      HOLOSCAN_LOG_ERROR("Timeout waiting for buffer - no data received in 1 seconds");
      return;
    }

    /// Get the buffer
    Buffer buffer = buffer_future.get();

    // Create entity with tensor(s) - supports both packed (RGBA) and planar (I420, NV12) formats
    auto entity = gst_sink_resource_.get()->create_entity_from_buffer(context, buffer);
    if (!entity) {
      HOLOSCAN_LOG_ERROR("Failed to create entity from buffer data");
      return;
    }
    output.emit(entity, "output");
  }

 private:
  Parameter<SinkResourcePtr> gst_sink_resource_;
};

/**
 * @brief Simple Holoscan application that uses SinkResource
 */
class GstSinkApp : public Application {
 public:
  GstSinkApp(int64_t iteration_count, const std::string& caps)
    : iteration_count_(iteration_count), caps_(caps) {}

  void compose() override {
    // Create the GStreamer sink resource for data bridging
    // Use the caps parameter from command line arguments
    gst_sink_ = make_resource<SinkResource>("holoscan_sink", 
        Arg("capabilities", caps_));

    // Create the operator that uses the sink
    auto gst_op = make_operator<GstSinkOperator>(
        "gst_sink_op",
        make_condition<CountCondition>(iteration_count_),
        Arg("gst_sink_resource", gst_sink_)
    );

    // Create Holoviz operator for video visualization
    // Note: Resolution will be determined dynamically from GStreamer pipeline
    // The input spec matches the tensor name "video_frame" from create_tensor_wrapper
    ops::HolovizOp::InputSpec video_input("video_frame", ops::HolovizOp::InputType::COLOR);
    // For YUV formats, set image_format_ appropriately:
    // video_input.image_format_ = ops::HolovizOp::ImageFormat::Y8_U8V8_3PLANE_420_UNORM;  // For I420
    // video_input.image_format_ = ops::HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM;  // For NV12
    
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        Arg("allocator", make_resource<UnboundedAllocator>("holoviz_allocator")),
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{video_input})
    );

    // Add operators to the application
    add_operator(gst_op);
    add_operator(holoviz);

    // Connect GStreamer operator output to Holoviz input
    add_flow(gst_op, holoviz, {{"output", "receivers"}});
  }

  std::shared_ptr<SinkResource> get_sink_resource() const {
    return gst_sink_;
  }

 private:
  int64_t iteration_count_;
  std::string caps_;
  std::shared_ptr<SinkResource> gst_sink_;
};

// Pipeline management helper functions
namespace {

// Forward declaration
void monitor_pipeline_bus(const GstElementGuard& pipeline,
                         std::atomic<bool>& stop_bus_monitor);

void setup_pipeline(const std::string& pipeline_desc, 
                   std::shared_ptr<SinkResource> gst_sink,
                   GstElementGuard& pipeline) {
    HOLOSCAN_LOG_INFO("Setting up GStreamer pipeline");
    
    // Wait for the sink element to be initialized
    auto sink_element_future = gst_sink->get_gst_element();
    if (!sink_element_future.valid()) {
      throw std::runtime_error("Sink element future is invalid");
    }
    
    // Block until the element is ready
    GstElementGuard sink_element = sink_element_future.get();
    if (!sink_element || !sink_element.get()) {
      throw std::runtime_error("Failed to get initialized sink element");
    }
    
    // Parse the source pipeline
    GError* error = nullptr;
    pipeline = make_gst_object_guard(gst_parse_launch(pipeline_desc.c_str(), &error));
    if (error) {
      auto error_guard = make_gst_error_guard(error);
      HOLOSCAN_LOG_ERROR("Failed to parse pipeline: {}", error_guard->message);
      throw std::runtime_error("Failed to parse GStreamer pipeline description");
    }
    
    // Add sink element to pipeline
    gst_bin_add(GST_BIN(pipeline.get()), sink_element.get());
    
    // Find and link the "last" element
    GstElement *last_element = gst_bin_get_by_name(GST_BIN(pipeline.get()), "last");
    if (!last_element) {
      HOLOSCAN_LOG_ERROR("Could not find element named 'last' in pipeline");
      HOLOSCAN_LOG_ERROR("Please name your final pipeline element as 'last', e.g.: 'videoconvert name=last'");
      throw std::runtime_error("Could not find element named 'last' to connect to sink");
    }
    
    HOLOSCAN_LOG_INFO("Linking {} to sink", gst_element_get_name(last_element));
    
    if (!gst_element_link(last_element, sink_element.get())) {
      gst_object_unref(last_element);
      HOLOSCAN_LOG_ERROR("Failed to link {} to sink", gst_element_get_name(last_element));
      throw std::runtime_error("Failed to link pipeline to sink");
    }
    
    gst_object_unref(last_element);
    HOLOSCAN_LOG_INFO("Pipeline setup complete");
}

void start_pipeline(const GstElementGuard& pipeline,
                   std::thread& bus_monitor_thread,
                   std::atomic<bool>& stop_bus_monitor) {
    if (!pipeline || !pipeline.get()) {
      throw std::runtime_error("Pipeline not created");
    }
    
    // Start the GStreamer pipeline
    GstStateChangeReturn ret = gst_element_set_state(pipeline.get(), GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      HOLOSCAN_LOG_ERROR("Failed to start GStreamer pipeline");
      throw std::runtime_error("Failed to start GStreamer pipeline");
    }
    
    HOLOSCAN_LOG_INFO("GStreamer pipeline started");
    
    // Start bus monitoring in a background thread
    stop_bus_monitor = false;
    bus_monitor_thread = std::thread([&pipeline, &stop_bus_monitor]() {
      monitor_pipeline_bus(pipeline, stop_bus_monitor);
    });
}

void monitor_pipeline_bus(const GstElementGuard& pipeline,
                         std::atomic<bool>& stop_bus_monitor) {
    auto bus = make_gst_object_guard(gst_element_get_bus(pipeline.get()));
    
    while (!stop_bus_monitor) {
      auto msg = make_gst_message_guard(
          gst_bus_timed_pop_filtered(bus.get(), 100 * GST_MSECOND,
              static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)));
      
      if (msg) {
        switch (GST_MESSAGE_TYPE(msg.get())) {
          case GST_MESSAGE_ERROR: {
            GError* error;
            gchar* debug_info;
            gst_message_parse_error(msg.get(), &error, &debug_info);
            auto error_guard = make_gst_error_guard(error);
            HOLOSCAN_LOG_ERROR("GStreamer error: {}", error_guard->message);
            if (debug_info) {
              HOLOSCAN_LOG_DEBUG("Debug info: {}", debug_info);
              g_free(debug_info);
            }
            break;
          }
          case GST_MESSAGE_EOS:
            HOLOSCAN_LOG_INFO("End of stream reached");
            stop_bus_monitor = true;
            break;
          default:
            break;
        }
      }
    }
}

void cleanup_pipeline(GstElementGuard& pipeline,
                     std::thread& bus_monitor_thread,
                     std::atomic<bool>& stop_bus_monitor) {
    // Stop bus monitoring
    stop_bus_monitor = true;
    if (bus_monitor_thread.joinable()) {
      bus_monitor_thread.join();
    }
    
    // Stop and cleanup pipeline
    if (pipeline && pipeline.get() && GST_IS_ELEMENT(pipeline.get())) {
      gst_element_set_state(pipeline.get(), GST_STATE_NULL);
      pipeline.reset();
    }
}

}  // namespace

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --count <number>     Number of iterations to run (default: unlimited)\n";
    std::cout << "  -p, --pipeline <desc>    GStreamer pipeline description (default: videotestsrc pattern=0 ! videoconvert name=last)\n";
    std::cout << "                            IMPORTANT: Your pipeline MUST name the final element as 'last'\n";
    std::cout << "  --caps <caps_string>     GStreamer capabilities string for the sink (default: video/x-raw)\n";
    std::cout << "  -h, --help               Show this help message\n\n";
    std::cout << "Pipeline Requirements:\n";
    std::cout << "  - The final element in your pipeline MUST be named 'last'\n";
    std::cout << "  - GStreamer automatically handles dynamic linking within the pipeline\n\n";
    std::cout << "Examples:\n";
    std::cout << "  Basic examples:\n";
    std::cout << "    " << program_name << " --pipeline \"videotestsrc pattern=0 ! videoconvert name=last\"\n";
    std::cout << "    " << program_name << " --pipeline \"filesrc location=video.mp4 ! qtdemux ! h264parse ! avdec_h264 ! videoconvert name=last\"\n\n";
    std::cout << "  Dynamic elements (uridecodebin, decodebin work automatically):\n";
    std::cout << "    " << program_name << " --caps \"video/x-raw,format=RGBA\" --pipeline \"uridecodebin uri=https://example.com/video.webm ! videoconvert ! capsfilter caps=video/x-raw,format=RGBA name=last\"\n";
    std::cout << "    " << program_name << " --pipeline \"souphttpsrc location=https://example.com/stream ! decodebin ! videoconvert name=last\"\n\n";
    std::cout << "  Audio examples:\n";
    std::cout << "    " << program_name << " --caps \"ANY\" --pipeline \"audiotestsrc ! audioconvert name=last\"\n";
}

int main(int argc, char** argv) {
  int64_t iteration_count = INT64_MAX;  // Default: run until video ends (unlimited)
  std::string pipeline_desc = "videotestsrc pattern=0 ! videoconvert name=last";  // Default value
  std::string caps = "video/x-raw";  // Default value

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    }
    else if ((arg == "-c" || arg == "--count") && i + 1 < argc) {
      iteration_count = std::stoll(argv[++i]);
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
    auto app = std::make_shared<GstSinkApp>(iteration_count, caps);

    HOLOSCAN_LOG_INFO("Starting Holoscan GStreamer Sink Example");
    HOLOSCAN_LOG_INFO("Configuration: {} iterations, pipeline: '{}', caps: '{}'", 
                      iteration_count, pipeline_desc, caps);
    HOLOSCAN_LOG_INFO("This will display video frames using our custom universal sink");

    // Compose the graph (creates resources and operators)
    app->compose_graph();

    // Setup the GStreamer pipeline (after resources are initialized)
    GstElementGuard pipeline;
    std::thread bus_monitor_thread;
    std::atomic<bool> stop_bus_monitor{false};
    
    // Run the Holoscan application asynchronously
    auto app_future = app->run_async();

    // Setup and start the GStreamer pipeline
    setup_pipeline(pipeline_desc, app->get_sink_resource(), pipeline);
    start_pipeline(pipeline, bus_monitor_thread, stop_bus_monitor);

    // Wait for the application to complete
    app_future.wait();

    // Cleanup
    cleanup_pipeline(pipeline, bus_monitor_thread, stop_bus_monitor);

    HOLOSCAN_LOG_INFO("Application finished");

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }

  return 0;
}
