#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

#include <gst/gst.h>
#include <holoscan/holoscan.hpp>
#include "../../operators/gstreamer/core/gst_sink_resource.hpp"

using namespace holoscan;

/**
 * @brief Simple Holoscan operator that uses the GstSinkResource
 */
class GstSinkOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GstSinkOperator)

  GstSinkOperator() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(pipeline_desc_, "pipeline_desc", "Pipeline Description", 
               "GStreamer pipeline description", std::string("videotestsrc pattern=0 ! videoconvert"));
  }

  void initialize() override {
    Operator::initialize();

    // Create the GStreamer sink resource directly in the operator
    gst_sink_ = std::make_shared<GstSinkResource>("holoscan_sink", false, "/tmp", 30.0);
    gst_sink_->initialize();

    if (!gst_sink_ || !gst_sink_->valid()) {
      throw std::runtime_error("GstSinkResource not properly initialized");
    }

    // Create pipeline with the sink
    std::string pipeline_str = pipeline_desc_.get();
    HOLOSCAN_LOG_INFO("Creating pipeline: {}", pipeline_str);

    pipeline_ = gst_sink_->create_pipeline(pipeline_str);
    if (!pipeline_) {
      throw std::runtime_error("Failed to create GStreamer pipeline");
    }

    HOLOSCAN_LOG_INFO("GstSinkOperator initialized successfully");
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    // Start the pipeline if not already running
    static bool started = false;
    if (!started) {
      GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
      if (ret == GST_STATE_CHANGE_FAILURE) {
        HOLOSCAN_LOG_ERROR("Failed to start GStreamer pipeline");
        return;
      }
      started = true;
      HOLOSCAN_LOG_INFO("GStreamer pipeline started");
    }

    // In a real operator, you would process input data here
    // For this example, we just let the pipeline run and process one iteration

    // Sleep a bit to let the pipeline process data
    std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS

    // Check for EOS or errors
    GstBus* bus = gst_element_get_bus(pipeline_);
    GstMessage* msg = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(
        GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    if (msg) {
      switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
          GError* error;
          gchar* debug_info;
          gst_message_parse_error(msg, &error, &debug_info);
          HOLOSCAN_LOG_ERROR("GStreamer error: {}", error->message);
          if (debug_info) {
            HOLOSCAN_LOG_DEBUG("Debug info: {}", debug_info);
          }
          g_error_free(error);
          g_free(debug_info);
          break;
        }
        case GST_MESSAGE_EOS:
          HOLOSCAN_LOG_INFO("End of stream reached");
          break;
        default:
          break;
      }
      gst_message_unref(msg);
    }

    gst_object_unref(bus);
  }

  void stop() override {
    if (pipeline_ && GST_IS_ELEMENT(pipeline_)) {
      gst_element_set_state(pipeline_, GST_STATE_NULL);
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
    }
    Operator::stop();
  }

 private:
  Parameter<std::string> pipeline_desc_;
  GstElement* pipeline_ = nullptr;
  std::shared_ptr<GstSinkResource> gst_sink_;
};

/**
 * @brief Simple Holoscan application that uses GstSinkResource
 */
class GstSinkApp : public Application {
 public:
  GstSinkApp(int64_t iteration_count, const std::string& pipeline_desc)
    : iteration_count_(iteration_count), pipeline_desc_(pipeline_desc) {}

  void compose() override {
    // Create the operator that uses the sink
    auto gst_op = make_operator<GstSinkOperator>(
        "gst_sink_op",
        make_condition<CountCondition>(iteration_count_),
        Arg("pipeline_desc", pipeline_desc_)
    );

    // Add to the application
    add_operator(gst_op);
  }

 private:
  int64_t iteration_count_;
  std::string pipeline_desc_;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --count <number>     Number of iterations to run (default: 300)\n";
    std::cout << "  -p, --pipeline <desc>    GStreamer pipeline description (default: videotestsrc pattern=0 ! videoconvert)\n";
    std::cout << "  -h, --help               Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --count 150 --pipeline \"videotestsrc pattern=1 ! videoconvert\"\n";
    std::cout << "  " << program_name << " -c 600 -p \"audiotestsrc ! audioconvert\"\n";
    std::cout << "  " << program_name << " --pipeline \"autovideosrc ! videoconvert\"  # Use camera\n";
}

int main(int argc, char** argv) {
  int64_t iteration_count = 300;  // Default value
  std::string pipeline_desc = "videotestsrc pattern=0 ! videoconvert";  // Default value

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
  auto app = std::make_shared<GstSinkApp>(iteration_count, pipeline_desc);

  HOLOSCAN_LOG_INFO("Starting Holoscan GStreamer Sink Example");
  HOLOSCAN_LOG_INFO("Configuration: {} iterations, pipeline: '{}'", iteration_count, pipeline_desc);
  HOLOSCAN_LOG_INFO("This will display video frames using our custom universal sink");

  app->run();

  HOLOSCAN_LOG_INFO("Application finished");

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }

  return 0;
}
