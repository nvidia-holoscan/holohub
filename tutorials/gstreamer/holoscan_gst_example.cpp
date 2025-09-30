#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

#include <gst/gst.h>
#include <holoscan/holoscan.hpp>
#include "../../operators/gstreamer/gst_sink_resource.hpp"

using namespace holoscan;
using namespace holoscan::gst;

/**
 * @brief Simple Holoscan operator that uses the SinkResource
 */
class GstSinkOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GstSinkOperator)

  GstSinkOperator() = default;

  void setup(OperatorSpec& spec) override {
    /// Add parameters to the operator spec
    spec.param(gst_sink_resource_, "gst_sink_resource", "GStreamerSink", "GStreamer sink resource object");
    spec.param(pipeline_desc_, "pipeline_desc", "Pipeline Description", 
               "GStreamer pipeline description", std::string("videotestsrc pattern=0 ! videoconvert"));
  }

  void initialize() override {
    Operator::initialize();

    // Ensure the GStreamer sink resource is provided and valid
    assert(gst_sink_resource_.get());
    assert(gst_sink_resource_.get()->valid());

    // Create pipeline and add our sink element
    std::string pipeline_str = pipeline_desc_.get();
    HOLOSCAN_LOG_INFO("Creating pipeline: {}", pipeline_str);

    // Create the main pipeline
    pipeline_ = gst_pipeline_new("holoscan-pipeline");
    if (!pipeline_) {
      throw std::runtime_error("Failed to create GStreamer pipeline");
    }

    // Parse the pipeline description to create source elements
    GError* error = nullptr;
    GstElement* source_bin = gst_parse_bin_from_description(pipeline_str.c_str(), TRUE, &error);
    if (error) {
      HOLOSCAN_LOG_ERROR("Failed to parse pipeline: {}", error->message);
      g_error_free(error);
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
      throw std::runtime_error("Failed to parse GStreamer pipeline description");
    }

    // Get the sink element from our resource
    GstElement* sink_element = gst_sink_resource()->get_element();
    if (!sink_element) {
      gst_object_unref(source_bin);
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
      throw std::runtime_error("Failed to get sink element from resource");
    }

    // Add elements to pipeline
    gst_bin_add_many(GST_BIN(pipeline_), source_bin, sink_element, nullptr);

    // Link the source bin to our sink
    if (!gst_element_link(source_bin, sink_element)) {
      HOLOSCAN_LOG_ERROR("Failed to link pipeline elements to sink");
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
      throw std::runtime_error("Failed to link GStreamer pipeline elements");
    }

    HOLOSCAN_LOG_INFO("GstSinkOperator initialized successfully");
  }

  void start() override {
    Operator::start();

    // Start the GStreamer pipeline
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      HOLOSCAN_LOG_ERROR("Failed to start GStreamer pipeline");
      throw std::runtime_error("Failed to start GStreamer pipeline");
    }

    HOLOSCAN_LOG_INFO("GStreamer pipeline started");
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {

    // Demonstrate the new client-side buffer retrieval and analysis
    try {
      // Get a mapped buffer asynchronously from the GStreamer pipeline (blocks until available)
      auto mapped_buffer_future = gst_sink_resource()->get_buffer();
      MappedBuffer mapped_buffer = mapped_buffer_future.get(); // Blocks until buffer arrives

      // Client-side buffer counting
      buffer_count_++;

      // Get the VideoInfo from the MappedBuffer
      const VideoInfo& video_info = mapped_buffer.get_video_info();

      // Access GstVideoInfo directly through operator->()
      auto format = video_info->finfo->format;
      auto width = video_info->width;
      auto height = video_info->height;

      // Demonstrate raw data access capabilities
      auto plane_count = video_info->finfo->n_planes;
      auto total_size = video_info.get_total_size();

      HOLOSCAN_LOG_INFO("Buffer {}: {}x{} {} ({} planes, {} bytes total)", 
          buffer_count_, width, height,
          gst_video_format_to_string(format),
          plane_count, total_size);

      // Show plane information
      for (int i = 0; i < plane_count; i++) {
        auto plane_stride = video_info.get_plane_stride(i);
        auto plane_size = video_info.get_plane_size(i);
        HOLOSCAN_LOG_DEBUG("  Plane {}: stride={} bytes, size={} bytes", 
            i, plane_stride, plane_size);
      }

      // Demonstrate MappedBuffer for safe data access
      const guint8* buffer_data = mapped_buffer.data();
      gsize buffer_size = mapped_buffer.size();

      HOLOSCAN_LOG_DEBUG("Buffer data access: {} bytes mapped at address {}", 
          buffer_size, static_cast<const void*>(buffer_data));

      // Demonstrate plane-specific data access
      if (plane_count > 0) {
        const guint8* plane_0_data = mapped_buffer.get_plane_data(0);
        if (plane_0_data) {
          HOLOSCAN_LOG_DEBUG("Plane 0 data accessible at address {}", 
              static_cast<const void*>(plane_0_data));
        }
      }

      // No manual cleanup needed - GstCaps handles it automatically!

      // Demonstrate accessing actual buffer data using RAII mapping
      {
        // MapInfo map_info(buffer, GST_MAP_READ); // Removed - using MappedBuffer instead
        // if (map_info.is_mapped()) { // Removed - using MappedBuffer instead
          // HOLOSCAN_LOG_DEBUG("Mapped buffer: {} bytes at address {}",
          //                  map_info.size(), static_cast<void*>(map_info.data()));

          // Example: Show first few bytes of data (safe for any buffer type)
          // if (map_info.size() >= 8) {
          //   const guint8* data = map_info.data();
          //   HOLOSCAN_LOG_INFO("First 8 bytes: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
          //                    data[0], data[1], data[2], data[3],
          //                    data[4], data[5], data[6], data[7]);
          // }

          // In a real application, you would:
          // - Convert to OpenCV Mat for image processing
          // - Copy to CUDA memory for GPU processing
          // - Pass to neural networks for inference
          // - Write to files or network streams
          // Example pseudocode:
          // if (g_str_has_prefix(media_type, "video/")) {
          //   cv::Mat frame = gst_buffer_to_opencv_mat(map_info.data(), width, height, format);
          //   your_processing_function(frame);
          // }
        // } else {
        //   HOLOSCAN_LOG_WARN("Failed to map buffer data");
        // }
      } // GstMapInfo destructor automatically unmaps the buffer
      
      // Demonstrate accessing actual buffer data using MappedBuffer
      if (buffer_size >= 8) {
        HOLOSCAN_LOG_INFO("First 8 bytes: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                         buffer_data[0], buffer_data[1], buffer_data[2], buffer_data[3],
                         buffer_data[4], buffer_data[5], buffer_data[6], buffer_data[7]);
      }

      // In a real application, you would:
      // - Convert to OpenCV Mat for image processing
      // - Copy to CUDA memory for GPU processing
      // - Pass to neural networks for inference
      // - Write to files or network streams
      // Example pseudocode:
      // if (g_str_has_prefix(media_type, "video/")) {
      //   cv::Mat frame = gst_buffer_to_opencv_mat(mapped_buffer.data(), width, height, format);
      //   cv::imshow("Frame", frame);
      // }

    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Error processing buffer: {}", e.what());
    }

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
    // Log final buffer count processed by this client
    HOLOSCAN_LOG_INFO("GstSinkOperator processed {} buffers total", buffer_count_);

    if (pipeline_ && GST_IS_ELEMENT(pipeline_)) {
      gst_element_set_state(pipeline_, GST_STATE_NULL);
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
    }
    Operator::stop();
  }

 protected:
  SinkResourcePtr gst_sink_resource() { return gst_sink_resource_.get(); }

 private:
  Parameter<SinkResourcePtr> gst_sink_resource_;
  Parameter<std::string> pipeline_desc_;
  GstElement* pipeline_ = nullptr;
  uint32_t buffer_count_ = 0;  // Client-side buffer counting
};

/**
 * @brief Simple Holoscan application that uses SinkResource
 */
class GstSinkApp : public Application {
 public:
  GstSinkApp(int64_t iteration_count, const std::string& pipeline_desc)
    : iteration_count_(iteration_count), pipeline_desc_(pipeline_desc) {}

  void compose() override {
    // Create the GStreamer sink resource for data bridging
    auto gst_sink = make_resource<SinkResource>("gst_sink", "holoscan_sink");

    // Create the operator that uses the sink
    auto gst_op = make_operator<GstSinkOperator>(
        "gst_sink_op",
        make_condition<CountCondition>(iteration_count_),
        Arg("gst_sink_resource", gst_sink),
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
