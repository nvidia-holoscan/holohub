#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

#include <gst/gst.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <dlpack/dlpack.h>
#include <gxf/std/tensor.hpp>
#include "../../operators/gstreamer/gst_sink_resource.hpp"

using namespace holoscan;
using namespace holoscan::gst;

class GstSinkOperator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GstSinkOperator)

  GstSinkOperator() = default;

  void setup(OperatorSpec& spec) override {
    /// Add parameters to the operator spec
    spec.param(gst_sink_resource_, "gst_sink_resource", "GStreamerSink", "GStreamer sink resource object");
    spec.param(pipeline_desc_, "pipeline_desc", "Pipeline Description", 
               "GStreamer pipeline description", std::string("videotestsrc pattern=0 ! videoconvert"));
    
    /// Add output for video frames to Holoviz
    spec.output<holoscan::Tensor>("output");
  }

  void initialize() override {
    // Create an allocator for the operator
    allocator_ = fragment()->make_resource<UnboundedAllocator>("pool");
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_);

    // Call the base class initialize function
    Operator::initialize();

    // Ensure the GStreamer sink resource is provided and valid
    if (!gst_sink_resource_.get()) {
      throw std::runtime_error("GStreamer sink resource is not provided");
    }
    if (!gst_sink_resource_.get()->valid()) {
      throw std::runtime_error("GStreamer sink resource is not valid");
    }

    // Create pipeline and add our sink element
    std::string pipeline_str = pipeline_desc_.get();
    
    HOLOSCAN_LOG_INFO("Creating pipeline: {}", pipeline_str);

    // Parse the source pipeline and let GStreamer handle internal connections
    GError* error = nullptr;
    pipeline_ = make_gst_object_guard(gst_parse_launch(pipeline_str.c_str(), &error));
    if (error) {
      auto error_guard = make_gst_error_guard(error);
      HOLOSCAN_LOG_ERROR("Failed to parse pipeline: {}", error_guard->message);
      throw std::runtime_error("Failed to parse GStreamer pipeline description");
    }
    
    // Get the sink element from our resource
    GstElement* sink_element = gst_sink_resource_.get()->get_element();
    
    // Add our sink element to the pipeline
    gst_bin_add(GST_BIN(pipeline_.get()), sink_element);
    
    // Look for an element named "last" - user should name their final element as "last"
    GstElement *last_element = gst_bin_get_by_name(GST_BIN(pipeline_.get()), "last");
    
    if (last_element) {
      HOLOSCAN_LOG_INFO("Found user-specified last element: {} - linking directly to sink", gst_element_get_name(last_element));
      
      // Link directly: last_element -> sink_element
      if (gst_element_link(last_element, sink_element)) {
        HOLOSCAN_LOG_INFO("Successfully linked: {} -> sink", gst_element_get_name(last_element));
      } else {
        HOLOSCAN_LOG_ERROR("Failed to link {} to sink", gst_element_get_name(last_element));
        throw std::runtime_error("Failed to link pipeline to sink");
      }
      
      // Clean up the reference
      gst_object_unref(last_element);
    } else {
      HOLOSCAN_LOG_ERROR("Could not find element named 'last' in pipeline");
      HOLOSCAN_LOG_ERROR("Please name your final pipeline element as 'last', e.g.: 'videoconvert name=last'");
      throw std::runtime_error("Could not find element named 'last' to connect to sink");
    }
    HOLOSCAN_LOG_INFO("Pipeline created and connected successfully");

    HOLOSCAN_LOG_INFO("GstSinkOperator initialized successfully");
  }

  void start() override {
    Operator::start();

    // Start the GStreamer pipeline
    GstStateChangeReturn ret = gst_element_set_state(pipeline_.get(), GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      HOLOSCAN_LOG_ERROR("Failed to start GStreamer pipeline");
      throw std::runtime_error("Failed to start GStreamer pipeline");
    }

    HOLOSCAN_LOG_INFO("GStreamer pipeline started");
  }

  /**
   * @brief Create a Holoscan tensor from MappedBuffer using shared memory
   * @param mapped_buffer Shared pointer to MappedBuffer containing GStreamer data
   * @param height Image height
   * @param width Image width
   * @param context Execution context for GXF operations
   * @return Shared pointer to Holoscan tensor, or nullptr if creation fails
   */
  std::shared_ptr<holoscan::Tensor> create_tensor(std::shared_ptr<MappedBuffer> mapped_buffer,
                                                  int height, int width, 
                                                  ExecutionContext& context) {
    try {
      // Get the data from the MappedBuffer
      const guint8* data = mapped_buffer->data();
      gsize size = mapped_buffer->size();
      
      // Create a GXF entity to hold our tensor
      auto entity = holoscan::gxf::Entity::New(&context);
      
      // Create a GXF tensor with proper memory management
      auto gxf_tensor_result = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("image");
      if (!gxf_tensor_result) {
          HOLOSCAN_LOG_ERROR("Failed to add tensor to entity");
          return nullptr;
      }
      auto gxf_tensor = gxf_tensor_result.value();
      
      // Use the GStreamer buffer data directly without copying
      // Wrap the existing GStreamer memory in a GXF tensor
      gxf_tensor->wrapMemory(
          nvidia::gxf::Shape({height, width, 4}),
          nvidia::gxf::PrimitiveType::kUnsigned8,
          nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8),
          nvidia::gxf::ComputeTrivialStrides(
              nvidia::gxf::Shape({height, width, 4}), 
              nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8)),
          nvidia::gxf::MemoryStorageType::kSystem,
          const_cast<uint8_t*>(data),
          [mapped_buffer](void*) {
              // Custom deleter that keeps the MappedBuffer alive
              // The shared_ptr will automatically manage the lifetime
              return nvidia::gxf::Success;
          });
      
      // Convert GXF tensor to DLManagedTensorContext
      auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();
      if (!maybe_dl_ctx) {
          HOLOSCAN_LOG_ERROR("Failed to convert GXF tensor to DLManagedTensorContext");
          return nullptr;
      }
      
      // Create Holoscan tensor from DLManagedTensorContext
      return std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value());
      
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to create tensor: {}", e.what());
      return nullptr;
    }
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {

    // Demonstrate the new client-side buffer retrieval and analysis
    try {
      // Get a mapped buffer asynchronously from the GStreamer pipeline (blocks until available)
      auto mapped_buffer_future = gst_sink_resource_.get()->get_buffer();
      // Wait for buffer with timeout to avoid hanging
      auto status = mapped_buffer_future.wait_for(std::chrono::seconds(5));
      if (status == std::future_status::timeout) {
        HOLOSCAN_LOG_ERROR("Timeout waiting for buffer - no data received in 5 seconds");
        HOLOSCAN_LOG_ERROR("This usually indicates a GStreamer pipeline linking issue");
        return;
      }
      
      std::shared_ptr<MappedBuffer> mapped_buffer = std::make_shared<MappedBuffer>(mapped_buffer_future.get());

      // Client-side buffer counting
      buffer_count_++;

      // Get the VideoInfo from the MappedBuffer
      const VideoInfo& video_info = mapped_buffer->get_video_info();

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
      const guint8* buffer_data = mapped_buffer->data();
      gsize buffer_size = mapped_buffer->size();

      HOLOSCAN_LOG_DEBUG("Buffer data access: {} bytes mapped at address {}", 
          buffer_size, static_cast<const void*>(buffer_data));

      // Demonstrate plane-specific data access
      if (plane_count > 0) {
        const guint8* plane_0_data = mapped_buffer->get_plane_data(0);
        if (plane_0_data) {
          HOLOSCAN_LOG_DEBUG("Plane 0 data accessible at address {}", 
              static_cast<const void*>(plane_0_data));
        }
      }

      // Demonstrate accessing actual buffer data using MappedBuffer
      if (buffer_size >= 8) {
        HOLOSCAN_LOG_INFO("First 8 bytes: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                         buffer_data[0], buffer_data[1], buffer_data[2], buffer_data[3],
                         buffer_data[4], buffer_data[5], buffer_data[6], buffer_data[7]);
      }

      // Demonstrate new validation features (Phase 2 Step 6)
      bool is_valid = mapped_buffer->validate();
      HOLOSCAN_LOG_INFO("Buffer validation: {}", is_valid ? "PASS" : "FAIL");

      // Show detailed validation report (only for first few buffers to avoid spam)
      if (buffer_count_ <= 3) {
        std::string validation_report = mapped_buffer->get_validation_report();
        HOLOSCAN_LOG_INFO("Validation Report:\n{}", validation_report);
      }

      // Create a tensor from the buffer data using our helper function
      auto tensor = create_tensor(mapped_buffer, height, width, context);
      if (tensor) {
          output.emit(tensor, "output");
          HOLOSCAN_LOG_INFO("Emitted tensor to Holoviz: {}x{}x{} (using shared memory)", 
                            height, width, 4);
      } else {
          HOLOSCAN_LOG_ERROR("Failed to create tensor from buffer data");
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
    auto bus = make_gst_object_guard(gst_element_get_bus(pipeline_.get()));
    auto msg = make_gst_message_guard(gst_bus_pop_filtered(bus.get(), static_cast<GstMessageType>(
        GST_MESSAGE_ERROR | GST_MESSAGE_EOS)));

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
          break;
        default:
          break;
      }
      // Message is automatically cleaned up by the guard
    }

    // Bus is automatically cleaned up by the guard
  }

  void stop() override {
    // Log final buffer count processed by this client
    HOLOSCAN_LOG_INFO("GstSinkOperator processed {} buffers total", buffer_count_);

    if (pipeline_ && GST_IS_ELEMENT(pipeline_.get())) {
      gst_element_set_state(pipeline_.get(), GST_STATE_NULL);
      pipeline_.reset();  // Automatic cleanup via guard
    }
    Operator::stop();
  }


 private:
  Parameter<SinkResourcePtr> gst_sink_resource_;
  Parameter<std::string> pipeline_desc_;
  GstElementGuard pipeline_;
  uint32_t buffer_count_ = 0;  // Client-side buffer counting
  std::shared_ptr<UnboundedAllocator> allocator_;
};

/**
 * @brief Simple Holoscan application that uses SinkResource
 */
class GstSinkApp : public Application {
 public:
  GstSinkApp(int64_t iteration_count, const std::string& pipeline_desc, const std::string& caps)
    : iteration_count_(iteration_count), pipeline_desc_(pipeline_desc), caps_(caps) {}

  void compose() override {
    // Create the GStreamer sink resource for data bridging
    // Use the caps parameter from command line arguments
    auto gst_sink = make_resource<SinkResource>("holoscan_sink", 
        Arg("capabilities", caps_));

    // Create the operator that uses the sink
    auto gst_op = make_operator<GstSinkOperator>(
        "gst_sink_op",
        make_condition<CountCondition>(iteration_count_),
        Arg("gst_sink_resource", gst_sink),
        Arg("pipeline_desc", pipeline_desc_)
    );

    // Create Holoviz operator for video visualization
    // Note: Resolution will be determined dynamically from GStreamer pipeline
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        Arg("allocator", make_resource<UnboundedAllocator>("holoviz_allocator")),
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{
            ops::HolovizOp::InputSpec("", ops::HolovizOp::InputType::COLOR)
        })
    );

    // Add operators to the application
    add_operator(gst_op);
    add_operator(holoviz);

    // Connect GStreamer operator output to Holoviz input
    add_flow(gst_op, holoviz, {{"output", "receivers"}});
  }

 private:
  int64_t iteration_count_;
  std::string pipeline_desc_;
  std::string caps_;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --count <number>     Number of iterations to run (default: 300)\n";
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
  int64_t iteration_count = 300;  // Default value
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
  auto app = std::make_shared<GstSinkApp>(iteration_count, pipeline_desc, caps);

  HOLOSCAN_LOG_INFO("Starting Holoscan GStreamer Sink Example");
  HOLOSCAN_LOG_INFO("Configuration: {} iterations, pipeline: '{}', caps: '{}'", iteration_count, pipeline_desc, caps);
  HOLOSCAN_LOG_INFO("This will display video frames using our custom universal sink");

  app->run();

  HOLOSCAN_LOG_INFO("Application finished");

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }

  return 0;
}
