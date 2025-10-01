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
    assert(gst_sink_resource_.get());
    assert(gst_sink_resource_.get()->valid());

    // Create pipeline and add our sink element
    std::string pipeline_str = pipeline_desc_.get();
    
    // Automatically append RGBA conversion if not already present
    if (pipeline_str.find("format=RGBA") == std::string::npos) {
        pipeline_str += " ! video/x-raw,format=RGBA ! videoconvert";
        HOLOSCAN_LOG_INFO("Auto-appended RGBA conversion to pipeline");
    }
    
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

  /**
   * @brief Create a Holoscan tensor from raw data
   * @param data Raw data pointer
   * @param size Size of the data in bytes
   * @param height Image height
   * @param width Image width
   * @param context Execution context for GXF operations
   * @return Shared pointer to Holoscan tensor, or nullptr if creation fails
   */
  std::shared_ptr<holoscan::Tensor> create_tensor(const guint8* data, gsize size, 
                                                  int height, int width, 
                                                  ExecutionContext& context) {
    try {
      // Create a GXF entity to hold our tensor
      auto entity = holoscan::gxf::Entity::New(&context);
      
      // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
      auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                             allocator_->gxf_cid());
      if (!gxf_allocator) {
          HOLOSCAN_LOG_ERROR("Failed to create GXF allocator handle");
          return nullptr;
      }
      
      // Create a GXF tensor with proper memory management
      auto gxf_tensor_result = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("image");
      if (!gxf_tensor_result) {
          HOLOSCAN_LOG_ERROR("Failed to add tensor to entity");
          return nullptr;
      }
      auto gxf_tensor = gxf_tensor_result.value();
      
      // Reshape the tensor to match our image dimensions (HWC format)
      auto reshape_result = gxf_tensor->reshape<uint8_t>(
          nvidia::gxf::Shape({height, width, 4}), 
          nvidia::gxf::MemoryStorageType::kHost, 
          gxf_allocator.value());
      if (!reshape_result) {
          HOLOSCAN_LOG_ERROR("Failed to reshape tensor");
          return nullptr;
      }
      
      // Copy the data from source to the tensor
      std::memcpy(gxf_tensor->pointer(), data, size);
      
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

      // Demonstrate accessing actual buffer data using MappedBuffer
      if (buffer_size >= 8) {
        HOLOSCAN_LOG_INFO("First 8 bytes: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                         buffer_data[0], buffer_data[1], buffer_data[2], buffer_data[3],
                         buffer_data[4], buffer_data[5], buffer_data[6], buffer_data[7]);
      }

      // Demonstrate new validation features (Phase 2 Step 6)
      bool is_valid = mapped_buffer.validate();
      HOLOSCAN_LOG_INFO("Buffer validation: {}", is_valid ? "PASS" : "FAIL");

      // Show detailed validation report (only for first few buffers to avoid spam)
      if (buffer_count_ <= 3) {
        std::string validation_report = mapped_buffer.get_validation_report();
        HOLOSCAN_LOG_INFO("Validation Report:\n{}", validation_report);
      }

      // Create a tensor from the buffer data using our helper function
      auto tensor = create_tensor(buffer_data, buffer_size, height, width, context);
      if (tensor) {
          output.emit(tensor, "output");
          HOLOSCAN_LOG_INFO("Emitted tensor to Holoviz: {}x{}x{} ({} bytes)", 
                            height, width, 4, buffer_size);
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
  std::shared_ptr<UnboundedAllocator> allocator_;
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
    // Test with specific video caps to verify the caps validation works:
    auto gst_sink = make_resource<SinkResource>("holoscan_sink", 
        Arg("capabilities", "video/x-raw,format=(string){I420,YV12,RGBx,BGRx,RGBA,BGRA}"));
    
    // You can also use "ANY" for maximum flexibility:
    // auto gst_sink = make_resource<SinkResource>("holoscan_sink", 
    //     Arg("capabilities", "ANY"));

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
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --count <number>     Number of iterations to run (default: 300)\n";
    std::cout << "  -p, --pipeline <desc>    GStreamer pipeline description (default: videotestsrc pattern=0 ! videoconvert)\n";
    std::cout << "                            Note: RGBA conversion is automatically appended if not present\n";
    std::cout << "  -h, --help               Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --count 150 --pipeline \"videotestsrc pattern=1 ! videoconvert\"\n";
    std::cout << "  " << program_name << " -c 600 -p \"audiotestsrc ! audioconvert\"\n";
    std::cout << "  " << program_name << " --pipeline \"autovideosrc ! videoconvert\"  # Use camera\n";
    std::cout << "  " << program_name << " --pipeline \"souphttpsrc location=https://example.com/video.mp4 ! decodebin ! videoconvert\"  # Network video\n\n";
    std::cout << "Note: The SinkResource now supports configurable capabilities. You can modify the code to use specific caps like:\n";
    std::cout << "  - video/x-raw,format=RGBA (for raw video)\n";
    std::cout << "  - audio/x-raw (for raw audio)\n";
    std::cout << "  - ANY (default, for maximum flexibility)\n";
    std::cout << "Example: make_resource<SinkResource>(\"my_sink\", Arg(\"capabilities\", \"video/x-raw,format=RGBA\"))\n";
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
