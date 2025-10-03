#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <cstdint>

#include <gst/gst.h>
#include <gst/cuda/gstcudamemory.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <gxf/std/tensor.hpp>
#include "../../operators/gstreamer/gst_sink_resource.hpp"

using namespace holoscan;
using namespace holoscan::gst;

// Convenient constant for mapping CUDA memory for reading
constexpr ::GstMapFlags GST_MAP_READ_CUDA = static_cast<::GstMapFlags>(GST_MAP_READ | GST_MAP_CUDA);

/**
 * @brief Convert CUDA memory type to DLPack device type
 * @param cuda_memory_type CUDA memory type from cudaPointerAttributes
 * @return Corresponding DLDeviceType
 */
DLDeviceType cuda_memory_type_to_dldevice_type(cudaMemoryType cuda_memory_type) {
  switch (cuda_memory_type) {
    case cudaMemoryTypeHost:
      return kDLCUDAHost;     // CUDA pinned memory
    case cudaMemoryTypeDevice:
      return kDLCUDA;         // True CUDA device memory
    case cudaMemoryTypeManaged:
      return kDLCUDAManaged;  // CUDA managed/unified memory
    case cudaMemoryTypeUnregistered:
      return kDLCPU;          // Regular host memory
    default:
      return kDLCPU;          // Unknown -> treat as CPU
  }
}

/**
 * @brief Convert GStreamer video format info to GXF PrimitiveType
 * @param video_format_info GStreamer video format info structure
 * @return Corresponding GXF PrimitiveType
 */
nvidia::gxf::PrimitiveType gstreamer_format_to_primitive_type(const GstVideoFormatInfo* video_format_info) {
  if (!video_format_info) {
    HOLOSCAN_LOG_WARN("Invalid video format info, defaulting to kUnsigned8");
    return nvidia::gxf::PrimitiveType::kUnsigned8;
  }

  // Calculate bytes per component from bits per component
  guint bits_per_component = video_format_info->depth[0];  // Bits per first component
  guint bytes_per_component = (bits_per_component + 7) / 8;  // Round up to nearest byte

  // Determine GXF primitive type based on bytes per component
  // This follows the same pattern as GXF VideoFormat to PrimitiveType conversion
  switch (bytes_per_component) {
    case 1:
      return nvidia::gxf::PrimitiveType::kUnsigned8;
    case 2:
      return nvidia::gxf::PrimitiveType::kUnsigned16;
    case 4:
      // For video formats, 4-byte components are typically unsigned integers
      // Float formats are rare in video and would need special handling
      return nvidia::gxf::PrimitiveType::kUnsigned32;
    case 8:
      return nvidia::gxf::PrimitiveType::kFloat64;
    default:
      HOLOSCAN_LOG_WARN("Unsupported component size: {} bytes, defaulting to kUnsigned8", 
                       bytes_per_component);
      return nvidia::gxf::PrimitiveType::kUnsigned8;
  }
}

/**
 * @brief Accurately detect CUDA memory type using CUDA API
 * @param ptr Pointer to check
 * @return DLDeviceType corresponding to the memory type
 */
DLDeviceType detect_cuda_memory_type(void* ptr) {
  if (!ptr) {
    return kDLCPU;  // Treat null pointer as CPU memory
  }

  cudaPointerAttributes attributes;
  cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
  
  if (result != cudaSuccess) {
    // Reset the CUDA error state
    cudaGetLastError();
    return kDLCPU;  // Treat non-CUDA memory as CPU memory
  }

  return cuda_memory_type_to_dldevice_type(attributes.type);
}


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
   * @brief Helper function to map GStreamer memory with the specified flags
   * @param memory GStreamer memory block to map
   * @param map_info Output map info structure
   * @param flags Mapping flags (GST_MAP_READ, GST_MAP_READ_CUDA, etc.)
   * @param mapped_device_type Output device type detected by CUDA API
   * @param storage_type Output GXF storage type based on device type
   * @param data_ptr Output pointer to mapped data
   * @param size Output size of mapped data
   * @return true if mapping succeeded, false otherwise
   */
  bool map_gst_memory(::GstMemory* memory, ::GstMapInfo& map_info, ::GstMapFlags flags,
                      DLDeviceType& mapped_device_type, nvidia::gxf::MemoryStorageType& storage_type,
                      void*& data_ptr, gsize& size) {
    if (!gst_memory_map(memory, &map_info, flags)) {
      return false;
    }
    
    // Use CUDA API to accurately detect the memory type
    mapped_device_type = detect_cuda_memory_type(map_info.data);
    
    // Set storage type based on actual device type
    if (mapped_device_type == kDLCUDA || mapped_device_type == kDLCUDAManaged) {
      storage_type = nvidia::gxf::MemoryStorageType::kDevice;
    } else if (mapped_device_type == kDLCUDAHost) {
      storage_type = nvidia::gxf::MemoryStorageType::kHost;  // CUDA pinned memory
    } else {
      storage_type = nvidia::gxf::MemoryStorageType::kSystem;  // Regular CPU memory
    }
    
    data_ptr = map_info.data;
    size = map_info.size;
    
    return true;
  }

  /**
   * @brief Create a Holoscan tensor from GStreamer buffer with optimal memory handling
   * @param buffer GStreamer buffer containing the data
   * @param caps GStreamer capabilities containing format and memory type information
   * @param context Execution context for GXF operations
   * @return Shared pointer to Holoscan tensor, or nullptr if creation fails
   */
  std::shared_ptr<holoscan::Tensor> create_tensor(const Buffer& buffer, 
                                                  const Caps& caps,
                                                  ExecutionContext& context) {
    try {
      auto video_info_opt = caps.get_video_info();
      if (!video_info_opt) {
        HOLOSCAN_LOG_ERROR("Cannot get video info from caps");
        return nullptr;
      }
      const VideoInfo& video_info = *video_info_opt;
      
      // Get dimensions and format info from video info
      int width = video_info->width;
      int height = video_info->height;
      auto format = video_info->finfo->format;
      auto plane_count = video_info->finfo->n_planes;
      auto total_size = video_info.get_total_size();
      
      // Extract format-specific information for tensor creation
      guint n_components = GST_VIDEO_INFO_N_COMPONENTS(video_info.get());
      guint bits_per_component = GST_VIDEO_INFO_COMP_DEPTH(video_info.get(), 0);  // Bits per first component
      guint bytes_per_component = (bits_per_component + 7) / 8;  // Round up to nearest byte
      
      // Determine GXF primitive type from GStreamer format info
      nvidia::gxf::PrimitiveType primitive_type = gstreamer_format_to_primitive_type(video_info->finfo);
      
      // Get the first memory block from the GstBuffer
      ::GstMemory* memory = gst_buffer_peek_memory(buffer.get(), 0);
      if (!memory) {
        HOLOSCAN_LOG_ERROR("No memory found in GstBuffer");
        return nullptr;
      }

      // Unified mapping approach for both CUDA and CPU memory
      ::GstMapInfo map_info;
      void* data_ptr = nullptr;
      gsize size = 0;
      DLDeviceType mapped_device_type = kDLCPU;
      nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kSystem;
      
      // Extract information from caps
      bool caps_indicate_cuda_memory = caps.has_feature(GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY);
      HOLOSCAN_LOG_INFO("Tensor creation: {}x{} {} ({} components, {} bits/comp, {} planes, {} bytes total), caps_indicate_cuda_memory={}", 
                        width, height, gst_video_format_to_string(format),
                        n_components, bits_per_component, plane_count, total_size, caps_indicate_cuda_memory);
      
      bool memory_mapped = false;
      
      // Try CUDA mapping first if caps indicate CUDA memory
      if (caps_indicate_cuda_memory) {
        if (map_gst_memory(memory, map_info, GST_MAP_READ_CUDA, mapped_device_type, storage_type, data_ptr, size)) {
          memory_mapped = true;
        }
      }
      
      // If CUDA mapping failed or we're dealing with CPU memory, use host mapping
      if (!memory_mapped) {
        if (!map_gst_memory(memory, map_info, GST_MAP_READ, mapped_device_type, storage_type, data_ptr, size)) {
          HOLOSCAN_LOG_ERROR("Failed to map memory with CPU flags");
          return nullptr;
        }
      }
      
      // Unified logging for all mapping scenarios
      const char* mapping_result;
      if (caps_indicate_cuda_memory) {
        mapping_result = memory_mapped ? "✅ CUDA mapping succeeded" : "⚠️  CUDA mapping failed, using CPU fallback";
      } else {
        mapping_result = "✅ CPU mapping succeeded";
      }
      
      HOLOSCAN_LOG_INFO("{}: device_type={}, pointer={}, size={} bytes", 
                       mapping_result, static_cast<int>(mapped_device_type),
                       static_cast<void*>(map_info.data), map_info.size);
      
      HOLOSCAN_LOG_INFO("Buffer info: {}x{}, buffer_size={}", 
                        width, height, size);
      
      // Check if CUDA memory was mapped to CPU (zero-copy warning)
      if (mapped_device_type == kDLCUDA || mapped_device_type == kDLCUDAHost || mapped_device_type == kDLCUDAManaged) {
        if (!memory_mapped || !caps_indicate_cuda_memory) {
          HOLOSCAN_LOG_WARN("CUDA memory was mapped to CPU - true zero-copy not achieved (device_type={})", 
                           static_cast<int>(mapped_device_type));
        }
      }
      
      // Create a GXF entity to hold our tensor
      auto entity = holoscan::gxf::Entity::New(&context);
      
      // Create a GXF tensor with proper memory management
      auto gxf_tensor_result = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("image");
      if (!gxf_tensor_result) {
          HOLOSCAN_LOG_ERROR("Failed to add tensor to entity");
          gst_memory_unmap(memory, &map_info);
          return nullptr;
      }
      auto gxf_tensor = gxf_tensor_result.value();
      
      // Use actual GStreamer strides - must match the wrapped memory layout
      // GXF requires std::array<size_t, 8> for up to 8-dimensional tensors
      std::array<size_t, 8> tensor_strides{{
        static_cast<size_t>(video_info.get_plane_stride(0)),                    // Row stride from GStreamer
        static_cast<size_t>(video_info->finfo->pixel_stride[0]),                // Pixel stride from GStreamer
        static_cast<size_t>(GST_VIDEO_INFO_COMP_PSTRIDE(video_info.get(), 0)),  // Component stride from GStreamer
        0, 0, 0, 0, 0  // Unused dimensions (4-8) set to 0
      }};
      
      HOLOSCAN_LOG_INFO("Tensor strides: row={}, pixel={}, component={}", 
                        tensor_strides[0], tensor_strides[1], tensor_strides[2]);
      
      HOLOSCAN_LOG_INFO("Tensor format: {}x{}x{} components, primitive_type={}, element_size={} bytes", 
                        height, width, n_components, 
                        static_cast<int>(primitive_type), bytes_per_component);
          
      HOLOSCAN_LOG_INFO("Creating tensor with storage type: {} (caps_indicate_cuda_memory={}, device_type={})", 
                        (storage_type == nvidia::gxf::MemoryStorageType::kDevice) ? "kDevice" : 
                        (storage_type == nvidia::gxf::MemoryStorageType::kHost) ? "kHost" : "kSystem",
                        caps_indicate_cuda_memory, static_cast<int>(mapped_device_type));
          
      // Create deleter for memory unmapping - captures buffer to keep it alive
      // The Buffer object must outlive the mapped memory to prevent GStreamer reference counting issues
      std::function<nvidia::gxf::Expected<void>(void*)> deleter = [buffer, memory, map_info](void*) mutable {
        // Unmap the GstMemory for both CUDA and CPU paths
        gst_memory_unmap(memory, const_cast<::GstMapInfo*>(&map_info));
        // Buffer will be destroyed here when deleter is called, properly cleaning up references
        return nvidia::gxf::Success;
      };
      
      gxf_tensor->wrapMemory(
          nvidia::gxf::Shape({static_cast<int32_t>(height), static_cast<int32_t>(width), static_cast<int32_t>(n_components)}),
          primitive_type,
          nvidia::gxf::PrimitiveTypeSize(primitive_type),
          tensor_strides,
          storage_type,
          static_cast<uint8_t*>(data_ptr),
          deleter);
      
      // Convert GXF tensor to DLManagedTensorContext
      auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();
      if (!maybe_dl_ctx) {
          HOLOSCAN_LOG_ERROR("Failed to convert GXF tensor to DLManagedTensorContext");
          gst_memory_unmap(memory, &map_info);
          return nullptr;
      }
      
      // Create Holoscan tensor from DLManagedTensorContext
      auto tensor = std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value());
      
      // Log final tensor device information
      auto tensor_device = tensor->device();
      HOLOSCAN_LOG_INFO("Created tensor: device_type={}, device_id={}", 
                        static_cast<int>(tensor_device.device_type), tensor_device.device_id);
      
      return tensor;
      
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to create tensor: {}", e.what());
      return nullptr;
    }
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {

    // Demonstrate the new client-side buffer retrieval and analysis
    try {
      // Get a buffer asynchronously from the GStreamer pipeline (blocks until available)
      auto buffer_future = gst_sink_resource_.get()->get_buffer();
      // Wait for buffer with timeout to avoid hanging
      auto status = buffer_future.wait_for(std::chrono::seconds(5));
      if (status == std::future_status::timeout) {
        HOLOSCAN_LOG_ERROR("Timeout waiting for buffer - no data received in 5 seconds");
        HOLOSCAN_LOG_ERROR("This usually indicates a GStreamer pipeline linking issue");
        return;
      }
      
      Buffer buffer = buffer_future.get();

      // Client-side buffer counting
      buffer_count_++;

      // Get the current caps which contain all the information we need
      Caps caps = gst_sink_resource_.get()->get_caps();
      if (caps.is_empty()) {
        HOLOSCAN_LOG_ERROR("No caps available for buffer");
        return;
      }

      HOLOSCAN_LOG_INFO("Processing buffer {}", buffer_count_);
      
      // Create tensor using unified function that extracts all needed info from caps
      std::shared_ptr<holoscan::Tensor> tensor = create_tensor(buffer, caps, context);
      if (!tensor) {
        HOLOSCAN_LOG_ERROR("Failed to create tensor from buffer data");
        return;
      }
      output.emit(tensor, "output");
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
