/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdint>

#include <gst/gst.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <gst_sink_resource.hpp>
#include <gst_sink_op.hpp>
#include <gst/pipeline.hpp>
#include <gst_pipeline_bus_monitor.hpp>
#include "../common/arg_parser.hpp"

namespace {

/**
 * @brief Configuration parameters for the GStreamer to Holoscan bridge
 */
struct AppConfig {
  int64_t iteration_count = INT64_MAX;  // Default: run until video ends (unlimited)
  std::string pipeline_desc = "videotestsrc pattern=0 ! videoconvert name=sink";
  std::string caps = "video/x-raw,format=RGBA";
};

/**
 * @brief Print usage information for the application
 *
 * @param program_name Name of the program (argv[0])
 */
void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
  std::cout << "Options:\n";
  std::cout << "  -c, --count <number>     Number of iterations to run "
               "(default: unlimited)\n";
  std::cout << "  -p, --pipeline <desc>    GStreamer pipeline description\n";
  std::cout << "                            (default: videotestsrc pattern=0 ! "
               "videoconvert name=sink)\n";
  std::cout << "                            IMPORTANT: Your pipeline MUST name the "
               "final element as 'sink'\n";
  std::cout << "  --caps <caps_string>     GStreamer capabilities string for the sink\n";
  std::cout << "                            (default: video/x-raw,format=RGBA)\n";
  std::cout << "  -h, --help               Show this help message\n\n";
  std::cout << "Pipeline Requirements:\n";
  std::cout << "  - The final element in your pipeline MUST be named 'sink'\n";
  std::cout << "  - GStreamer automatically handles dynamic linking within "
               "the pipeline\n";
  std::cout << "  - The pipeline will be connected to a Holoscan sink for "
               "video display\n\n";
  std::cout << "File Input:\n";
  std::cout << "  - When running in container (via holohub), use absolute paths:\n";
  std::cout << "    location=/workspace/holohub/input.mp4\n";
  std::cout << "  - Files should be accessible in your workspace root\n\n";
  std::cout << "Examples:\n";
  std::cout << "  Display test pattern (default):\n";
  std::cout << "    " << program_name << "\n\n";
  std::cout << "  Display test pattern with count limit:\n";
  std::cout << "    " << program_name << " --count 300\n\n";
  std::cout << "  Display from MP4 file (CPU decode):\n";
  std::cout << "    " << program_name
               << " --pipeline \"filesrc location=/workspace/holohub/video.mp4 ! "
               "qtdemux ! h264parse ! avdec_h264 ! videoconvert name=sink\"\n\n";
  std::cout << "  Display from MP4 file (GPU decode):\n";
  std::cout << "    " << program_name
               << " --pipeline \"filesrc location=/workspace/holohub/video.mp4 ! "
               "qtdemux ! h264parse ! nvh264dec ! cudaconvert name=sink\" "
               "--caps \"video/x-raw(memory:CUDAMemory),format=RGBA\"\n\n";
  std::cout << "  Display from camera (V4L2):\n";
  std::cout << "    " << program_name
               << " --pipeline \"v4l2src device=/dev/video0 ! videoconvert name=sink\"\n\n";
  std::cout << "  Display from RTSP stream:\n";
  std::cout << "    " << program_name
               << " --pipeline \"rtspsrc location=rtsp://example.com/stream ! "
               "decodebin ! videoconvert name=sink\"\n\n";
  std::cout << "  Display with uridecodebin (automatic format detection):\n";
  std::cout << "    " << program_name
               << " --pipeline \"uridecodebin uri=file:///workspace/holohub/video.mp4 ! "
               "videoconvert name=sink\"\n";
}

/**
 * @brief Parse command-line arguments into application configuration
 *
 * @param argc Argument count
 * @param argv Argument values
 * @param config Output parameter for parsed configuration
 * @return true if parsing succeeded, false on error (prints error message internally)
 */
bool parse_arguments(int argc, char** argv, AppConfig& config) {
  using holoscan::gstreamer::common::parse_validated;

  try {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);
        std::exit(0);  // Exit successfully after showing help
      } else if ((arg == "-c" || arg == "--count") && i + 1 < argc) {
        // Validate: 1 to 1 billion frames
        config.iteration_count = parse_validated<int64_t>(argv[++i], "frame count", 1, 1000000000);
      } else if ((arg == "-p" || arg == "--pipeline") && i + 1 < argc) {
        config.pipeline_desc = argv[++i];
        if (config.pipeline_desc.empty()) {
          throw std::invalid_argument("Pipeline description cannot be empty");
        }
      } else if (arg == "--caps" && i + 1 < argc) {
        config.caps = argv[++i];
        if (config.caps.empty()) {
          throw std::invalid_argument("Caps string cannot be empty");
        }
      } else if ((arg == "-c" || arg == "--count" || arg == "-p" || arg == "--pipeline" ||
                  arg == "--caps") && i + 1 >= argc) {
        throw std::invalid_argument("Missing value for argument: " + arg);
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    return true;  // Success
  } catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl;
    print_usage(argv[0]);
    return false;  // Failure
  }
}

}  // namespace

namespace holoscan {

/**
 * @brief Simple Holoscan application that uses GstSinkResource
 *
 * Receives video frames from GStreamer and displays them using Holoviz
 */
class GstSinkApp : public Application {
 public:
  GstSinkApp(int64_t iteration_count, const std::string& caps)
    : iteration_count_(iteration_count), caps_(caps) {}

  void compose() override {
    // Create the GStreamer sink resource for data bridging
    // Use the caps parameter from command line arguments
    holoscan_gst_sink_ = make_resource<GstSinkResource>("holoscan_sink",
                                                         Arg("caps", caps_));

    // Create the operator that uses the sink
    auto gst_op = make_operator<GstSinkOp>(
        "gst_sink_op",
        make_condition<CountCondition>(iteration_count_),
        Arg("gst_sink_resource", holoscan_gst_sink_));

    // Create Holoviz operator for video visualization
    // Note: Resolution will be determined dynamically from GStreamer pipeline
    // The input spec matches the tensor name "video_frame" from create_tensor_wrapper
    ops::HolovizOp::InputSpec video_input("video_frame", ops::HolovizOp::InputType::COLOR);
    // For YUV formats, set image_format_ appropriately:
    // video_input.image_format_ = ops::HolovizOp::ImageFormat::Y8_U8V8_3PLANE_420_UNORM;  // I420
    // video_input.image_format_ = ops::HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM;  // NV12

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        Arg("allocator", make_resource<UnboundedAllocator>("holoviz_allocator")),
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{video_input}));

    // Add operators to the application
    add_operator(gst_op);
    add_operator(holoviz);

    // Connect GStreamer operator output to Holoviz input
    add_flow(gst_op, holoviz, {{"output", "receivers"}});
  }

  std::shared_ptr<GstSinkResource> get_sink_resource() const {
    return holoscan_gst_sink_;
  }

 private:
  int64_t iteration_count_;
  std::string caps_;
  std::shared_ptr<GstSinkResource> holoscan_gst_sink_;
};

}  // namespace holoscan

/**
 * @brief GStreamerApp - Manages GStreamer pipeline lifecycle
 *
 * Encapsulates all GStreamer pipeline operations for providing data to holoscan_sink
 */
class GStreamerApp {
 public:
  GStreamerApp(const std::string& pipeline_desc,
               holoscan::gst::Element sink_element)
    : pipeline_desc_(pipeline_desc),
      sink_element_(sink_element) {
    HOLOSCAN_LOG_INFO("Setting up GStreamer pipeline");

    // Validate sink element
    if (!sink_element_ || !sink_element_.get()) {
      throw std::runtime_error("Invalid sink element");
    }

    // Parse the source pipeline
    ::GError* error = nullptr;
    pipeline_ = holoscan::gst::Pipeline(GST_PIPELINE(
      gst_parse_launch(pipeline_desc_.c_str(), &error)));
    if (error) {
      auto error_object = holoscan::gst::Error(error);
      HOLOSCAN_LOG_ERROR("Failed to parse pipeline: {}", error_object->message);
      throw std::runtime_error("Failed to parse GStreamer pipeline description");
    }
    pipeline_.ref_sink();

    // Add sink element to pipeline
    pipeline_.add(sink_element_);

    // Find and link the "sink" element
    auto pipeline_sink = pipeline_.get_by_name("sink");
    if (!pipeline_sink) {
      HOLOSCAN_LOG_ERROR("Could not find element named 'sink' in pipeline");
      HOLOSCAN_LOG_ERROR("Please name your last pipeline element as 'sink', "
                         "e.g.: 'videoconvert name=sink'");
      throw std::runtime_error(
          "Could not find element named 'sink' to connect to appsink");
    }

    HOLOSCAN_LOG_INFO("Linking {} to appsink", pipeline_sink.get_name());

    if (!pipeline_sink.link(sink_element_)) {
      HOLOSCAN_LOG_ERROR("Failed to link {} to appsink",
                         pipeline_sink.get_name());
      throw std::runtime_error("Failed to link pipeline sink to appsink");
    }

    HOLOSCAN_LOG_INFO("Pipeline setup complete");

    // Start the GStreamer pipeline
    ::GstStateChangeReturn ret = pipeline_.set_state(GST_STATE_PLAYING);

    if (ret == GST_STATE_CHANGE_FAILURE) {
      // Get more details from the bus about what failed
      auto bus = pipeline_.get_bus();
      auto msg = bus.pop_filtered(
          static_cast<::GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));

      if (msg) {
        if (GST_MESSAGE_TYPE(msg.get()) == GST_MESSAGE_ERROR) {
          std::string debug_info;
          auto error = msg.parse_error(debug_info);
          HOLOSCAN_LOG_ERROR("GStreamer error: {} (source: {})",
                            error->message,
                            GST_MESSAGE_SRC_NAME(msg.get()));
          if (!debug_info.empty()) {
            HOLOSCAN_LOG_ERROR("Debug info: {}", debug_info);
          }
        }
      }

      HOLOSCAN_LOG_ERROR("Failed to start GStreamer pipeline");
      throw std::runtime_error("Failed to start GStreamer pipeline");
    }

    HOLOSCAN_LOG_INFO("GStreamer pipeline started");

    // Check pipeline state (informational - pipeline may not reach PLAYING until data flows)
    // This is expected for sink-based pipelines
    ::GstState state;
    auto state_result = pipeline_.get_state(&state, nullptr, 2 * GST_SECOND);
    if (state_result == GST_STATE_CHANGE_ASYNC || state == GST_STATE_PLAYING) {
      HOLOSCAN_LOG_INFO("GStreamer pipeline is PLAYING or transitioning to "
                        "PLAYING");
    } else {
      HOLOSCAN_LOG_INFO("Pipeline state: {} (will transition as data flows)",
                        gst_element_state_get_name(state));
    }

    // Create and start bus monitor
    bus_monitor_ = std::make_unique<holoscan::gst::PipelineBusMonitor>(pipeline_);
    bus_monitor_->start();
  }

  ~GStreamerApp() {
    HOLOSCAN_LOG_INFO("Stopping GStreamer pipeline");

    // Stop bus monitoring (will wait for thread to complete)
    if (bus_monitor_) {
      bus_monitor_->stop();
    }

    // Stop and cleanup pipeline
    if (pipeline_) {
      pipeline_.set_state(GST_STATE_NULL);
      pipeline_.reset();
    }

    HOLOSCAN_LOG_INFO("GStreamer pipeline cleanup completed");
  }

  // Delete copy constructor and assignment
  GStreamerApp(const GStreamerApp&) = delete;
  GStreamerApp& operator=(const GStreamerApp&) = delete;

 private:
  std::string pipeline_desc_;
  holoscan::gst::Element sink_element_;
  holoscan::gst::Pipeline pipeline_;
  std::unique_ptr<holoscan::gst::PipelineBusMonitor> bus_monitor_;
};


int main(int argc, char** argv) {
  // Parse command-line arguments
  AppConfig config;
  if (!parse_arguments(argc, argv, config)) {
    return 1;  // Error already printed by parse_arguments
  }

  try {
    // Initialize GStreamer
    gst_init(&argc, &argv);

    HOLOSCAN_LOG_INFO("Starting Holoscan GStreamer Sink Example");
    HOLOSCAN_LOG_INFO("Configuration: {} iterations, pipeline: '{}', caps: '{}'",
                      config.iteration_count, config.pipeline_desc, config.caps);
    HOLOSCAN_LOG_INFO("This will display video frames using our custom universal sink");

    // Create the Holoscan application with parsed configuration
    auto holoscan_app = std::make_shared<holoscan::GstSinkApp>(
        config.iteration_count, config.caps);

    // Run the Holoscan application asynchronously
    auto app_future = holoscan_app->run_async();

    // Wait for the sink element to be initialized
    auto sink_element_future = holoscan_app->get_sink_resource()->get_gst_element();
    if (!sink_element_future.valid()) {
      throw std::runtime_error("Sink element future is invalid");
    }

    // Wait for the element to be ready with timeout
    if (sink_element_future.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
      throw std::runtime_error("Timeout waiting for sink element initialization");
    }

    holoscan::gst::Element sink_element = sink_element_future.get();
    if (!sink_element || !sink_element.get()) {
      throw std::runtime_error("Failed to get initialized sink element");
    }

    HOLOSCAN_LOG_INFO("Sink element initialized successfully");

    // Create the GStreamer application with the sink element and start it
    GStreamerApp gstreamer_app(config.pipeline_desc, sink_element);

    // Wait for the application to complete
    HOLOSCAN_LOG_INFO("Application running - close the Holoviz window or press Ctrl+C to exit");
    app_future.wait();

    HOLOSCAN_LOG_INFO("Holoscan frame processing complete");
    HOLOSCAN_LOG_INFO("Application finished successfully");
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    return 1;
  }

  return 0;
}

